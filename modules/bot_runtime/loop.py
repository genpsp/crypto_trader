from __future__ import annotations

import asyncio
import logging
from collections import deque

from modules.common import guarded_call, log_event
from modules.storage import StorageGateway
from modules.trading import (
    HeliusRateLimitError,
    PairConfig,
    RuntimeConfig,
    TransactionPendingConfirmationError,
    TraderEngine,
)

from .loop_helpers import (
    bootstrap_dependencies,
    check_and_recover_pending_guard,
    compute_rate_limit_backoff_seconds,
    drawdown_pct_from_peak,
    estimate_execution_fee_lamports,
    prepare_execution_result_metadata,
    to_float,
    try_resume_order_intake,
    wait_with_stop,
)
from .settings import AppSettings


async def run_trading_loop(
    *,
    logger: logging.Logger,
    stop_event: asyncio.Event,
    app_settings: AppSettings,
    storage: StorageGateway,
    pair: PairConfig,
    runtime_defaults: RuntimeConfig,
    trader_engine: TraderEngine,
) -> None:
    loop = asyncio.get_running_loop()
    next_tick = loop.time()

    order_intake_paused = False
    pause_reason = ""
    recent_execution_timestamps: deque[float] = deque()
    recent_execution_fees: deque[tuple[float, int]] = deque()
    recent_execution_fee_total = 0
    recent_pnl_deltas: deque[tuple[float, float]] = deque()
    recent_pnl_total = 0.0
    last_execution_at = 0.0
    consecutive_execution_errors = 0
    execution_rate_limited_count = 0
    execution_circuit_open_until = 0.0
    drawdown_circuit_open_until = 0.0
    peak_wallet_balance_lamports: int | None = None
    final_stop_active = False
    next_pending_recovery_at = loop.time()

    async def open_execution_circuit(*, reason: str, error: str = "") -> None:
        nonlocal execution_circuit_open_until, consecutive_execution_errors

        cooldown_seconds = app_settings.live_execution_circuit_breaker_seconds
        execution_circuit_open_until = loop.time() + cooldown_seconds
        log_event(
            logger,
            level="warning",
            event="execution_circuit_breaker_opened",
            message="Execution circuit breaker opened after repeated execution errors",
            reason=reason,
            error=error,
            cooldown_seconds=cooldown_seconds,
            threshold=app_settings.live_max_consecutive_execution_errors,
        )
        await guarded_call(
            lambda: storage.record_position(
                {
                    "pair": pair.symbol,
                    "status": "execution_circuit_open",
                    "reason": reason,
                    "error": error,
                }
            ),
            logger=logger,
            event="execution_circuit_position_record_failed",
            message="Failed to record execution circuit breaker state",
        )
        await guarded_call(
            lambda: storage.publish_event(
                level="WARNING",
                event="execution_circuit_breaker_opened",
                message="Execution circuit breaker opened",
                details={
                    "pair": pair.symbol,
                    "reason": reason,
                    "error": error,
                    "cooldown_seconds": cooldown_seconds,
                    "threshold": app_settings.live_max_consecutive_execution_errors,
                },
            ),
            logger=logger,
            event="execution_circuit_publish_failed",
            message="Failed to publish execution circuit breaker event",
        )
        consecutive_execution_errors = 0

    def is_soft_execution_reject(error: Exception) -> bool:
        error_text = str(error).lower()
        return (
            "custom program error: 0x1788" in error_text
            or "insufficient funds" in error_text
        )

    def refresh_recent_pnl(now_time: float) -> float:
        nonlocal recent_pnl_total
        window_seconds = app_settings.live_drawdown_window_seconds
        while recent_pnl_deltas and (now_time - recent_pnl_deltas[0][0]) > window_seconds:
            _, expired_delta = recent_pnl_deltas.popleft()
            recent_pnl_total -= expired_delta
        return max(0.0, -recent_pnl_total)

    def record_realized_pnl(*, now_time: float, pnl_delta_lamports: float) -> None:
        nonlocal recent_pnl_total
        recent_pnl_deltas.append((now_time, pnl_delta_lamports))
        recent_pnl_total += pnl_delta_lamports

    async def open_drawdown_circuit(
        *,
        reason: str,
        current_drawdown_lamports: float,
        current_drawdown_pct: float | None = None,
    ) -> None:
        nonlocal drawdown_circuit_open_until

        cooldown_seconds = app_settings.live_drawdown_circuit_breaker_seconds
        drawdown_circuit_open_until = loop.time() + cooldown_seconds
        max_drawdown_pct = app_settings.live_max_drawdown_pct
        log_event(
            logger,
            level="warning",
            event="drawdown_circuit_breaker_opened",
            message="Drawdown circuit breaker opened",
            reason=reason,
            cooldown_seconds=cooldown_seconds,
            current_drawdown_lamports=round(current_drawdown_lamports, 4),
            max_drawdown_lamports=round(app_settings.live_max_drawdown_lamports, 4),
            current_drawdown_pct=round(current_drawdown_pct, 4) if current_drawdown_pct is not None else None,
            max_drawdown_pct=round(max_drawdown_pct, 4) if max_drawdown_pct > 0 else None,
        )
        await guarded_call(
            lambda: storage.record_position(
                {
                    "pair": pair.symbol,
                    "status": "drawdown_circuit_open",
                    "reason": reason,
                    "current_drawdown_lamports": round(current_drawdown_lamports, 4),
                    "current_drawdown_pct": (
                        round(current_drawdown_pct, 4) if current_drawdown_pct is not None else None
                    ),
                }
            ),
            logger=logger,
            event="drawdown_circuit_position_record_failed",
            message="Failed to record drawdown circuit breaker state",
        )
        await guarded_call(
            lambda: storage.publish_event(
                level="WARNING",
                event="drawdown_circuit_breaker_opened",
                message="Drawdown circuit breaker opened",
                details={
                    "pair": pair.symbol,
                    "reason": reason,
                    "current_drawdown_lamports": round(current_drawdown_lamports, 4),
                    "max_drawdown_lamports": round(app_settings.live_max_drawdown_lamports, 4),
                    "current_drawdown_pct": (
                        round(current_drawdown_pct, 4) if current_drawdown_pct is not None else None
                    ),
                    "max_drawdown_pct": round(max_drawdown_pct, 4) if max_drawdown_pct > 0 else None,
                    "cooldown_seconds": cooldown_seconds,
                },
            ),
            logger=logger,
            event="drawdown_circuit_publish_failed",
            message="Failed to publish drawdown circuit breaker event",
        )

    async def enforce_equity_guards(
        *,
        observation_forward_price: float,
        now_time: float,
    ) -> bool:
        nonlocal peak_wallet_balance_lamports, final_stop_active

        if app_settings.dry_run:
            return False

        if final_stop_active:
            return True

        needs_balance_check = (
            app_settings.live_max_drawdown_pct > 0
            or app_settings.live_final_stop_equity_usd > 0
        )
        if not needs_balance_check:
            return False

        balance_reader = getattr(trader_engine.executor, "get_wallet_balance_lamports", None)
        if not callable(balance_reader):
            return False

        wallet_balance_raw = await guarded_call(
            balance_reader,
            logger=logger,
            event="wallet_balance_fetch_failed",
            message="Failed to fetch wallet balance for equity guard checks",
            level="warning",
            default=None,
        )
        if wallet_balance_raw is None:
            return False

        wallet_balance_lamports = int(to_float(wallet_balance_raw, -1))
        if wallet_balance_lamports < 0:
            return False

        if peak_wallet_balance_lamports is None or wallet_balance_lamports > peak_wallet_balance_lamports:
            peak_wallet_balance_lamports = wallet_balance_lamports

        peak_balance = max(1, peak_wallet_balance_lamports or wallet_balance_lamports)
        balance_drawdown_lamports = max(0, peak_balance - wallet_balance_lamports)
        balance_drawdown_pct = drawdown_pct_from_peak(
            current_lamports=wallet_balance_lamports,
            peak_lamports=peak_balance,
        )

        if (
            app_settings.live_max_drawdown_pct > 0
            and balance_drawdown_pct >= app_settings.live_max_drawdown_pct
        ):
            if drawdown_circuit_open_until <= now_time:
                await open_drawdown_circuit(
                    reason="drawdown_pct_limit_exceeded",
                    current_drawdown_lamports=float(balance_drawdown_lamports),
                    current_drawdown_pct=balance_drawdown_pct,
                )
            return True

        if app_settings.live_final_stop_equity_usd <= 0:
            return False

        if observation_forward_price <= 0:
            log_event(
                logger,
                level="warning",
                event="final_equity_stop_price_unavailable",
                message="Final equity USD stop could not be evaluated because forward price was unavailable",
            )
            return False

        wallet_balance_units = wallet_balance_lamports / (10**pair.base_decimals)
        current_equity_usd = wallet_balance_units * observation_forward_price
        if current_equity_usd > app_settings.live_final_stop_equity_usd:
            return False

        final_stop_active = True
        log_event(
            logger,
            level="error",
            event="final_equity_stop_triggered",
            message="Final equity stop triggered; execution is now disabled until manual intervention",
            current_equity_usd=round(current_equity_usd, 6),
            final_stop_equity_usd=round(app_settings.live_final_stop_equity_usd, 6),
            wallet_balance_lamports=wallet_balance_lamports,
            peak_wallet_balance_lamports=peak_balance,
            current_drawdown_pct=round(balance_drawdown_pct, 6),
        )
        await guarded_call(
            lambda: storage.record_position(
                {
                    "pair": pair.symbol,
                    "status": "final_stop_active",
                    "reason": "equity_below_usd_threshold",
                    "current_equity_usd": round(current_equity_usd, 6),
                    "final_stop_equity_usd": round(app_settings.live_final_stop_equity_usd, 6),
                    "wallet_balance_lamports": wallet_balance_lamports,
                }
            ),
            logger=logger,
            event="final_equity_stop_position_record_failed",
            message="Failed to record final equity stop state",
        )
        await guarded_call(
            lambda: storage.publish_event(
                level="ERROR",
                event="final_equity_stop_triggered",
                message="Final equity stop triggered and execution has been disabled",
                details={
                    "pair": pair.symbol,
                    "current_equity_usd": round(current_equity_usd, 6),
                    "final_stop_equity_usd": round(app_settings.live_final_stop_equity_usd, 6),
                    "wallet_balance_lamports": wallet_balance_lamports,
                    "peak_wallet_balance_lamports": peak_balance,
                    "current_drawdown_pct": round(balance_drawdown_pct, 6),
                },
            ),
            logger=logger,
            event="final_equity_stop_publish_failed",
            message="Failed to publish final equity stop event",
        )
        return True

    while not stop_event.is_set():
        transient_backoff_seconds = 0.0

        try:
            if order_intake_paused:
                recovered = await try_resume_order_intake(
                    logger=logger,
                    storage=storage,
                    trader_engine=trader_engine,
                    pause_reason=pause_reason,
                )
                if not recovered:
                    await guarded_call(
                        storage.update_heartbeat,
                        logger=logger,
                        event="order_intake_paused_heartbeat_failed",
                        message="Failed to update heartbeat while intake is paused",
                    )
                    continue

                order_intake_paused = False
                pause_reason = ""
                await guarded_call(
                    lambda: storage.record_position(
                        {
                            "pair": pair.symbol,
                            "status": "order_intake_resumed",
                            "reason": "dependencies_recovered",
                        }
                    ),
                    logger=logger,
                    event="order_intake_resumed_position_record_failed",
                    message="Failed to record resumed position state",
                )
                await guarded_call(
                    lambda: storage.publish_event(
                        level="INFO",
                        event="order_intake_resumed",
                        message="Order intake resumed after successful recovery",
                        details={"pair": pair.symbol},
                        event_id=f"order_intake_resumed:{storage.run_id}",
                    ),
                    logger=logger,
                    event="order_intake_resumed_publish_failed",
                    message="Failed to publish order_intake_resumed event",
                )

            observation = await trader_engine.watcher.fetch_spread(pair)
            await storage.record_price(
                pair=pair.symbol,
                price=observation.forward_price,
                raw={
                    "forward_out_amount": observation.forward_out_amount,
                    "reverse_out_amount": observation.reverse_out_amount,
                    "spread_bps": observation.spread_bps,
                    "timestamp": observation.timestamp,
                },
            )

            redis_config = await storage.get_runtime_config()
            runtime_config = RuntimeConfig.from_redis(redis_config, runtime_defaults)
            priority_fee_plan = await trader_engine.resolve_priority_fee(runtime_config=runtime_config)

            now_time = loop.time()
            if not app_settings.dry_run and now_time >= next_pending_recovery_at:
                recover_pending = getattr(trader_engine.executor, "recover_pending", None)
                if callable(recover_pending):
                    await guarded_call(
                        recover_pending,
                        logger=logger,
                        event="pending_recovery_failed",
                        message="Failed to recover pending live executions",
                        level="warning",
                    )
                next_pending_recovery_at = now_time + app_settings.live_pending_recovery_interval_seconds

            decision = trader_engine.evaluate(
                observation=observation,
                runtime_config=runtime_config,
                pair=pair,
                priority_fee_plan=priority_fee_plan,
            )

            await storage.record_spread(
                pair=pair.symbol,
                spread_bps=decision.spread_bps,
                required_spread_bps=decision.required_spread_bps,
                total_fee_bps=decision.total_fee_bps,
                profitable=decision.profitable,
                extra={
                    "priority_fee_micro_lamports": decision.priority_fee_micro_lamports,
                    "priority_fee_source": priority_fee_plan.source,
                    "priority_fee_recommended_micro_lamports": priority_fee_plan.recommended_micro_lamports,
                    "priority_fee_sample_size": priority_fee_plan.sample_size,
                    "blocked_by_fee_cap": decision.blocked_by_fee_cap,
                    "tip_lamports": decision.tip_lamports,
                    "atomic_margin_bps": decision.atomic_margin_bps,
                    "expected_net_bps": decision.expected_net_bps,
                    "execution_mode": runtime_config.execution_mode,
                    "atomic_send_mode": runtime_config.atomic_send_mode,
                },
            )
            await storage.update_heartbeat()

            now_time = loop.time()
            equity_guard_blocked = await enforce_equity_guards(
                observation_forward_price=observation.forward_price,
                now_time=now_time,
            )
            if equity_guard_blocked:
                continue

            if decision.should_execute:
                if not app_settings.dry_run and app_settings.live_max_drawdown_lamports > 0:
                    current_drawdown_lamports = refresh_recent_pnl(now_time)
                    if current_drawdown_lamports >= app_settings.live_max_drawdown_lamports:
                        if drawdown_circuit_open_until <= now_time:
                            await open_drawdown_circuit(
                                reason="drawdown_limit_exceeded",
                                current_drawdown_lamports=current_drawdown_lamports,
                            )
                        continue

                if (not app_settings.dry_run) and drawdown_circuit_open_until > now_time:
                    remaining_seconds = max(0.0, drawdown_circuit_open_until - now_time)
                    log_event(
                        logger,
                        level="warning",
                        event="drawdown_circuit_open",
                        message="Execution skipped because drawdown circuit breaker is open",
                        remaining_seconds=round(remaining_seconds, 3),
                    )
                    continue

                if (not app_settings.dry_run) and execution_circuit_open_until > now_time:
                    remaining_seconds = max(0.0, execution_circuit_open_until - now_time)
                    log_event(
                        logger,
                        level="warning",
                        event="execution_circuit_open",
                        message="Execution skipped because circuit breaker is open",
                        remaining_seconds=round(remaining_seconds, 3),
                    )
                    continue

                cooldown_seconds = app_settings.live_execution_cooldown_seconds
                if cooldown_seconds > 0 and (now_time - last_execution_at) < cooldown_seconds:
                    log_event(
                        logger,
                        level="warning",
                        event="execution_cooldown_active",
                        message="Execution skipped due to cooldown guard",
                        cooldown_seconds=cooldown_seconds,
                        remaining_seconds=round(cooldown_seconds - (now_time - last_execution_at), 3),
                    )
                    continue

                window_seconds = app_settings.live_execution_window_seconds
                while recent_execution_timestamps and (now_time - recent_execution_timestamps[0]) > window_seconds:
                    recent_execution_timestamps.popleft()

                if len(recent_execution_timestamps) >= app_settings.live_max_executions_per_window:
                    log_event(
                        logger,
                        level="warning",
                        event="execution_rate_limited",
                        message="Execution skipped due to rate-limit guard",
                        window_seconds=window_seconds,
                        max_executions=app_settings.live_max_executions_per_window,
                        current_count=len(recent_execution_timestamps),
                    )
                    continue

                estimated_fee_lamports = 0
                if not app_settings.dry_run:
                    pending_guard_active = await check_and_recover_pending_guard(
                        logger=logger,
                        storage=storage,
                        trader_engine=trader_engine,
                        runtime_config=runtime_config,
                        app_settings=app_settings,
                    )
                    if pending_guard_active:
                        continue

                    tip_lamports = 0
                    if runtime_config.execution_mode == "atomic" and runtime_config.atomic_send_mode != "single_tx":
                        recommended_tip = max(0, runtime_config.jito_tip_lamports_recommended)
                        max_tip = max(0, runtime_config.jito_tip_lamports_max)
                        tip_lamports = min(recommended_tip, max_tip) if max_tip > 0 else recommended_tip

                    estimated_fee_lamports = estimate_execution_fee_lamports(
                        priority_fee_micro_lamports=priority_fee_plan.selected_micro_lamports,
                        priority_compute_units=runtime_config.priority_compute_units,
                        base_fee_lamports=app_settings.live_estimated_base_fee_lamports,
                        tip_lamports=tip_lamports,
                    )

                    fee_window_seconds = app_settings.live_fee_budget_window_seconds
                    while recent_execution_fees and (now_time - recent_execution_fees[0][0]) > fee_window_seconds:
                        _, expired_fee = recent_execution_fees.popleft()
                        recent_execution_fee_total = max(0, recent_execution_fee_total - expired_fee)

                    max_fee_per_window = app_settings.live_max_estimated_fee_lamports_per_window
                    if max_fee_per_window > 0 and (
                        recent_execution_fee_total + estimated_fee_lamports
                    ) > max_fee_per_window:
                        log_event(
                            logger,
                            level="warning",
                            event="execution_fee_budget_guard",
                            message="Execution skipped due to fee budget guard",
                            fee_window_seconds=fee_window_seconds,
                            estimated_fee_lamports=estimated_fee_lamports,
                            current_window_fee_lamports=recent_execution_fee_total,
                            max_window_fee_lamports=max_fee_per_window,
                            tip_lamports=tip_lamports,
                        )
                        continue

                idempotency_key = trader_engine.build_idempotency_key(pair=pair, observation=observation)
                try:
                    result = await trader_engine.execute(
                        pair=pair,
                        observation=observation,
                        runtime_config=runtime_config,
                        priority_fee_plan=priority_fee_plan,
                        idempotency_key=idempotency_key,
                    )
                except TransactionPendingConfirmationError as error:
                    pending_retry_after_seconds = max(
                        app_settings.live_confirm_poll_interval_seconds,
                        min(app_settings.live_pending_guard_ttl_seconds, app_settings.error_backoff_seconds),
                    )
                    transient_backoff_seconds = max(
                        transient_backoff_seconds,
                        pending_retry_after_seconds,
                    )
                    log_event(
                        logger,
                        level="warning",
                        event="execution_pending_confirmation",
                        message="Execution is pending confirmation; keeping intake active without incrementing error circuit",
                        error=str(error),
                        pending_retry_after_seconds=round(pending_retry_after_seconds, 3),
                        consecutive_errors=consecutive_execution_errors,
                    )
                    await guarded_call(
                        storage.update_heartbeat,
                        logger=logger,
                        event="execution_pending_confirmation_heartbeat_failed",
                        message="Failed to update heartbeat during pending confirmation backoff",
                    )
                    continue
                except Exception as error:
                    if is_soft_execution_reject(error):
                        log_event(
                            logger,
                            level="warning",
                            event="execution_soft_reject",
                            message=(
                                "Execution simulation was rejected due to insufficient funds in swap legs; "
                                "skipping without incrementing circuit breaker errors"
                            ),
                            error=str(error),
                            consecutive_errors=consecutive_execution_errors,
                        )
                        transient_backoff_seconds = max(
                            transient_backoff_seconds,
                            app_settings.live_confirm_poll_interval_seconds,
                        )
                        await guarded_call(
                            storage.update_heartbeat,
                            logger=logger,
                            event="execution_soft_reject_heartbeat_failed",
                            message="Failed to update heartbeat after soft execution reject",
                        )
                        continue

                    consecutive_execution_errors += 1
                    log_event(
                        logger,
                        level="warning",
                        event="execution_attempt_failed",
                        message="Execution attempt failed",
                        error=str(error),
                        consecutive_errors=consecutive_execution_errors,
                        threshold=app_settings.live_max_consecutive_execution_errors,
                    )
                    if (
                        not app_settings.dry_run
                        and consecutive_execution_errors >= app_settings.live_max_consecutive_execution_errors
                    ):
                        await open_execution_circuit(
                            reason="consecutive_execution_errors",
                            error=str(error),
                        )
                    continue

                if result.status == "rate_limited":
                    execution_rate_limited_count += 1
                    retry_after_seconds = to_float(result.metadata.get("retry_after_seconds"), 0.0)
                    adaptive_backoff_seconds = compute_rate_limit_backoff_seconds(
                        rate_limited_count=execution_rate_limited_count
                    )
                    transient_backoff_seconds = max(
                        transient_backoff_seconds,
                        retry_after_seconds,
                        adaptive_backoff_seconds,
                    )
                    log_event(
                        logger,
                        level="warning",
                        event="execution_rate_limited",
                        message="Execution skipped due to Jito rate limiting",
                        rate_limited_count=execution_rate_limited_count,
                        consecutive_errors=consecutive_execution_errors,
                        retry_after_seconds=retry_after_seconds if retry_after_seconds > 0 else None,
                        adaptive_backoff_seconds=round(adaptive_backoff_seconds, 3),
                        applied_backoff_seconds=round(transient_backoff_seconds, 3),
                        reason=result.reason,
                    )
                    await guarded_call(
                        storage.update_heartbeat,
                        logger=logger,
                        event="execution_rate_limited_heartbeat_failed",
                        message="Failed to update heartbeat during execution rate-limit backoff",
                    )
                else:
                    execution_rate_limited_count = 0

                if result.status == "failed":
                    consecutive_execution_errors += 1
                    if (
                        not app_settings.dry_run
                        and consecutive_execution_errors >= app_settings.live_max_consecutive_execution_errors
                    ):
                        await open_execution_circuit(
                            reason="execution_result_failed",
                            error=result.reason,
                        )
                else:
                    consecutive_execution_errors = 0

                is_non_execution_result = (
                    result.status.startswith("skipped_")
                    or result.status in {"rate_limited", "pending_confirmation"}
                )
                realized_pnl_lamports = to_float(result.metadata.get("pnl_delta"), 0.0)
                if (
                    not app_settings.dry_run
                    and realized_pnl_lamports == 0.0
                    and not is_non_execution_result
                    and estimated_fee_lamports > 0
                ):
                    realized_pnl_lamports = -float(estimated_fee_lamports)
                result_metadata, metrics_source, plan_expected_net_bps, requote_decay_bps = (
                    prepare_execution_result_metadata(
                        result_metadata=result.metadata if isinstance(result.metadata, dict) else None,
                        initial_expected_net_bps=decision.expected_net_bps,
                    )
                )
                if plan_expected_net_bps is not None:
                    if (
                        requote_decay_bps is not None
                        and requote_decay_bps <= -app_settings.live_requote_decay_warn_bps
                    ):
                        log_event(
                            logger,
                            level="warning",
                            event="requote_decay_detected",
                            message="Re-quote edge degraded compared with initial decision",
                            pair=pair.symbol,
                            initial_expected_net_bps=round(decision.expected_net_bps, 6),
                            plan_expected_net_bps=round(plan_expected_net_bps, 6),
                            requote_decay_bps=round(requote_decay_bps, 6),
                            plan_id=result_metadata.get("plan_id"),
                        )
                await storage.record_position(
                    {
                        "pair": pair.symbol,
                        "status": result.status,
                        "reason": result.reason,
                        "spread_bps": decision.spread_bps,
                        "required_spread_bps": decision.required_spread_bps,
                        "order_id": result.order_id,
                        "idempotency_key": result.idempotency_key,
                    }
                )
                await storage.record_trade(
                    trade={
                        "order_id": result.order_id,
                        "idempotency_key": result.idempotency_key,
                        "pair": pair.symbol,
                        "status": result.status,
                        "reason": result.reason,
                        "tx_signature": result.tx_signature,
                        "spread_bps": decision.spread_bps,
                        "required_spread_bps": decision.required_spread_bps,
                        "total_fee_bps": decision.total_fee_bps,
                        "priority_fee_micro_lamports": result.priority_fee_micro_lamports,
                        "config_schema_version": runtime_config.config_schema_version,
                        "metrics_source": metrics_source,
                        "pnl_delta": realized_pnl_lamports,
                        "metadata": result_metadata,
                    },
                    trade_id=result.order_id,
                )
                if storage.settings.firestore_publish_order_execution_events:
                    await storage.publish_event(
                        level="INFO",
                        event="order_execution",
                        message="Order execution attempted",
                        details={
                            "pair": pair.symbol,
                            "decision_reason": decision.reason,
                            "execution": result.to_dict(),
                            "priority_fee_plan": priority_fee_plan.to_dict(),
                        },
                        event_id=f"order_execution:{result.order_id}",
                    )
                if not is_non_execution_result:
                    execution_time = loop.time()
                    recent_execution_timestamps.append(execution_time)
                    last_execution_at = execution_time
                    if not app_settings.dry_run and estimated_fee_lamports > 0:
                        recent_execution_fees.append((execution_time, estimated_fee_lamports))
                        recent_execution_fee_total += estimated_fee_lamports
                    if not app_settings.dry_run and realized_pnl_lamports != 0.0:
                        record_realized_pnl(
                            now_time=execution_time,
                            pnl_delta_lamports=realized_pnl_lamports,
                        )
                        current_drawdown_lamports = refresh_recent_pnl(execution_time)
                        if (
                            app_settings.live_max_drawdown_lamports > 0
                            and current_drawdown_lamports >= app_settings.live_max_drawdown_lamports
                        ):
                            await open_drawdown_circuit(
                                reason="post_execution_drawdown_limit",
                                current_drawdown_lamports=current_drawdown_lamports,
                            )

            if decision.blocked_by_fee_cap and decision.profitable:
                log_event(
                    logger,
                    level="info",
                    event="max_fee_guard_triggered",
                    message="Opportunity skipped by max_fee guard",
                    pair=pair.symbol,
                    spread_bps=decision.spread_bps,
                    required_spread_bps=decision.required_spread_bps,
                    priority_fee_recommended=priority_fee_plan.recommended_micro_lamports,
                    max_fee=priority_fee_plan.max_fee_micro_lamports,
                )

            if decision.profitable and not decision.should_execute and not decision.blocked_by_fee_cap:
                log_event(
                    logger,
                    level="info",
                    event="opportunity_detected",
                    message="Opportunity detected while trade is disabled",
                    pair=pair.symbol,
                    spread_bps=decision.spread_bps,
                    required_spread_bps=decision.required_spread_bps,
                )

        except HeliusRateLimitError as error:
            transient_backoff_seconds = max(
                app_settings.error_backoff_seconds,
                error.retry_after_seconds or 0.0,
            )
            provider = getattr(error, "provider", "quote")
            log_event(
                logger,
                level="warning",
                event=f"{provider}_rate_limited",
                message=f"{provider.capitalize()} quote rate-limited; keeping order intake active and backing off",
                error=str(error),
                backoff_seconds=transient_backoff_seconds,
            )
            await guarded_call(
                storage.update_heartbeat,
                logger=logger,
                event="helius_rate_limited_heartbeat_failed",
                message="Failed to update heartbeat during rate-limit backoff",
            )
        except Exception as error:
            pause_reason = str(error)
            order_intake_paused = True

            log_event(
                logger,
                level="exception",
                event="main_loop_error",
                message="Main loop failed and order intake has been paused",
                error=str(error),
            )
            await guarded_call(
                lambda: storage.record_position(
                    {
                        "pair": pair.symbol,
                        "status": "order_intake_paused",
                        "reason": pause_reason,
                    }
                ),
                logger=logger,
                event="main_loop_pause_position_record_failed",
                message="Failed to record paused position state",
            )
            await guarded_call(
                lambda: storage.publish_event(
                    level="ERROR",
                    event="order_intake_paused",
                    message="Order intake paused due to dependency or runtime error",
                    details={"error": str(error), "pair": pair.symbol},
                ),
                logger=logger,
                event="main_loop_pause_publish_failed",
                message="Failed to publish order_intake_paused event",
            )
        finally:
            next_tick += app_settings.watch_interval_seconds
            now = loop.time()
            if next_tick <= now:
                missed_cycles = int((now - next_tick) / app_settings.watch_interval_seconds) + 1
                next_tick += missed_cycles * app_settings.watch_interval_seconds

            delay_seconds = max(0.0, next_tick - now)
            if order_intake_paused:
                delay_seconds = max(delay_seconds, app_settings.error_backoff_seconds)
            if transient_backoff_seconds > 0:
                delay_seconds = max(delay_seconds, transient_backoff_seconds)

            await wait_with_stop(stop_event, delay_seconds)


__all__ = [
    "bootstrap_dependencies",
    "run_trading_loop",
]
