from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any

from modules.common import guarded_call, log_event
from modules.storage import ConfigUpdateHandler, StorageGateway
from modules.trading import (
    DryRunOrderExecutor,
    HeliusQuoteWatcher,
    HeliusRateLimitError,
    LiveOrderExecutor,
    PairConfig,
    RuntimeConfig,
    TraderEngine,
)

from .settings import AppSettings


def to_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


async def wait_with_stop(stop_event: asyncio.Event, timeout_seconds: float) -> None:
    if timeout_seconds <= 0:
        return

    try:
        await asyncio.wait_for(stop_event.wait(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        pass


def estimate_execution_fee_lamports(
    *,
    priority_fee_micro_lamports: int,
    priority_compute_units: int,
    base_fee_lamports: int,
) -> int:
    priority_fee_lamports = int((priority_fee_micro_lamports * priority_compute_units) / 1_000_000)
    return max(0, priority_fee_lamports) + max(0, base_fee_lamports)


async def bootstrap_dependencies(
    *,
    logger: logging.Logger,
    stop_event: asyncio.Event,
    app_settings: AppSettings,
    storage: StorageGateway,
    watcher: HeliusQuoteWatcher,
    executor: DryRunOrderExecutor | LiveOrderExecutor,
    config_listener_loop: asyncio.AbstractEventLoop,
    on_config_update: ConfigUpdateHandler,
) -> None:
    while not stop_event.is_set():
        try:
            await storage.connect()
            storage.start_config_listener(config_listener_loop, on_update=on_config_update)
            await watcher.connect()
            await executor.connect()
            return
        except Exception as error:
            log_event(
                logger,
                level="exception",
                event="bootstrap_error",
                message="Dependency bootstrap failed",
                error=str(error),
            )
            await guarded_call(
                lambda: storage.publish_event(
                    level="ERROR",
                    event="bootstrap_error",
                    message="Failed to initialize dependencies",
                    details={"error": str(error)},
                ),
                logger=logger,
                event="bootstrap_publish_error_failed",
                message="Failed to publish bootstrap error",
            )
            await guarded_call(
                watcher.close,
                logger=logger,
                event="bootstrap_watcher_close_failed",
                message="Failed to close watcher during bootstrap retry",
            )
            await guarded_call(
                executor.close,
                logger=logger,
                event="bootstrap_executor_close_failed",
                message="Failed to close executor during bootstrap retry",
            )
            await guarded_call(
                storage.close,
                logger=logger,
                event="bootstrap_storage_close_failed",
                message="Failed to close storage during bootstrap retry",
            )

            await wait_with_stop(stop_event, app_settings.error_backoff_seconds)

    raise RuntimeError("Shutdown requested before dependencies were initialized.")


async def try_resume_order_intake(
    *,
    logger: logging.Logger,
    storage: StorageGateway,
    trader_engine: TraderEngine,
    pause_reason: str,
) -> bool:
    try:
        await storage.healthcheck()
        await trader_engine.healthcheck()
        log_event(
            logger,
            level="info",
            event="order_intake_recovered",
            message="Order intake resumed after dependency recovery",
            reason=pause_reason,
        )
        return True
    except Exception as error:
        log_event(
            logger,
            level="warning",
            event="order_intake_still_paused",
            message="Order intake remains paused",
            reason=pause_reason,
            error=str(error),
        )
        return False


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
    execution_circuit_open_until = 0.0
    drawdown_circuit_open_until = 0.0

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
    ) -> None:
        nonlocal drawdown_circuit_open_until

        cooldown_seconds = app_settings.live_drawdown_circuit_breaker_seconds
        drawdown_circuit_open_until = loop.time() + cooldown_seconds
        log_event(
            logger,
            level="warning",
            event="drawdown_circuit_breaker_opened",
            message="Drawdown circuit breaker opened",
            reason=reason,
            cooldown_seconds=cooldown_seconds,
            current_drawdown_lamports=round(current_drawdown_lamports, 4),
            max_drawdown_lamports=round(app_settings.live_max_drawdown_lamports, 4),
        )
        await guarded_call(
            lambda: storage.record_position(
                {
                    "pair": pair.symbol,
                    "status": "drawdown_circuit_open",
                    "reason": reason,
                    "current_drawdown_lamports": round(current_drawdown_lamports, 4),
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
                    "cooldown_seconds": cooldown_seconds,
                },
            ),
            logger=logger,
            event="drawdown_circuit_publish_failed",
            message="Failed to publish drawdown circuit breaker event",
        )

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
                },
            )
            await storage.update_heartbeat()

            if decision.should_execute:
                now_time = loop.time()
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
                    pending_records = await storage.list_order_records(
                        statuses={"pending", "submitted", "pending_confirmation", "confirming"},
                        limit=app_settings.live_max_pending_orders + 1,
                    )
                    pending_count = len(pending_records)
                    if pending_count >= app_settings.live_max_pending_orders:
                        log_event(
                            logger,
                            level="warning",
                            event="execution_pending_guard",
                            message="Execution skipped due to pending order guard",
                            pending_count=pending_count,
                            max_pending_orders=app_settings.live_max_pending_orders,
                        )
                        continue

                    estimated_fee_lamports = estimate_execution_fee_lamports(
                        priority_fee_micro_lamports=priority_fee_plan.selected_micro_lamports,
                        priority_compute_units=runtime_config.priority_compute_units,
                        base_fee_lamports=app_settings.live_estimated_base_fee_lamports,
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
                except Exception as error:
                    if not app_settings.dry_run and estimated_fee_lamports > 0:
                        execution_time = loop.time()
                        recent_execution_timestamps.append(execution_time)
                        last_execution_at = execution_time
                        recent_execution_fees.append((execution_time, estimated_fee_lamports))
                        recent_execution_fee_total += estimated_fee_lamports
                        record_realized_pnl(
                            now_time=execution_time,
                            pnl_delta_lamports=-float(estimated_fee_lamports),
                        )
                        current_drawdown_lamports = refresh_recent_pnl(execution_time)
                        if (
                            app_settings.live_max_drawdown_lamports > 0
                            and current_drawdown_lamports >= app_settings.live_max_drawdown_lamports
                        ):
                            await open_drawdown_circuit(
                                reason="execution_exception_fee_loss",
                                current_drawdown_lamports=current_drawdown_lamports,
                            )
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

                realized_pnl_lamports = to_float(result.metadata.get("pnl_delta"), 0.0)
                if (
                    not app_settings.dry_run
                    and realized_pnl_lamports == 0.0
                    and result.status not in {"skipped_duplicate", "skipped_max_fee", "skipped_low_balance"}
                    and estimated_fee_lamports > 0
                ):
                    realized_pnl_lamports = -float(estimated_fee_lamports)
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
                        "pnl_delta": realized_pnl_lamports,
                        "metadata": result.metadata,
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
                if result.status not in {"skipped_duplicate", "skipped_max_fee", "skipped_low_balance"}:
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
