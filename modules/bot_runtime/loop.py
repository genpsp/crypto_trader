from __future__ import annotations

import asyncio
import logging
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
                idempotency_key = trader_engine.build_idempotency_key(pair=pair, observation=observation)
                result = await trader_engine.execute(
                    pair=pair,
                    observation=observation,
                    runtime_config=runtime_config,
                    priority_fee_plan=priority_fee_plan,
                    idempotency_key=idempotency_key,
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
                        "pnl_delta": to_float(result.metadata.get("pnl_delta"), 0.0),
                        "metadata": result.metadata,
                    },
                    trade_id=result.order_id,
                )
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
