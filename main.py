from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

from modules.storage import StorageGateway, StorageSettings
from modules.trader import (
    DryRunOrderExecutor,
    JupiterWatcher,
    LiveOrderExecutor,
    PairConfig,
    RuntimeConfig,
    TraderEngine,
)


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


STANDARD_LOG_FIELDS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key in STANDARD_LOG_FIELDS or key.startswith("_"):
                continue
            if key in {"message", "asctime"}:
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("solana_bot")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False

    return logger


async def wait_with_stop(stop_event: asyncio.Event, timeout_seconds: float) -> None:
    if timeout_seconds <= 0:
        return

    try:
        await asyncio.wait_for(stop_event.wait(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        pass


@dataclass(slots=True)
class AppSettings:
    watch_interval_seconds: float
    error_backoff_seconds: float
    jupiter_quote_api: str
    rpc_url: str
    private_key: str
    dry_run: bool

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            watch_interval_seconds=max(0.05, float(os.getenv("WATCH_INTERVAL_SECONDS", "1.0"))),
            error_backoff_seconds=max(0.2, float(os.getenv("ERROR_BACKOFF_SECONDS", "2.0"))),
            jupiter_quote_api=os.getenv("JUPITER_QUOTE_API", "https://quote-api.jup.ag/v6"),
            rpc_url=os.getenv("RPC_URL", ""),
            private_key=os.getenv("PRIVATE_KEY", ""),
            dry_run=_to_bool(os.getenv("DRY_RUN"), True),
        )


async def bootstrap_dependencies(
    *,
    logger: logging.Logger,
    stop_event: asyncio.Event,
    app_settings: AppSettings,
    storage: StorageGateway,
    watcher: JupiterWatcher,
    executor: DryRunOrderExecutor | LiveOrderExecutor,
    config_listener_loop: asyncio.AbstractEventLoop,
    on_config_update: Any,
) -> None:
    while not stop_event.is_set():
        try:
            await storage.connect()
            storage.start_config_listener(config_listener_loop, on_update=on_config_update)
            await watcher.connect()
            await executor.connect()
            return
        except Exception as error:
            logger.exception(
                "Dependency bootstrap failed",
                extra={"event": "bootstrap_error", "error": str(error)},
            )
            with contextlib.suppress(Exception):
                await storage.publish_event(
                    level="ERROR",
                    event="bootstrap_error",
                    message="Failed to initialize dependencies",
                    details={"error": str(error)},
                )
            with contextlib.suppress(Exception):
                await watcher.close()
            with contextlib.suppress(Exception):
                await executor.close()
            with contextlib.suppress(Exception):
                await storage.close()

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
        logger.info(
            "Order intake resumed after dependency recovery",
            extra={"event": "order_intake_recovered", "reason": pause_reason},
        )
        return True
    except Exception as error:
        logger.warning(
            "Order intake remains paused",
            extra={
                "event": "order_intake_still_paused",
                "reason": pause_reason,
                "error": str(error),
            },
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
        cycle_started = loop.time()

        try:
            if order_intake_paused:
                recovered = await try_resume_order_intake(
                    logger=logger,
                    storage=storage,
                    trader_engine=trader_engine,
                    pause_reason=pause_reason,
                )
                if not recovered:
                    with contextlib.suppress(Exception):
                        await storage.update_heartbeat()
                    continue

                order_intake_paused = False
                pause_reason = ""
                with contextlib.suppress(Exception):
                    await storage.record_position(
                        {
                            "pair": pair.symbol,
                            "status": "order_intake_resumed",
                            "reason": "dependencies_recovered",
                        }
                    )
                with contextlib.suppress(Exception):
                    await storage.publish_event(
                        level="INFO",
                        event="order_intake_resumed",
                        message="Order intake resumed after successful recovery",
                        details={"pair": pair.symbol},
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
                )

            if decision.blocked_by_fee_cap and decision.profitable:
                logger.info(
                    "Opportunity skipped by max_fee guard",
                    extra={
                        "event": "max_fee_guard_triggered",
                        "pair": pair.symbol,
                        "spread_bps": decision.spread_bps,
                        "required_spread_bps": decision.required_spread_bps,
                        "priority_fee_recommended": priority_fee_plan.recommended_micro_lamports,
                        "max_fee": priority_fee_plan.max_fee_micro_lamports,
                    },
                )

            if decision.profitable and not decision.should_execute and not decision.blocked_by_fee_cap:
                logger.info(
                    "Opportunity detected while trade is disabled",
                    extra={
                        "event": "opportunity_detected",
                        "pair": pair.symbol,
                        "spread_bps": decision.spread_bps,
                        "required_spread_bps": decision.required_spread_bps,
                    },
                )

        except Exception as error:
            pause_reason = str(error)
            order_intake_paused = True

            logger.exception(
                "Main loop failed and order intake has been paused",
                extra={"event": "main_loop_error", "error": str(error)},
            )
            with contextlib.suppress(Exception):
                await storage.record_position(
                    {
                        "pair": pair.symbol,
                        "status": "order_intake_paused",
                        "reason": pause_reason,
                    }
                )
            with contextlib.suppress(Exception):
                await storage.publish_event(
                    level="ERROR",
                    event="order_intake_paused",
                    message="Order intake paused due to dependency or runtime error",
                    details={"error": str(error), "pair": pair.symbol},
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

            await wait_with_stop(stop_event, delay_seconds)


async def main() -> None:
    load_dotenv()
    logger = setup_logger()

    app_settings = AppSettings.from_env()
    storage_settings = StorageSettings.from_env()
    pair = PairConfig.from_env()
    runtime_defaults = RuntimeConfig.from_env_defaults()

    storage = StorageGateway(storage_settings, logger)
    watcher = JupiterWatcher(logger=logger, api_base_url=app_settings.jupiter_quote_api)
    if app_settings.dry_run:
        executor: DryRunOrderExecutor | LiveOrderExecutor = DryRunOrderExecutor(
            logger=logger,
            order_store=storage,
        )
    else:
        executor = LiveOrderExecutor(
            logger=logger,
            rpc_url=app_settings.rpc_url,
            private_key=app_settings.private_key,
            order_store=storage,
        )

    trader_engine = TraderEngine(logger=logger, watcher=watcher, executor=executor)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_shutdown(sig: signal.Signals) -> None:
        logger.info(
            "Shutdown signal received",
            extra={"event": "shutdown_signal_received", "signal": sig.name},
        )
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, request_shutdown, sig)

    async def on_config_update(config: dict[str, Any]) -> None:
        logger.info(
            "Runtime config updated",
            extra={"event": "runtime_config_updated", "items": len(config)},
        )

    await bootstrap_dependencies(
        logger=logger,
        stop_event=stop_event,
        app_settings=app_settings,
        storage=storage,
        watcher=watcher,
        executor=executor,
        config_listener_loop=loop,
        on_config_update=on_config_update,
    )

    await storage.publish_event(
        level="INFO",
        event="bot_started",
        message="Bot process started",
        details={
            "pair": pair.symbol,
            "dry_run": app_settings.dry_run,
            "watch_interval_seconds": app_settings.watch_interval_seconds,
        },
    )

    try:
        await run_trading_loop(
            logger=logger,
            stop_event=stop_event,
            app_settings=app_settings,
            storage=storage,
            pair=pair,
            runtime_defaults=runtime_defaults,
            trader_engine=trader_engine,
        )
    finally:
        with contextlib.suppress(Exception):
            await storage.publish_event(
                level="INFO",
                event="bot_stopped",
                message="Bot process stopped gracefully",
            )

        with contextlib.suppress(Exception):
            await watcher.close()
        with contextlib.suppress(Exception):
            await executor.close()
        with contextlib.suppress(Exception):
            await storage.close()

        logger.info("Shutdown completed", extra={"event": "shutdown_completed"})


if __name__ == "__main__":
    asyncio.run(main())
