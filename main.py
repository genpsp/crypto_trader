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
            watch_interval_seconds=float(os.getenv("WATCH_INTERVAL_SECONDS", "1.0")),
            error_backoff_seconds=float(os.getenv("ERROR_BACKOFF_SECONDS", "2.0")),
            jupiter_quote_api=os.getenv("JUPITER_QUOTE_API", "https://quote-api.jup.ag/v6"),
            rpc_url=os.getenv("RPC_URL", ""),
            private_key=os.getenv("PRIVATE_KEY", ""),
            dry_run=_to_bool(os.getenv("DRY_RUN"), True),
        )


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
    while not stop_event.is_set():
        try:
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
            decision = trader_engine.evaluate(
                observation=observation,
                runtime_config=runtime_config,
                pair=pair,
            )

            await storage.record_spread(
                pair=pair.symbol,
                spread_bps=decision.spread_bps,
                required_spread_bps=decision.required_spread_bps,
                total_fee_bps=decision.total_fee_bps,
                profitable=decision.profitable,
            )
            await storage.update_heartbeat()

            if decision.should_execute:
                result = await trader_engine.execute(
                    pair=pair,
                    observation=observation,
                    runtime_config=runtime_config,
                )
                await storage.record_position(
                    {
                        "pair": pair.symbol,
                        "status": result.status,
                        "reason": result.reason,
                        "spread_bps": decision.spread_bps,
                        "required_spread_bps": decision.required_spread_bps,
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
                    },
                )

            if decision.profitable and not decision.should_execute:
                logger.info(
                    "Opportunity detected while trade is disabled",
                    extra={
                        "event": "opportunity_detected",
                        "pair": pair.symbol,
                        "spread_bps": decision.spread_bps,
                        "required_spread_bps": decision.required_spread_bps,
                    },
                )

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=app_settings.watch_interval_seconds)
            except asyncio.TimeoutError:
                pass
        except Exception as error:
            logger.exception(
                "Main loop failed",
                extra={"event": "main_loop_error", "error": str(error)},
            )
            await storage.publish_event(
                level="ERROR",
                event="main_loop_error",
                message="Unhandled exception in trading loop",
                details={"error": str(error)},
            )
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=app_settings.error_backoff_seconds)
            except asyncio.TimeoutError:
                pass


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
        executor = DryRunOrderExecutor(logger=logger)
    else:
        executor = LiveOrderExecutor(
            logger=logger,
            rpc_url=app_settings.rpc_url,
            private_key=app_settings.private_key,
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

    await storage.connect()
    storage.start_config_listener(loop, on_update=on_config_update)
    await watcher.connect()
    await executor.connect()

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
