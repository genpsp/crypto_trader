from __future__ import annotations

import asyncio
import contextlib
import signal
from typing import Any

from dotenv import load_dotenv

from modules.common import guarded_call, log_event
from modules.bot_runtime.logging import setup_logger
from modules.bot_runtime.loop import bootstrap_dependencies, run_trading_loop
from modules.bot_runtime.settings import AppSettings
from modules.storage import StorageGateway, StorageSettings
from modules.trading import (
    DryRunOrderExecutor,
    HeliusQuoteWatcher,
    LiveAtomicArbExecutor,
    LiveOrderExecutor,
    PairConfig,
    RuntimeConfig,
    TraderEngine,
)


async def main() -> None:
    load_dotenv()
    logger = setup_logger()

    app_settings = AppSettings.from_env()
    storage_settings = StorageSettings.from_env()
    pair = PairConfig.from_env()
    runtime_defaults = RuntimeConfig.from_env_defaults()

    storage = StorageGateway(storage_settings, logger)
    watcher = HeliusQuoteWatcher(
        logger=logger,
        api_base_url=app_settings.helius_quote_api,
        api_key=app_settings.helius_api_key or None,
        jupiter_api_key=app_settings.jupiter_api_key or None,
        enable_helius_jup_proxy=app_settings.helius_jup_proxy_enabled,
    )
    if app_settings.dry_run:
        executor: DryRunOrderExecutor | LiveOrderExecutor | LiveAtomicArbExecutor = DryRunOrderExecutor(
            logger=logger,
            order_store=storage,
        )
    else:
        if app_settings.execution_mode == "atomic":
            executor = LiveAtomicArbExecutor(
                logger=logger,
                rpc_url=app_settings.solana_rpc_url,
                private_key=app_settings.private_key,
                watcher=watcher,
                order_store=storage,
                swap_api_url=app_settings.jupiter_swap_api,
                jupiter_api_key=app_settings.jupiter_api_key or None,
                send_max_attempts=app_settings.live_send_max_attempts,
                send_retry_backoff_seconds=app_settings.live_send_retry_backoff_seconds,
                confirm_timeout_seconds=app_settings.live_confirm_timeout_seconds,
                confirm_poll_interval_seconds=app_settings.live_confirm_poll_interval_seconds,
                rebuild_max_attempts=app_settings.live_rebuild_max_attempts,
                pending_guard_ttl_seconds=app_settings.live_pending_guard_ttl_seconds,
                pending_recovery_limit=app_settings.live_pending_recovery_limit,
                min_balance_lamports=app_settings.live_min_balance_lamports,
                atomic_send_mode=app_settings.atomic_send_mode,
                atomic_expiry_ms=app_settings.atomic_expiry_ms,
                atomic_margin_bps=app_settings.atomic_margin_bps,
                jito_block_engine_url=app_settings.jito_block_engine_url,
                jito_tip_lamports_max=app_settings.jito_tip_lamports_max,
                jito_tip_lamports_recommended=app_settings.jito_tip_lamports_recommended,
            )
        else:
            executor = LiveOrderExecutor(
                logger=logger,
                rpc_url=app_settings.solana_rpc_url,
                private_key=app_settings.private_key,
                order_store=storage,
                swap_api_url=app_settings.jupiter_swap_api,
                jupiter_api_key=app_settings.jupiter_api_key or None,
                send_max_attempts=app_settings.live_send_max_attempts,
                send_retry_backoff_seconds=app_settings.live_send_retry_backoff_seconds,
                confirm_timeout_seconds=app_settings.live_confirm_timeout_seconds,
                confirm_poll_interval_seconds=app_settings.live_confirm_poll_interval_seconds,
                rebuild_max_attempts=app_settings.live_rebuild_max_attempts,
                pending_guard_ttl_seconds=app_settings.live_pending_guard_ttl_seconds,
                pending_recovery_limit=app_settings.live_pending_recovery_limit,
                min_balance_lamports=app_settings.live_min_balance_lamports,
            )

    trader_engine = TraderEngine(logger=logger, watcher=watcher, executor=executor)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_shutdown(sig: signal.Signals) -> None:
        log_event(
            logger,
            level="info",
            event="shutdown_signal_received",
            message="Shutdown signal received",
            signal=sig.name,
        )
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, request_shutdown, sig)

    async def on_config_update(config: dict[str, Any]) -> None:
        log_event(
            logger,
            level="info",
            event="runtime_config_updated",
            message="Runtime config updated",
            items=len(config),
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
            "execution_mode": app_settings.execution_mode,
        },
        event_id=f"bot_started:{storage.run_id}",
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
        await guarded_call(
            lambda: storage.mark_run_stopped(reason="graceful_shutdown"),
            logger=logger,
            event="shutdown_mark_run_stopped_failed",
            message="Failed to mark run stopped",
        )

        await guarded_call(
            lambda: storage.publish_event(
                level="INFO",
                event="bot_stopped",
                message="Bot process stopped gracefully",
                event_id=f"bot_stopped:{storage.run_id}",
            ),
            logger=logger,
            event="shutdown_publish_stopped_failed",
            message="Failed to publish bot_stopped event",
        )

        await guarded_call(
            watcher.close,
            logger=logger,
            event="shutdown_watcher_close_failed",
            message="Failed to close watcher",
        )
        await guarded_call(
            executor.close,
            logger=logger,
            event="shutdown_executor_close_failed",
            message="Failed to close executor",
        )
        await guarded_call(
            storage.close,
            logger=logger,
            event="shutdown_storage_close_failed",
            message="Failed to close storage",
        )

        log_event(
            logger,
            level="info",
            event="shutdown_completed",
            message="Shutdown completed",
        )


if __name__ == "__main__":
    asyncio.run(main())
