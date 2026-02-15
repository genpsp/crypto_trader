from __future__ import annotations

import asyncio
import contextlib
import json
import signal
from typing import Any
from urllib.parse import urlsplit

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

    log_event(
        logger,
        level="info",
        event="effective_config_dump_startup",
        message="Startup effective config from environment defaults",
        config_priority=app_settings.env_config_priority,
        runtime_transport="startup_env",
        TRADE_ENABLED=runtime_defaults.trade_enabled,
        TRADE_ENABLED_source="env",
        trade_enabled=runtime_defaults.trade_enabled,
        trade_enabled_source="env",
        ATOMIC_SEND_MODE=runtime_defaults.atomic_send_mode,
        ATOMIC_SEND_MODE_source="env",
        ATOMIC_EXPIRY_MS=runtime_defaults.atomic_expiry_ms,
        ATOMIC_EXPIRY_MS_source="env",
        MIN_SPREAD_BPS=runtime_defaults.min_spread_bps,
        MIN_SPREAD_BPS_source="env",
        INITIAL_MIN_SPREAD_BPS=runtime_defaults.initial_min_spread_bps,
        INITIAL_MIN_SPREAD_BPS_source="env",
        ATOMIC_MARGIN_BPS=runtime_defaults.atomic_margin_bps,
        ATOMIC_MARGIN_BPS_source="env",
        INITIAL_ATOMIC_MARGIN_BPS=runtime_defaults.initial_atomic_margin_bps,
        INITIAL_ATOMIC_MARGIN_BPS_source="env",
        MIN_STAGEA_MARGIN_BPS=runtime_defaults.min_stagea_margin_bps,
        MIN_STAGEA_MARGIN_BPS_source="env",
        PRIORITY_FEE_MICRO_LAMPORTS=runtime_defaults.priority_fee_micro_lamports,
        PRIORITY_FEE_MICRO_LAMPORTS_source="env",
        PRIORITY_COMPUTE_UNITS=runtime_defaults.priority_compute_units,
        PRIORITY_COMPUTE_UNITS_source="env",
        JITO_TIP_LAMPORTS_MAX=runtime_defaults.jito_tip_lamports_max,
        JITO_TIP_LAMPORTS_MAX_source="env",
        JITO_TIP_LAMPORTS_RECOMMENDED=runtime_defaults.jito_tip_lamports_recommended,
        JITO_TIP_LAMPORTS_RECOMMENDED_source="env",
        JITO_TIP_SHARE=runtime_defaults.jito_tip_share,
        JITO_TIP_SHARE_source="env",
        BASE_AMOUNT_SWEEP_CANDIDATES_RAW=list(runtime_defaults.base_amount_sweep_candidates_raw),
        BASE_AMOUNT_SWEEP_CANDIDATES_RAW_source="env",
        BASE_AMOUNT_MAX_RAW=runtime_defaults.base_amount_max_raw,
        BASE_AMOUNT_MAX_RAW_source="env",
        PAIR_BASE_AMOUNT=pair.base_amount,
        PAIR_BASE_AMOUNT_source="env",
        QUOTE_MAX_RPS=runtime_defaults.quote_max_rps,
        QUOTE_MAX_RPS_source="env",
        QUOTE_EXPLORATION_MAX_RPS=runtime_defaults.quote_exploration_max_rps,
        QUOTE_EXPLORATION_MAX_RPS_source="env",
        QUOTE_EXECUTION_MAX_RPS=runtime_defaults.quote_execution_max_rps,
        QUOTE_EXECUTION_MAX_RPS_source="env",
        QUOTE_EXPLORATION_MODE=runtime_defaults.quote_exploration_mode,
        QUOTE_EXPLORATION_MODE_source="env",
        ENABLE_PROBE_MULTI_AMOUNT=runtime_defaults.enable_probe_multi_amount,
        ENABLE_PROBE_MULTI_AMOUNT_source="env",
        ENABLE_PROBE_UNCONSTRAINED=runtime_defaults.enable_probe_unconstrained,
        ENABLE_PROBE_UNCONSTRAINED_source="env",
        QUOTE_DEX_SWEEP_TOPK=runtime_defaults.quote_dex_sweep_topk,
        QUOTE_DEX_SWEEP_TOPK_source="env",
        QUOTE_DEX_SWEEP_COMBO_LIMIT=runtime_defaults.quote_dex_sweep_combo_limit,
        QUOTE_DEX_SWEEP_COMBO_LIMIT_source="env",
        MAX_REQUOTE_RANGE_BPS=runtime_defaults.max_requote_range_bps,
        MAX_REQUOTE_RANGE_BPS_source="env",
    )

    def normalize_quote_params(payload: Any) -> dict[str, str | int | float | bool]:
        if not isinstance(payload, dict):
            return {}
        normalized: dict[str, str | int | float | bool] = {}
        for raw_key, raw_value in payload.items():
            key = str(raw_key).strip()
            if not key:
                continue
            if isinstance(raw_value, bool):
                normalized[key] = raw_value
                continue
            if isinstance(raw_value, int):
                normalized[key] = raw_value
                continue
            if isinstance(raw_value, float):
                normalized[key] = int(raw_value) if raw_value.is_integer() else raw_value
                continue
            value = str(raw_value).strip()
            if not value:
                continue
            lowered = value.lower()
            if lowered in {"true", "false"}:
                normalized[key] = lowered == "true"
                continue
            with contextlib.suppress(ValueError):
                normalized[key] = int(value)
                continue
            with contextlib.suppress(ValueError):
                parsed_float = float(value)
                normalized[key] = int(parsed_float) if parsed_float.is_integer() else parsed_float
                continue
            normalized[key] = value
        return normalized

    def parse_quote_params(raw: str, *, event: str, label: str) -> dict[str, str | int | float | bool]:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            log_event(
                logger,
                level="warning",
                event=event,
                message=f"{label} is invalid JSON; ignoring params",
            )
            return {}
        normalized = normalize_quote_params(parsed)
        if not normalized:
            return {}
        return normalized

    quote_default_params = parse_quote_params(
        app_settings.quote_default_params_json,
        event="quote_default_params_invalid_json",
        label="QUOTE_DEFAULT_PARAMS_JSON",
    )
    quote_initial_params = parse_quote_params(
        app_settings.quote_initial_params_json,
        event="quote_initial_params_invalid_json",
        label="QUOTE_INITIAL_PARAMS_JSON",
    )
    quote_plan_params = parse_quote_params(
        app_settings.quote_plan_params_json,
        event="quote_plan_params_invalid_json",
        label="QUOTE_PLAN_PARAMS_JSON",
    )

    if not quote_initial_params:
        quote_initial_params = dict(quote_default_params)
    if not quote_plan_params:
        quote_plan_params = dict(quote_default_params)

    quote_api_base = app_settings.helius_quote_api
    parsed_quote_api = urlsplit(quote_api_base)
    if (
        not app_settings.helius_jup_proxy_enabled
        and "helius-rpc.com" in parsed_quote_api.netloc.lower()
        and "/jup-proxy/" in parsed_quote_api.path
    ):
        quote_api_base = "https://api.jup.ag/swap/v1/quote"
        log_event(
            logger,
            level="info",
            event="quote_provider_aligned",
            message="Helius jup-proxy is disabled; forcing Jupiter direct quote endpoint",
            configured_quote_api=app_settings.helius_quote_api,
            active_quote_api=quote_api_base,
        )

    storage = StorageGateway(storage_settings, logger)
    watcher = HeliusQuoteWatcher(
        logger=logger,
        api_base_url=quote_api_base,
        api_key=app_settings.helius_api_key or None,
        jupiter_api_key=app_settings.jupiter_api_key or None,
        enable_helius_jup_proxy=app_settings.helius_jup_proxy_enabled,
        default_quote_params=quote_initial_params,
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
                single_tx_compact_requote_max_strategies=(
                    app_settings.live_single_tx_compact_requote_max_strategies
                ),
                plan_quote_params=quote_plan_params,
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
