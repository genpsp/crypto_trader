from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

ConfigUpdateHandler = Callable[[dict[str, Any]], Awaitable[None]]


def to_int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_bot_id(value: str, default: str) -> str:
    normalized = (value.strip() or default).replace("/", "-")
    return normalized or default


@dataclass(slots=True)
class StorageSettings:
    redis_url: str
    redis_config_key: str
    firestore_project_id: str | None
    firestore_config_doc: str
    firestore_config_leaf_doc_id: str
    bot_collection: str
    bot_id: str
    firestore_results_bot_id: str
    dry_run: bool
    firestore_split_dry_run_results: bool
    firestore_dry_run_results_suffix: str
    bot_env: str
    bot_run_id: str
    bot_runs_collection: str
    bot_events_collection: str
    bot_trades_collection: str
    bot_pnl_daily_collection: str
    bot_metrics_collection: str
    bot_metrics_doc_id: str
    firestore_publish_order_execution_events: bool
    firestore_aggregate_batch_size: int
    firestore_aggregate_flush_interval_seconds: float
    config_schema_version: int
    heartbeat_key: str
    price_prefix: str
    spread_prefix: str
    position_key: str
    order_guard_prefix: str
    order_record_prefix: str
    pending_atomic_prefix: str
    runtime_counter_key: str
    runtime_summary_key: str
    rate_limit_pause_key: str
    runtime_metrics_ttl_seconds: int

    @classmethod
    def from_env(cls) -> "StorageSettings":
        bot_collection = (os.getenv("BOT_COLLECTION", "bots").strip("/") or "bots")
        bot_id = _sanitize_bot_id(os.getenv("BOT_ID", "solana-bot"), "solana-bot")
        default_config_doc = f"{bot_collection}/{bot_id}/config/runtime"

        dry_run = to_bool(os.getenv("DRY_RUN"), True)
        split_dry_run_results = to_bool(os.getenv("FIRESTORE_SPLIT_DRY_RUN_RESULTS"), True)
        dry_run_suffix = (os.getenv("FIRESTORE_DRY_RUN_RESULTS_SUFFIX", "-dryrun") or "-dryrun").strip()
        if not dry_run_suffix:
            dry_run_suffix = "-dryrun"

        explicit_results_bot_id_raw = os.getenv("FIRESTORE_RESULTS_BOT_ID", "")
        explicit_results_bot_id = _sanitize_bot_id(explicit_results_bot_id_raw, "")

        if explicit_results_bot_id:
            firestore_results_bot_id = explicit_results_bot_id
        elif dry_run and split_dry_run_results:
            firestore_results_bot_id = _sanitize_bot_id(f"{bot_id}{dry_run_suffix}", bot_id)
        else:
            firestore_results_bot_id = bot_id

        return cls(
            redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
            redis_config_key=os.getenv("REDIS_CONFIG_KEY", "config"),
            firestore_project_id=os.getenv("FIRESTORE_PROJECT_ID") or None,
            firestore_config_doc=os.getenv("FIRESTORE_CONFIG_DOC") or default_config_doc,
            firestore_config_leaf_doc_id=os.getenv("FIRESTORE_CONFIG_LEAF_DOC_ID", "runtime"),
            bot_collection=bot_collection,
            bot_id=bot_id,
            firestore_results_bot_id=firestore_results_bot_id,
            dry_run=dry_run,
            firestore_split_dry_run_results=split_dry_run_results,
            firestore_dry_run_results_suffix=dry_run_suffix,
            bot_env=os.getenv("BOT_ENV", "dev"),
            bot_run_id=os.getenv("BOT_RUN_ID")
            or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ"),
            bot_runs_collection=os.getenv("BOT_RUNS_COLLECTION", "runs"),
            bot_events_collection=os.getenv("BOT_EVENTS_COLLECTION", "events"),
            bot_trades_collection=os.getenv("BOT_TRADES_COLLECTION", "trades"),
            bot_pnl_daily_collection=os.getenv("BOT_PNL_DAILY_COLLECTION", "pnl_daily"),
            bot_metrics_collection=os.getenv("BOT_METRICS_COLLECTION", "metrics"),
            bot_metrics_doc_id=os.getenv("BOT_METRICS_DOC_ID", "runtime"),
            firestore_publish_order_execution_events=to_bool(
                os.getenv("FIRESTORE_PUBLISH_ORDER_EXECUTION_EVENTS"),
                False,
            ),
            firestore_aggregate_batch_size=max(
                1,
                to_int(os.getenv("FIRESTORE_AGGREGATE_BATCH_SIZE"), 20),
            ),
            firestore_aggregate_flush_interval_seconds=max(
                1.0,
                to_float(os.getenv("FIRESTORE_AGGREGATE_FLUSH_INTERVAL_SECONDS"), 30.0),
            ),
            config_schema_version=max(1, to_int(os.getenv("CONFIG_SCHEMA_VERSION"), 1)),
            heartbeat_key=os.getenv("REDIS_HEARTBEAT_KEY", "bot:heartbeat"),
            price_prefix=os.getenv("REDIS_PRICE_PREFIX", "prices"),
            spread_prefix=os.getenv("REDIS_SPREAD_PREFIX", "spreads"),
            position_key=os.getenv("REDIS_POSITION_KEY", "position:current"),
            order_guard_prefix=os.getenv("REDIS_ORDER_GUARD_PREFIX", "orders:guard"),
            order_record_prefix=os.getenv("REDIS_ORDER_RECORD_PREFIX", "orders:record"),
            pending_atomic_prefix=os.getenv("REDIS_PENDING_ATOMIC_PREFIX", "pending_atomic"),
            runtime_counter_key=os.getenv("REDIS_RUNTIME_COUNTER_KEY", "metrics:runtime:counters"),
            runtime_summary_key=os.getenv("REDIS_RUNTIME_SUMMARY_KEY", "metrics:runtime:summary"),
            rate_limit_pause_key=os.getenv("REDIS_RATE_LIMIT_PAUSE_KEY", "bot:rate_limit_pause_until"),
            runtime_metrics_ttl_seconds=max(
                0,
                to_int(os.getenv("REDIS_RUNTIME_METRICS_TTL_SECONDS"), 172800),
            ),
        )
