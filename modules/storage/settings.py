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


@dataclass(slots=True)
class StorageSettings:
    redis_url: str
    redis_config_key: str
    firestore_project_id: str | None
    firestore_config_doc: str
    firestore_config_leaf_doc_id: str
    bot_collection: str
    bot_id: str
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

    @classmethod
    def from_env(cls) -> "StorageSettings":
        bot_collection = (os.getenv("BOT_COLLECTION", "bots").strip("/") or "bots")
        bot_id = (os.getenv("BOT_ID", "solana-bot").strip() or "solana-bot").replace("/", "-")
        default_config_doc = f"{bot_collection}/{bot_id}/config/runtime"

        return cls(
            redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
            redis_config_key=os.getenv("REDIS_CONFIG_KEY", "config"),
            firestore_project_id=os.getenv("FIRESTORE_PROJECT_ID") or None,
            firestore_config_doc=os.getenv("FIRESTORE_CONFIG_DOC") or default_config_doc,
            firestore_config_leaf_doc_id=os.getenv("FIRESTORE_CONFIG_LEAF_DOC_ID", "runtime"),
            bot_collection=bot_collection,
            bot_id=bot_id,
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
        )
