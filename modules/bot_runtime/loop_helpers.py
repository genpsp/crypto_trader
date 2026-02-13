from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from modules.common import guarded_call, log_event

if TYPE_CHECKING:
    from modules.storage import ConfigUpdateHandler, StorageGateway
    from modules.trading import RuntimeConfig, TraderEngine
    from .settings import AppSettings


def to_float(value: Any, default: float | None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def parse_iso_timestamp(value: Any) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return None


def drawdown_pct_from_peak(*, current_lamports: int, peak_lamports: int) -> float:
    if peak_lamports <= 0:
        return 0.0
    return max(0.0, ((peak_lamports - current_lamports) / peak_lamports) * 100.0)


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
    tip_lamports: int = 0,
) -> int:
    priority_fee_lamports = int((priority_fee_micro_lamports * priority_compute_units) / 1_000_000)
    return max(0, priority_fee_lamports) + max(0, base_fee_lamports) + max(0, tip_lamports)


def compute_rate_limit_backoff_seconds(*, rate_limited_count: int) -> float:
    if rate_limited_count <= 0:
        return 0.0
    base_seconds = min(30.0, float(2 ** max(0, rate_limited_count - 1)))
    jitter_seconds = random.uniform(0.0, max(0.1, base_seconds * 0.25))
    return min(30.0, base_seconds + jitter_seconds)


async def bootstrap_dependencies(
    *,
    logger: logging.Logger,
    stop_event: asyncio.Event,
    app_settings: AppSettings,
    storage: StorageGateway,
    watcher: Any,
    executor: Any,
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


async def check_and_recover_pending_guard(
    *,
    logger: logging.Logger,
    storage: StorageGateway,
    trader_engine: TraderEngine,
    runtime_config: RuntimeConfig,
    app_settings: AppSettings,
) -> bool:
    pending_records_snapshot: list[dict[str, Any]] = []
    if runtime_config.execution_mode == "atomic":
        pending_atomic_records = await storage.list_pending_atomic(
            statuses={"submitted", "pending_confirmation", "confirming"},
            limit=app_settings.live_max_pending_orders + 1,
        )
        pending_count = len(pending_atomic_records)
        pending_records_snapshot = pending_atomic_records
    else:
        pending_records = await storage.list_order_records(
            statuses={"pending", "submitted", "pending_confirmation", "confirming"},
            limit=app_settings.live_max_pending_orders + 1,
        )
        pending_count = len(pending_records)
        pending_records_snapshot = pending_records

    if pending_count < app_settings.live_max_pending_orders:
        return False

    pending_count_before_recovery = pending_count
    recover_pending = getattr(trader_engine.executor, "recover_pending", None)
    if callable(recover_pending):
        await guarded_call(
            recover_pending,
            logger=logger,
            event="pending_recovery_during_guard_failed",
            message="Failed to recover pending executions while pending guard was active",
            level="warning",
        )
        if runtime_config.execution_mode == "atomic":
            pending_atomic_records = await storage.list_pending_atomic(
                statuses={"submitted", "pending_confirmation", "confirming"},
                limit=app_settings.live_max_pending_orders + 1,
            )
            pending_count = len(pending_atomic_records)
            pending_records_snapshot = pending_atomic_records
        else:
            pending_records = await storage.list_order_records(
                statuses={"pending", "submitted", "pending_confirmation", "confirming"},
                limit=app_settings.live_max_pending_orders + 1,
            )
            pending_count = len(pending_records)
            pending_records_snapshot = pending_records

    if pending_count >= app_settings.live_max_pending_orders:
        oldest_pending_age_seconds = None
        if pending_records_snapshot:
            now_epoch = datetime.now(timezone.utc).timestamp()
            known_ages = []
            for record in pending_records_snapshot:
                updated_at_ts = parse_iso_timestamp(record.get("updated_at"))
                created_at_ts = parse_iso_timestamp(record.get("created_at"))
                anchor_ts = created_at_ts if created_at_ts is not None else updated_at_ts
                if anchor_ts is None:
                    continue
                known_ages.append(max(0.0, now_epoch - anchor_ts))
            if known_ages:
                oldest_pending_age_seconds = max(known_ages)

        log_event(
            logger,
            level="warning",
            event="execution_pending_guard",
            message="Execution skipped due to pending order guard",
            pending_count=pending_count,
            pending_count_before_recovery=pending_count_before_recovery,
            max_pending_orders=app_settings.live_max_pending_orders,
            execution_mode=runtime_config.execution_mode,
            oldest_pending_age_seconds=(
                round(oldest_pending_age_seconds, 3)
                if oldest_pending_age_seconds is not None
                else None
            ),
        )
        return True

    log_event(
        logger,
        level="info",
        event="execution_pending_guard_recovered",
        message="Pending guard was cleared after immediate recovery",
        pending_count_before_recovery=pending_count_before_recovery,
        pending_count_after_recovery=pending_count,
        execution_mode=runtime_config.execution_mode,
    )
    return False


def prepare_execution_result_metadata(
    *,
    result_metadata: dict[str, Any] | None,
    initial_expected_net_bps: float,
) -> tuple[dict[str, Any], str, float | None, float | None]:
    normalized = dict(result_metadata or {})
    explicit_metrics_source = str(normalized.get("metrics_source") or "").strip().lower()
    plan_expected_net_raw = normalized.get("expected_net_bps")
    has_plan_expected_net = not (plan_expected_net_raw is None or str(plan_expected_net_raw).strip() == "")

    if explicit_metrics_source in {"initial", "plan"}:
        metrics_source = explicit_metrics_source
    elif has_plan_expected_net or str(normalized.get("plan_id") or "").strip():
        metrics_source = "plan"
    else:
        metrics_source = "initial"

    normalized.setdefault("metrics_source", metrics_source)
    normalized.setdefault("initial_expected_net_bps", initial_expected_net_bps)

    plan_expected_net_bps = to_float(normalized.get("expected_net_bps"), None)
    requote_decay_bps = None
    if plan_expected_net_bps is not None:
        requote_decay_bps = plan_expected_net_bps - initial_expected_net_bps
        normalized.setdefault("requote_decay_bps", requote_decay_bps)

    return normalized, metrics_source, plan_expected_net_bps, requote_decay_bps


__all__ = [
    "bootstrap_dependencies",
    "check_and_recover_pending_guard",
    "compute_rate_limit_backoff_seconds",
    "drawdown_pct_from_peak",
    "estimate_execution_fee_lamports",
    "prepare_execution_result_metadata",
    "to_float",
    "try_resume_order_intake",
    "wait_with_stop",
]
