from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Awaitable

from google.cloud import firestore

from modules.common import guarded_call, log_event

from .helpers import to_float as _to_float
from .helpers import utc_day_id as _utc_day_id
from .settings import ConfigUpdateHandler


class FirestoreStorageOps:
    @staticmethod
    def _normalize_doc_path(doc_path: str, leaf_doc_id: str) -> tuple[str, bool]:
        normalized = doc_path.strip("/")
        if not normalized:
            raise ValueError("FIRESTORE_CONFIG_DOC must not be empty.")

        segments = [part for part in normalized.split("/") if part]
        if len(segments) % 2 == 0:
            return normalized, False

        return f"{normalized}/{leaf_doc_id}", True

    @staticmethod
    def _doc_id_from_text(value: str) -> str:
        normalized = value.strip().replace("/", "_")
        if not normalized:
            raise ValueError("Document id source must not be empty.")

        if len(normalized) <= 128:
            return normalized

        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        return f"{normalized[:96]}-{digest}"

    @staticmethod
    def _hash_payload(payload: dict[str, Any]) -> str:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]

    async def mark_run_stopped(self, *, reason: str) -> None:
        if self._run_doc_ref is None:
            return

        payload = {
            "status": "stopped",
            "stop_reason": reason,
            "stopped_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }
        await guarded_call(
            lambda: asyncio.to_thread(self._run_doc_ref.set, payload, merge=True),
            logger=self._logger,
            event="run_status_update_failed",
            message="Failed to update run status",
        )

    async def publish_event(
        self,
        *,
        level: str,
        event: str,
        message: str,
        details: dict[str, Any] | None = None,
        event_id: str | None = None,
    ) -> None:
        if self._firestore is None or self._events_collection_ref is None:
            log_event(
                self._logger,
                level="warning",
                event="publish_skipped",
                message="Skipping Firestore event because client is not ready",
            )
            return

        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc),
            "server_timestamp": firestore.SERVER_TIMESTAMP,
            "level": level,
            "event": event,
            "message": message,
            "bot_id": self.settings.bot_id,
            "run_id": self.settings.bot_run_id,
            "env": self.settings.bot_env,
            "schema_version": self.settings.config_schema_version,
        }
        if details:
            payload["details"] = details

        async def write_event() -> None:
            if event_id:
                document_id = self._doc_id_from_text(event_id)
                event_ref = self._events_collection_ref.document(document_id)
                await asyncio.to_thread(event_ref.set, payload, merge=True)
                return

            await asyncio.to_thread(self._events_collection_ref.add, payload)

        await guarded_call(
            write_event,
            logger=self._logger,
            event="publish_failed",
            message="Failed to publish Firestore event",
            level="error",
        )

    def start_config_listener(
        self,
        loop: asyncio.AbstractEventLoop,
        on_update: ConfigUpdateHandler | None = None,
    ) -> None:
        if self._config_doc_ref is None:
            raise RuntimeError("StorageGateway is not connected.")
        if self._watch is not None:
            return

        def schedule(coro: Awaitable[None]) -> None:
            task = asyncio.create_task(coro)

            def on_done(done_task: asyncio.Task[None]) -> None:
                with contextlib.suppress(asyncio.CancelledError):
                    error = done_task.exception()
                    if error:
                        log_event(
                            self._logger,
                            level="error",
                            event="config_sync_failed",
                            message="Config sync task failed",
                            error=str(error),
                        )

            task.add_done_callback(on_done)

        def on_snapshot(doc_snapshot: list[Any], _changes: list[Any], _read_time: Any) -> None:
            if not doc_snapshot or loop.is_closed():
                return
            snapshot = doc_snapshot[0]
            data = snapshot.to_dict() if snapshot.exists else {}
            if data is None:
                data = {}
            loop.call_soon_threadsafe(schedule, self._handle_config_update(data, on_update))

        self._watch = self._config_doc_ref.on_snapshot(on_snapshot)
        log_event(
            self._logger,
            level="info",
            event="config_watch_started",
            message="Config watcher started",
        )

    async def _handle_config_update(
        self,
        config: dict[str, Any],
        on_update: ConfigUpdateHandler | None,
    ) -> None:
        await self.sync_config_to_redis(config, source="snapshot")
        if on_update:
            await on_update(config)

    async def record_trade(self, *, trade: dict[str, Any], trade_id: str | None = None) -> None:
        if self._firestore is None or self._trades_collection_ref is None:
            log_event(
                self._logger,
                level="warning",
                event="trade_persist_skipped",
                message="Skipping trade persistence because Firestore client is not ready",
            )
            return

        payload = dict(trade)
        order_id_raw = str(payload.get("order_id") or "").strip()
        if not order_id_raw and trade_id:
            order_id_raw = trade_id

        resolved_trade_id = trade_id or order_id_raw
        if not resolved_trade_id:
            resolved_trade_id = f"trade-{self._hash_payload(payload)}"
        resolved_trade_id = self._doc_id_from_text(resolved_trade_id)

        payload.setdefault("order_id", order_id_raw or resolved_trade_id)
        payload["trade_id"] = resolved_trade_id
        payload["bot_id"] = self.settings.bot_id
        payload["run_id"] = self.settings.bot_run_id
        payload["env"] = self.settings.bot_env
        payload["schema_version"] = self.settings.config_schema_version
        payload["updated_at"] = firestore.SERVER_TIMESTAMP
        payload["created_at"] = firestore.SERVER_TIMESTAMP

        status = str(payload.get("status") or "")
        pnl_delta = _to_float(payload.get("pnl_delta"), 0.0)

        trade_ref = self._trades_collection_ref.document(resolved_trade_id)

        async def write_trade() -> bool:
            await asyncio.to_thread(trade_ref.set, payload, merge=True)
            return True

        written = await guarded_call(
            write_trade,
            logger=self._logger,
            event="trade_persist_failed",
            message="Failed to persist trade record",
            level="error",
            default=False,
            trade_id=resolved_trade_id,
        )
        if not written:
            return

        await guarded_call(
            lambda: self._queue_trade_aggregate(
                trade_id=resolved_trade_id,
                status=status,
                pnl_delta=pnl_delta,
            ),
            logger=self._logger,
            event="trade_aggregate_update_failed",
            message="Failed to queue trade aggregates",
            level="error",
            trade_id=resolved_trade_id,
        )

    async def _queue_trade_aggregate(self, *, trade_id: str, status: str, pnl_delta: float) -> None:
        if self._metrics_doc_ref is None or self._pnl_daily_collection_ref is None:
            return

        success_increment = 1 if status in {"filled", "dry_run"} else 0
        day_id = _utc_day_id()
        should_flush = False

        async with self._aggregate_lock:
            self._aggregate_pending_count += 1
            self._aggregate_pending_success_count += success_increment
            self._aggregate_pending_pnl_total += float(pnl_delta)
            self._aggregate_pending_last_trade_id = trade_id
            self._aggregate_pending_last_status = status

            day_bucket = self._aggregate_pending_by_day.setdefault(
                day_id,
                {"trade_count": 0.0, "successful_trade_count": 0.0, "pnl_total": 0.0},
            )
            day_bucket["trade_count"] += 1.0
            day_bucket["successful_trade_count"] += float(success_increment)
            day_bucket["pnl_total"] += float(pnl_delta)

            elapsed_seconds = (
                datetime.now(timezone.utc) - self._aggregate_last_flush_at
            ).total_seconds()
            if (
                self._aggregate_pending_count >= self.settings.firestore_aggregate_batch_size
                or elapsed_seconds >= self.settings.firestore_aggregate_flush_interval_seconds
            ):
                should_flush = True

        if should_flush:
            await self.flush_trade_aggregates(force=False)

    async def flush_trade_aggregates(self, *, force: bool) -> None:
        if self._metrics_doc_ref is None or self._pnl_daily_collection_ref is None:
            return

        batch: dict[str, Any] | None = None
        now_utc = datetime.now(timezone.utc)
        async with self._aggregate_lock:
            if self._aggregate_pending_count <= 0:
                return

            elapsed_seconds = (now_utc - self._aggregate_last_flush_at).total_seconds()
            if not force:
                if self._aggregate_pending_count < self.settings.firestore_aggregate_batch_size:
                    if elapsed_seconds < self.settings.firestore_aggregate_flush_interval_seconds:
                        return

            batch = {
                "trade_count": int(self._aggregate_pending_count),
                "successful_trade_count": int(self._aggregate_pending_success_count),
                "pnl_total": float(self._aggregate_pending_pnl_total),
                "last_trade_id": self._aggregate_pending_last_trade_id,
                "last_status": self._aggregate_pending_last_status,
                "by_day": {
                    key: {
                        "trade_count": int(value.get("trade_count", 0)),
                        "successful_trade_count": int(value.get("successful_trade_count", 0)),
                        "pnl_total": float(value.get("pnl_total", 0.0)),
                    }
                    for key, value in self._aggregate_pending_by_day.items()
                },
            }

            self._aggregate_pending_count = 0
            self._aggregate_pending_success_count = 0
            self._aggregate_pending_pnl_total = 0.0
            self._aggregate_pending_last_trade_id = ""
            self._aggregate_pending_last_status = ""
            self._aggregate_pending_by_day = {}
            self._aggregate_last_flush_at = now_utc

        if batch is None:
            return

        try:
            await self._apply_trade_aggregate_batch(batch)
        except Exception as error:
            await self._merge_trade_aggregate_batch(batch)
            log_event(
                self._logger,
                level="error",
                event="trade_aggregate_flush_failed",
                message="Failed to flush batched trade aggregates; re-queued pending aggregates",
                error=str(error),
                trade_count=batch["trade_count"],
            )

    async def _merge_trade_aggregate_batch(self, batch: dict[str, Any]) -> None:
        async with self._aggregate_lock:
            self._aggregate_pending_count += int(batch.get("trade_count", 0))
            self._aggregate_pending_success_count += int(batch.get("successful_trade_count", 0))
            self._aggregate_pending_pnl_total += float(batch.get("pnl_total", 0.0))
            last_trade_id = str(batch.get("last_trade_id", ""))
            last_status = str(batch.get("last_status", ""))
            if last_trade_id:
                self._aggregate_pending_last_trade_id = last_trade_id
            if last_status:
                self._aggregate_pending_last_status = last_status

            for day_id, day_payload in batch.get("by_day", {}).items():
                pending_day = self._aggregate_pending_by_day.setdefault(
                    day_id,
                    {"trade_count": 0.0, "successful_trade_count": 0.0, "pnl_total": 0.0},
                )
                pending_day["trade_count"] += float(day_payload.get("trade_count", 0))
                pending_day["successful_trade_count"] += float(day_payload.get("successful_trade_count", 0))
                pending_day["pnl_total"] += float(day_payload.get("pnl_total", 0.0))

    async def _apply_trade_aggregate_batch(self, batch: dict[str, Any]) -> None:
        if self._metrics_doc_ref is None or self._pnl_daily_collection_ref is None:
            return

        runtime_payload: dict[str, Any] = {
            "updated_at": firestore.SERVER_TIMESTAMP,
            "trade_count": firestore.Increment(int(batch["trade_count"])),
            "successful_trade_count": firestore.Increment(int(batch["successful_trade_count"])),
            "pnl_total": firestore.Increment(float(batch["pnl_total"])),
            "bot_id": self.settings.bot_id,
            "env": self.settings.bot_env,
            "schema_version": self.settings.config_schema_version,
            "run_id": self.settings.bot_run_id,
            "last_trade_id": batch.get("last_trade_id", ""),
            "last_status": batch.get("last_status", ""),
        }

        writes: list[Awaitable[Any]] = [
            asyncio.to_thread(self._metrics_doc_ref.set, runtime_payload, merge=True),
        ]

        for day_id, day_payload in batch.get("by_day", {}).items():
            payload = {
                "updated_at": firestore.SERVER_TIMESTAMP,
                "trade_count": firestore.Increment(int(day_payload.get("trade_count", 0))),
                "successful_trade_count": firestore.Increment(
                    int(day_payload.get("successful_trade_count", 0))
                ),
                "pnl_total": firestore.Increment(float(day_payload.get("pnl_total", 0.0))),
                "bot_id": self.settings.bot_id,
                "env": self.settings.bot_env,
                "schema_version": self.settings.config_schema_version,
                "day_id": day_id,
            }
            daily_doc_ref = self._pnl_daily_collection_ref.document(day_id)
            writes.append(asyncio.to_thread(daily_doc_ref.set, payload, merge=True))

        await asyncio.gather(*writes)

    async def _ensure_bot_namespace(self) -> None:
        if self._bot_doc_ref is None or self._run_doc_ref is None:
            raise RuntimeError("Firestore namespace references are not initialized.")

        bot_payload: dict[str, Any] = {
            "bot_id": self.settings.bot_id,
            "env": self.settings.bot_env,
            "schema_version": self.settings.config_schema_version,
            "config_doc": self._resolved_firestore_config_doc,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }

        run_payload: dict[str, Any] = {
            "run_id": self.settings.bot_run_id,
            "bot_id": self.settings.bot_id,
            "env": self.settings.bot_env,
            "status": "running",
            "pid": os.getpid(),
            "started_at": firestore.SERVER_TIMESTAMP,
            "updated_at": firestore.SERVER_TIMESTAMP,
        }

        await asyncio.gather(
            asyncio.to_thread(self._bot_doc_ref.set, bot_payload, merge=True),
            asyncio.to_thread(self._run_doc_ref.set, run_payload, merge=True),
        )

    def _initialize_namespace_refs(self) -> None:
        firestore_client = self._require_firestore()

        bot_doc_path = f"{self.settings.bot_collection}/{self.settings.bot_id}"
        self._bot_doc_ref = firestore_client.document(bot_doc_path)
        self._run_doc_ref = self._bot_doc_ref.collection(self.settings.bot_runs_collection).document(
            self.settings.bot_run_id
        )
        self._events_collection_ref = self._run_doc_ref.collection(self.settings.bot_events_collection)
        self._trades_collection_ref = self._bot_doc_ref.collection(self.settings.bot_trades_collection)
        self._pnl_daily_collection_ref = self._bot_doc_ref.collection(self.settings.bot_pnl_daily_collection)
        self._metrics_doc_ref = self._bot_doc_ref.collection(self.settings.bot_metrics_collection).document(
            self.settings.bot_metrics_doc_id
        )

    def _require_firestore(self) -> firestore.Client:
        if self._firestore is None:
            raise RuntimeError("Firestore client is not initialized.")
        return self._firestore
