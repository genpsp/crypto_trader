from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from google.cloud import firestore
from redis import asyncio as redis
from redis.asyncio.client import Redis

ConfigUpdateHandler = Callable[[dict[str, Any]], Awaitable[None]]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_for_redis(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


@dataclass(slots=True)
class StorageSettings:
    redis_url: str
    redis_config_key: str
    firestore_project_id: str | None
    firestore_config_doc: str
    events_collection: str
    heartbeat_key: str
    price_prefix: str
    spread_prefix: str
    position_key: str
    order_guard_prefix: str
    order_record_prefix: str

    @classmethod
    def from_env(cls) -> "StorageSettings":
        return cls(
            redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
            redis_config_key=os.getenv("REDIS_CONFIG_KEY", "config"),
            firestore_project_id=os.getenv("FIRESTORE_PROJECT_ID") or None,
            firestore_config_doc=os.getenv("FIRESTORE_CONFIG_DOC", "bots/solana-bot/config"),
            events_collection=os.getenv("BOT_EVENTS_COLLECTION", "bot_events"),
            heartbeat_key=os.getenv("REDIS_HEARTBEAT_KEY", "bot:heartbeat"),
            price_prefix=os.getenv("REDIS_PRICE_PREFIX", "prices"),
            spread_prefix=os.getenv("REDIS_SPREAD_PREFIX", "spreads"),
            position_key=os.getenv("REDIS_POSITION_KEY", "position:current"),
            order_guard_prefix=os.getenv("REDIS_ORDER_GUARD_PREFIX", "orders:guard"),
            order_record_prefix=os.getenv("REDIS_ORDER_RECORD_PREFIX", "orders:record"),
        )


class StorageGateway:
    def __init__(self, settings: StorageSettings, logger: logging.Logger) -> None:
        self.settings = settings
        self._logger = logger
        self._redis: Redis | None = None
        self._firestore: firestore.Client | None = None
        self._config_doc_ref: Any | None = None
        self._watch: Any | None = None

    async def connect(self) -> None:
        firebase_credentials = os.getenv("FIREBASE_CREDENTIALS")
        if firebase_credentials and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = firebase_credentials

        self._redis = redis.from_url(self.settings.redis_url, decode_responses=True)
        await self._redis.ping()
        self._logger.info("Connected to Redis", extra={"event": "redis_connected"})

        self._firestore = firestore.Client(project=self.settings.firestore_project_id)
        self._config_doc_ref = self._firestore.document(self.settings.firestore_config_doc)

        startup_snapshot = await asyncio.to_thread(self._config_doc_ref.get)
        if startup_snapshot.exists:
            await self.sync_config_to_redis(startup_snapshot.to_dict() or {}, source="startup")
        else:
            self._logger.warning(
                "Runtime config document does not exist",
                extra={"event": "config_missing", "doc_path": self.settings.firestore_config_doc},
            )

        self._logger.info(
            "Connected to Firestore",
            extra={"event": "firestore_connected", "doc_path": self.settings.firestore_config_doc},
        )

    async def healthcheck(self) -> None:
        redis_client = self._require_redis()
        await redis_client.ping()

        if self._config_doc_ref is None:
            raise RuntimeError("Firestore config document reference is not initialized.")

        await asyncio.to_thread(self._config_doc_ref.get)

    async def publish_event(
        self,
        *,
        level: str,
        event: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        if self._firestore is None:
            self._logger.warning(
                "Skipping Firestore event because client is not ready",
                extra={"event": "publish_skipped"},
            )
            return

        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc),
            "level": level,
            "event": event,
            "message": message,
        }
        if details:
            payload["details"] = details

        try:
            await asyncio.to_thread(self._firestore.collection(self.settings.events_collection).add, payload)
        except Exception as error:
            self._logger.error(
                "Failed to publish Firestore event",
                extra={"event": "publish_failed", "error": str(error)},
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
                        self._logger.error(
                            "Config sync task failed",
                            extra={"event": "config_sync_failed", "error": str(error)},
                        )

            task.add_done_callback(on_done)

        def on_snapshot(doc_snapshot: list[Any], _changes: list[Any], _read_time: Any) -> None:
            if not doc_snapshot or loop.is_closed():
                return
            data = doc_snapshot[0].to_dict() or {}
            loop.call_soon_threadsafe(schedule, self._handle_config_update(data, on_update))

        self._watch = self._config_doc_ref.on_snapshot(on_snapshot)
        self._logger.info("Config watcher started", extra={"event": "config_watch_started"})

    async def _handle_config_update(
        self,
        config: dict[str, Any],
        on_update: ConfigUpdateHandler | None,
    ) -> None:
        await self.sync_config_to_redis(config, source="snapshot")
        if on_update:
            await on_update(config)

    async def sync_config_to_redis(self, config: dict[str, Any], *, source: str) -> None:
        redis_client = self._require_redis()
        if not config:
            return

        mapping = {str(key): _serialize_for_redis(value) for key, value in config.items()}
        await redis_client.hset(self.settings.redis_config_key, mapping=mapping)
        self._logger.info(
            "Runtime config synced to Redis",
            extra={"event": "config_synced", "items": len(mapping), "source": source},
        )

    async def get_runtime_config(self) -> dict[str, str]:
        redis_client = self._require_redis()
        return await redis_client.hgetall(self.settings.redis_config_key)

    async def acquire_order_guard(
        self,
        *,
        guard_key: str,
        order_id: str,
        ttl_seconds: int,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        redis_client = self._require_redis()
        lock_key = f"{self.settings.order_guard_prefix}:{guard_key}"

        lock_payload: dict[str, Any] = {
            "order_id": order_id,
            "guard_key": guard_key,
            "created_at": _now_iso(),
        }
        if payload:
            lock_payload["payload"] = payload

        acquired = await redis_client.set(
            lock_key,
            json.dumps(lock_payload, ensure_ascii=False, separators=(",", ":"), default=str),
            ex=max(1, ttl_seconds),
            nx=True,
        )
        return bool(acquired)

    async def get_order_guard(self, *, guard_key: str) -> dict[str, Any] | None:
        redis_client = self._require_redis()
        lock_key = f"{self.settings.order_guard_prefix}:{guard_key}"
        raw = await redis_client.get(lock_key)
        if raw is None:
            return None

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"raw": raw}

        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}

    async def release_order_guard(self, *, guard_key: str) -> None:
        redis_client = self._require_redis()
        lock_key = f"{self.settings.order_guard_prefix}:{guard_key}"
        await redis_client.delete(lock_key)

    async def record_order_state(
        self,
        *,
        order_id: str,
        status: str,
        ttl_seconds: int,
        payload: dict[str, Any] | None = None,
        guard_key: str | None = None,
    ) -> None:
        redis_client = self._require_redis()
        record_key = f"{self.settings.order_record_prefix}:{order_id}"

        mapping: dict[str, str] = {
            "order_id": order_id,
            "status": status,
            "updated_at": _now_iso(),
        }
        if guard_key:
            mapping["guard_key"] = guard_key
        if payload is not None:
            mapping["payload"] = json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )

        await redis_client.hset(record_key, mapping=mapping)
        await redis_client.expire(record_key, max(60, ttl_seconds))

    async def record_price(self, *, pair: str, price: float, raw: dict[str, Any]) -> None:
        redis_client = self._require_redis()
        redis_key = f"{self.settings.price_prefix}:{pair}"
        await redis_client.hset(
            redis_key,
            mapping={
                "pair": pair,
                "price": f"{price:.10f}",
                "raw": json.dumps(raw, ensure_ascii=False, separators=(",", ":"), default=str),
                "updated_at": _now_iso(),
            },
        )

    async def record_spread(
        self,
        *,
        pair: str,
        spread_bps: float,
        required_spread_bps: float,
        total_fee_bps: float,
        profitable: bool,
        extra: dict[str, Any] | None = None,
    ) -> None:
        redis_client = self._require_redis()
        redis_key = f"{self.settings.spread_prefix}:{pair}"
        mapping: dict[str, str] = {
            "pair": pair,
            "spread_bps": f"{spread_bps:.6f}",
            "required_spread_bps": f"{required_spread_bps:.6f}",
            "total_fee_bps": f"{total_fee_bps:.6f}",
            "profitable": "1" if profitable else "0",
            "updated_at": _now_iso(),
        }
        if extra:
            mapping.update({str(key): _serialize_for_redis(value) for key, value in extra.items()})

        await redis_client.hset(redis_key, mapping=mapping)

    async def record_position(self, mapping: dict[str, Any]) -> None:
        redis_client = self._require_redis()
        payload = {key: _serialize_for_redis(value) for key, value in mapping.items()}
        payload["updated_at"] = _now_iso()
        await redis_client.hset(self.settings.position_key, mapping=payload)

    async def update_heartbeat(self) -> None:
        redis_client = self._require_redis()
        await redis_client.set(self.settings.heartbeat_key, _now_iso())

    async def close(self) -> None:
        if self._watch is not None:
            with contextlib.suppress(Exception):
                self._watch.unsubscribe()
            self._watch = None

        if self._redis is not None:
            close = getattr(self._redis, "aclose", None)
            if close:
                await close()
            else:
                await self._redis.close()
            self._redis = None

        self._config_doc_ref = None
        self._firestore = None

    def _require_redis(self) -> Redis:
        if self._redis is None:
            raise RuntimeError("Redis client is not initialized.")
        return self._redis
