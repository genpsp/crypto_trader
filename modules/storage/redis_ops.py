from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from redis.asyncio.client import Redis

from modules.common import log_event

from .helpers import now_iso as _now_iso
from .helpers import serialize_for_redis as _serialize_for_redis


class RedisStorageOps:
    @staticmethod
    def _order_lock_key(prefix: str, guard_key: str) -> str:
        return f"{prefix}:{guard_key}"

    @staticmethod
    def _pending_atomic_key(prefix: str, plan_id: str) -> str:
        return f"{prefix}:{plan_id}"

    async def sync_config_to_redis(self, config: dict[str, Any], *, source: str) -> None:
        redis_client = self._require_redis()

        mapping = {str(key): _serialize_for_redis(value) for key, value in config.items()}
        pipeline = redis_client.pipeline(transaction=True)
        pipeline.delete(self.settings.redis_config_key)
        if mapping:
            pipeline.hset(self.settings.redis_config_key, mapping=mapping)
        await pipeline.execute()
        log_event(
            self._logger,
            level="info",
            event="config_synced",
            message="Runtime config synced to Redis",
            items=len(mapping),
            source=source,
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
    ) -> bool:
        redis_client = self._require_redis()
        lock_key = self._order_lock_key(self.settings.order_guard_prefix, guard_key)

        acquired = await redis_client.set(
            lock_key,
            order_id,
            ex=max(1, ttl_seconds),
            nx=True,
        )
        return bool(acquired)

    async def get_order_guard(self, *, guard_key: str) -> dict[str, Any] | None:
        redis_client = self._require_redis()
        lock_key = self._order_lock_key(self.settings.order_guard_prefix, guard_key)
        raw, ttl_seconds = await asyncio.gather(
            redis_client.get(lock_key),
            redis_client.ttl(lock_key),
        )
        if raw is None:
            return None

        return {
            "order_id": raw,
            "ttl_seconds": int(ttl_seconds) if isinstance(ttl_seconds, int) else -2,
        }

    async def refresh_order_guard(
        self,
        *,
        guard_key: str,
        order_id: str,
        ttl_seconds: int,
    ) -> bool:
        redis_client = self._require_redis()
        lock_key = self._order_lock_key(self.settings.order_guard_prefix, guard_key)
        refreshed = await redis_client.eval(
            """
            if redis.call('get', KEYS[1]) == ARGV[1] then
              return redis.call('expire', KEYS[1], ARGV[2])
            end
            return 0
            """,
            1,
            lock_key,
            order_id,
            str(max(1, ttl_seconds)),
        )
        return bool(refreshed)

    async def release_order_guard(
        self,
        *,
        guard_key: str,
        order_id: str,
    ) -> bool:
        redis_client = self._require_redis()
        lock_key = self._order_lock_key(self.settings.order_guard_prefix, guard_key)
        deleted = await redis_client.eval(
            """
            if redis.call('get', KEYS[1]) == ARGV[1] then
              return redis.call('del', KEYS[1])
            end
            return 0
            """,
            1,
            lock_key,
            order_id,
        )
        return bool(deleted)

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

    async def list_order_records(
        self,
        *,
        statuses: set[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        redis_client = self._require_redis()
        normalized_limit = max(1, limit)
        normalized_statuses = {status for status in (statuses or set()) if status}
        pattern = f"{self.settings.order_record_prefix}:*"

        records: list[dict[str, Any]] = []
        async for key in redis_client.scan_iter(match=pattern, count=min(1000, normalized_limit * 4)):
            payload = await redis_client.hgetall(key)
            if not payload:
                continue

            status = str(payload.get("status", ""))
            if normalized_statuses and status not in normalized_statuses:
                continue

            parsed_payload: dict[str, Any] = {}
            raw_payload = payload.get("payload")
            if raw_payload:
                with contextlib.suppress(Exception):
                    candidate = json.loads(raw_payload)
                    if isinstance(candidate, dict):
                        parsed_payload = candidate

            order_id = str(payload.get("order_id", "")) or key.split(":")[-1]
            records.append(
                {
                    "order_id": order_id,
                    "status": status,
                    "guard_key": str(payload.get("guard_key", "")),
                    "updated_at": str(payload.get("updated_at", "")),
                    "payload": parsed_payload,
                }
            )
            if len(records) >= normalized_limit:
                break

        records.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return records

    async def save_pending_atomic(
        self,
        *,
        plan_id: str,
        status: str,
        order_id: str,
        guard_key: str,
        tx_signatures: list[str],
        ttl_seconds: int,
        mode: str,
        bundle_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        redis_client = self._require_redis()
        record_key = self._pending_atomic_key(self.settings.pending_atomic_prefix, plan_id)
        now = _now_iso()

        mapping: dict[str, str] = {
            "plan_id": plan_id,
            "status": status,
            "order_id": order_id,
            "guard_key": guard_key,
            "mode": mode,
            "created_at": now,
            "updated_at": now,
            "tx_signatures": json.dumps(tx_signatures, ensure_ascii=False, separators=(",", ":")),
        }
        if bundle_id:
            mapping["bundle_id"] = bundle_id
        if payload is not None:
            mapping["payload"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)

        await redis_client.hset(record_key, mapping=mapping)
        await redis_client.expire(record_key, max(60, ttl_seconds))

    async def update_pending_atomic(
        self,
        *,
        plan_id: str,
        status: str,
        ttl_seconds: int,
        tx_signatures: list[str] | None = None,
        bundle_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        redis_client = self._require_redis()
        record_key = self._pending_atomic_key(self.settings.pending_atomic_prefix, plan_id)
        existing = await redis_client.hgetall(record_key)

        mapping: dict[str, str] = {
            "plan_id": plan_id,
            "status": status,
            "updated_at": _now_iso(),
        }
        if tx_signatures is not None:
            mapping["tx_signatures"] = json.dumps(
                tx_signatures,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        if bundle_id is not None:
            mapping["bundle_id"] = bundle_id

        if payload is not None:
            merged_payload: dict[str, Any] = {}
            raw_existing_payload = existing.get("payload")
            if raw_existing_payload:
                with contextlib.suppress(Exception):
                    existing_payload = json.loads(raw_existing_payload)
                    if isinstance(existing_payload, dict):
                        merged_payload = dict(existing_payload)
            merged_payload.update(payload)
            mapping["payload"] = json.dumps(
                merged_payload,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )

        await redis_client.hset(record_key, mapping=mapping)
        await redis_client.expire(record_key, max(60, ttl_seconds))

    async def get_pending_atomic(self, *, plan_id: str) -> dict[str, Any] | None:
        redis_client = self._require_redis()
        record_key = self._pending_atomic_key(self.settings.pending_atomic_prefix, plan_id)
        payload = await redis_client.hgetall(record_key)
        if not payload:
            return None

        tx_signatures: list[str] = []
        raw_signatures = payload.get("tx_signatures")
        if raw_signatures:
            with contextlib.suppress(Exception):
                parsed = json.loads(raw_signatures)
                if isinstance(parsed, list):
                    tx_signatures = [str(item) for item in parsed]

        parsed_payload: dict[str, Any] = {}
        raw_payload = payload.get("payload")
        if raw_payload:
            with contextlib.suppress(Exception):
                candidate = json.loads(raw_payload)
                if isinstance(candidate, dict):
                    parsed_payload = candidate

        return {
            "plan_id": str(payload.get("plan_id", "")),
            "status": str(payload.get("status", "")),
            "order_id": str(payload.get("order_id", "")),
            "guard_key": str(payload.get("guard_key", "")),
            "mode": str(payload.get("mode", "")),
            "bundle_id": str(payload.get("bundle_id", "")) or None,
            "created_at": str(payload.get("created_at", "")),
            "updated_at": str(payload.get("updated_at", "")),
            "tx_signatures": tx_signatures,
            "payload": parsed_payload,
        }

    async def list_pending_atomic(
        self,
        *,
        statuses: set[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        redis_client = self._require_redis()
        normalized_limit = max(1, limit)
        normalized_statuses = {status for status in (statuses or set()) if status}
        pattern = f"{self.settings.pending_atomic_prefix}:*"

        records: list[dict[str, Any]] = []
        async for key in redis_client.scan_iter(match=pattern, count=min(1000, normalized_limit * 4)):
            payload = await redis_client.hgetall(key)
            if not payload:
                continue

            status = str(payload.get("status", ""))
            if normalized_statuses and status not in normalized_statuses:
                continue

            tx_signatures: list[str] = []
            raw_signatures = payload.get("tx_signatures")
            if raw_signatures:
                with contextlib.suppress(Exception):
                    parsed_signatures = json.loads(raw_signatures)
                    if isinstance(parsed_signatures, list):
                        tx_signatures = [str(item) for item in parsed_signatures]

            parsed_payload: dict[str, Any] = {}
            raw_payload = payload.get("payload")
            if raw_payload:
                with contextlib.suppress(Exception):
                    candidate = json.loads(raw_payload)
                    if isinstance(candidate, dict):
                        parsed_payload = candidate

            records.append(
                {
                    "plan_id": str(payload.get("plan_id", "")) or key.split(":")[-1],
                    "status": status,
                    "order_id": str(payload.get("order_id", "")),
                    "guard_key": str(payload.get("guard_key", "")),
                    "mode": str(payload.get("mode", "")),
                    "bundle_id": str(payload.get("bundle_id", "")) or None,
                    "created_at": str(payload.get("created_at", "")),
                    "updated_at": str(payload.get("updated_at", "")),
                    "tx_signatures": tx_signatures,
                    "payload": parsed_payload,
                }
            )
            if len(records) >= normalized_limit:
                break

        records.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return records

    async def delete_pending_atomic(self, *, plan_id: str) -> bool:
        redis_client = self._require_redis()
        record_key = self._pending_atomic_key(self.settings.pending_atomic_prefix, plan_id)
        deleted = await redis_client.delete(record_key)
        return bool(deleted)

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

    def _require_redis(self) -> Redis:
        if self._redis is None:
            raise RuntimeError("Redis client is not initialized.")
        return self._redis
