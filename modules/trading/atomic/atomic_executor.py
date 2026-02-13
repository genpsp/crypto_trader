from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Protocol
from urllib.parse import urlsplit

import aiohttp

from modules.common import guarded_call, log_event

from .atomic_types import AtomicExecutionPlan


class JitoBundleRateLimitError(RuntimeError):
    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


def _parse_retry_after_seconds(raw: str | None) -> float | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def _error_message_from_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        message = payload.get("message")
        if message:
            return str(message)
        details = payload.get("details")
        if details:
            return str(details)
    return str(payload)


def _is_jito_rate_limit_message(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in (
            "rate limit",
            "too many requests",
            "network congested",
            "congested",
            "try again later",
        )
    )


def _normalize_jito_endpoints(raw: str) -> list[str]:
    values: list[str] = []
    for token in (raw or "").replace("\n", ",").split(","):
        endpoint = token.strip()
        if endpoint and endpoint not in values:
            values.append(endpoint)
    return values


def _endpoint_for_log(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if not parsed.scheme or not parsed.netloc:
        return endpoint
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


class AtomicPendingStore(Protocol):
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
        ...

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
        ...

    async def list_pending_atomic(
        self,
        *,
        statuses: set[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        ...

    async def delete_pending_atomic(self, *, plan_id: str) -> bool:
        ...

    async def record_order_state(
        self,
        *,
        order_id: str,
        status: str,
        ttl_seconds: int,
        payload: dict[str, Any] | None = None,
        guard_key: str | None = None,
    ) -> None:
        ...

    async def release_order_guard(
        self,
        *,
        guard_key: str,
        order_id: str,
    ) -> bool:
        ...

    async def refresh_order_guard(
        self,
        *,
        guard_key: str,
        order_id: str,
        ttl_seconds: int,
    ) -> bool:
        ...


@dataclass(slots=True, frozen=True)
class AtomicRecoverySummary:
    scanned: int
    resolved: int
    unresolved: int


class AtomicPendingManager:
    def __init__(self, *, logger: logging.Logger, store: AtomicPendingStore | None) -> None:
        self._logger = logger
        self._store = store

    async def mark_submitted(
        self,
        *,
        plan: AtomicExecutionPlan,
        order_id: str,
        guard_key: str,
        tx_signatures: list[str],
        ttl_seconds: int,
        bundle_id: str | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        if self._store is None:
            return

        payload = {
            "plan": plan.to_dict(),
            "expected_net_bps": plan.expected_net_bps,
            "expected_fee_bps": plan.expected_fee_bps,
            "tip_lamports": plan.tip_lamports,
        }
        if extra_payload:
            payload.update(extra_payload)

        await self._store.save_pending_atomic(
            plan_id=plan.plan_id,
            status="submitted",
            order_id=order_id,
            guard_key=guard_key,
            tx_signatures=tx_signatures,
            ttl_seconds=ttl_seconds,
            mode=plan.resolved_mode,
            bundle_id=bundle_id,
            payload=payload,
        )

    async def mark_confirmed(
        self,
        *,
        plan_id: str,
        order_id: str,
        guard_key: str,
        tx_signatures: list[str],
        ttl_seconds: int,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self._store is None:
            return

        await self._store.update_pending_atomic(
            plan_id=plan_id,
            status="confirmed",
            ttl_seconds=ttl_seconds,
            tx_signatures=tx_signatures,
            payload=payload,
        )
        await self._store.record_order_state(
            order_id=order_id,
            status="confirmed",
            ttl_seconds=ttl_seconds,
            payload=payload,
            guard_key=guard_key,
        )

    async def mark_failed(
        self,
        *,
        plan_id: str,
        order_id: str,
        guard_key: str,
        tx_signatures: list[str],
        ttl_seconds: int,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self._store is None:
            return

        await self._store.update_pending_atomic(
            plan_id=plan_id,
            status="failed",
            ttl_seconds=ttl_seconds,
            tx_signatures=tx_signatures,
            payload=payload,
        )
        await self._store.record_order_state(
            order_id=order_id,
            status="failed",
            ttl_seconds=ttl_seconds,
            payload=payload,
            guard_key=guard_key,
        )

    async def mark_pending_confirmation(
        self,
        *,
        plan_id: str,
        order_id: str,
        guard_key: str,
        tx_signatures: list[str],
        ttl_seconds: int,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self._store is None:
            return

        await self._store.update_pending_atomic(
            plan_id=plan_id,
            status="pending_confirmation",
            ttl_seconds=ttl_seconds,
            tx_signatures=tx_signatures,
            payload=payload,
        )
        await self._store.record_order_state(
            order_id=order_id,
            status="pending_confirmation",
            ttl_seconds=ttl_seconds,
            payload=payload,
            guard_key=guard_key,
        )

    async def recover(
        self,
        *,
        fetch_signature_status: Callable[[str], Awaitable[dict[str, Any] | None]],
        ttl_seconds: int,
        limit: int,
        stale_after_seconds: float | None = None,
    ) -> AtomicRecoverySummary:
        if self._store is None:
            return AtomicRecoverySummary(scanned=0, resolved=0, unresolved=0)

        records = await self._store.list_pending_atomic(
            statuses={"submitted", "confirming", "pending_confirmation"},
            limit=limit,
        )
        if not records:
            return AtomicRecoverySummary(scanned=0, resolved=0, unresolved=0)

        resolved = 0
        unresolved = 0
        now_epoch = datetime.now(timezone.utc).timestamp()

        def parse_iso_timestamp(value: Any) -> float | None:
            raw = str(value or "").strip()
            if not raw:
                return None
            normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
            try:
                return datetime.fromisoformat(normalized).timestamp()
            except ValueError:
                return None

        stale_threshold = None
        if stale_after_seconds is not None and stale_after_seconds > 0:
            stale_threshold = float(stale_after_seconds)

        for record in records:
            plan_id = str(record.get("plan_id") or "").strip()
            order_id = str(record.get("order_id") or "").strip()
            guard_key = str(record.get("guard_key") or "").strip()
            tx_signatures = record.get("tx_signatures")
            signatures = tx_signatures if isinstance(tx_signatures, list) else []
            signatures = [str(sig).strip() for sig in signatures if str(sig).strip()]
            updated_at_ts = parse_iso_timestamp(record.get("updated_at"))
            created_at_ts = parse_iso_timestamp(record.get("created_at"))
            age_seconds = None
            anchor_ts = created_at_ts if created_at_ts is not None else updated_at_ts
            if anchor_ts is not None:
                age_seconds = max(0.0, now_epoch - anchor_ts)

            if not plan_id or not order_id or not guard_key or not signatures:
                unresolved += 1
                continue

            signature_statuses: list[dict[str, Any] | None] = []
            had_error = False
            for tx_signature in signatures:
                try:
                    status = await fetch_signature_status(tx_signature)
                except Exception as error:
                    log_event(
                        self._logger,
                        level="warning",
                        event="atomic_recovery_signature_status_failed",
                        message="Failed to fetch signature status while recovering atomic pending record",
                        plan_id=plan_id,
                        tx_signature=tx_signature,
                        error=str(error),
                    )
                    had_error = True
                    break
                signature_statuses.append(status)

            if had_error:
                unresolved += 1
                continue

            has_failed_leg = any(
                isinstance(status, dict) and status.get("err") is not None
                for status in signature_statuses
            )
            all_confirmed = bool(signature_statuses) and all(
                isinstance(status, dict)
                and str(status.get("confirmationStatus") or "") in {"confirmed", "finalized"}
                and status.get("err") is None
                for status in signature_statuses
            )

            if has_failed_leg:
                resolved += 1
                await self.mark_failed(
                    plan_id=plan_id,
                    order_id=order_id,
                    guard_key=guard_key,
                    tx_signatures=signatures,
                    ttl_seconds=ttl_seconds,
                    payload={"recovered": True, "reason": "at_least_one_leg_failed"},
                )
                await guarded_call(
                    lambda: self._store.release_order_guard(guard_key=guard_key, order_id=order_id),
                    logger=self._logger,
                    event="atomic_recovery_release_guard_failed",
                    message="Failed to release order guard after failed atomic recovery",
                    level="warning",
                    plan_id=plan_id,
                    order_id=order_id,
                )
                continue

            if all_confirmed:
                resolved += 1
                await self.mark_confirmed(
                    plan_id=plan_id,
                    order_id=order_id,
                    guard_key=guard_key,
                    tx_signatures=signatures,
                    ttl_seconds=ttl_seconds,
                    payload={"recovered": True, "reason": "all_legs_confirmed"},
                )
                await guarded_call(
                    lambda: self._store.release_order_guard(guard_key=guard_key, order_id=order_id),
                    logger=self._logger,
                    event="atomic_recovery_release_guard_failed",
                    message="Failed to release order guard after confirmed atomic recovery",
                    level="warning",
                    plan_id=plan_id,
                    order_id=order_id,
                )
                continue

            if (
                stale_threshold is not None
                and age_seconds is not None
                and age_seconds >= stale_threshold
            ):
                resolved += 1
                stale_reason = "pending_recovery_stale_timeout"
                log_event(
                    self._logger,
                    level="warning",
                    event="atomic_recovery_stale_pending_cleared",
                    message="Pending atomic record exceeded stale timeout and was force-cleared",
                    plan_id=plan_id,
                    order_id=order_id,
                    pending_age_seconds=round(age_seconds, 3),
                    stale_after_seconds=round(stale_threshold, 3),
                )
                await self.mark_failed(
                    plan_id=plan_id,
                    order_id=order_id,
                    guard_key=guard_key,
                    tx_signatures=signatures,
                    ttl_seconds=ttl_seconds,
                    payload={
                        "recovered": True,
                        "reason": stale_reason,
                        "pending_age_seconds": age_seconds,
                    },
                )
                await guarded_call(
                    lambda: self._store.release_order_guard(guard_key=guard_key, order_id=order_id),
                    logger=self._logger,
                    event="atomic_recovery_release_guard_failed",
                    message="Failed to release order guard after stale atomic recovery",
                    level="warning",
                    plan_id=plan_id,
                    order_id=order_id,
                )
                continue

            unresolved += 1
            await guarded_call(
                lambda: self._store.refresh_order_guard(
                    guard_key=guard_key,
                    order_id=order_id,
                    ttl_seconds=ttl_seconds,
                ),
                logger=self._logger,
                event="atomic_recovery_guard_refresh_failed",
                message="Failed to refresh order guard for unresolved atomic pending record",
                level="warning",
                plan_id=plan_id,
                order_id=order_id,
            )

        return AtomicRecoverySummary(scanned=len(records), resolved=resolved, unresolved=unresolved)


class JitoBlockEngineClient:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        block_engine_url: str,
    ) -> None:
        self._logger = logger
        self._block_engine_urls = _normalize_jito_endpoints(block_engine_url)
        self._active_endpoint_index = 0
        self._tip_accounts_cache_by_endpoint: dict[str, list[str]] = {}

    @property
    def block_engine_url(self) -> str:
        endpoints = self._block_engine_urls
        if not endpoints:
            return ""
        if self._active_endpoint_index >= len(endpoints):
            self._active_endpoint_index = 0
        return endpoints[self._active_endpoint_index]

    @property
    def block_engine_urls(self) -> list[str]:
        return list(self._block_engine_urls)

    def update_endpoint(self, block_engine_url: str) -> None:
        updated = _normalize_jito_endpoints(block_engine_url)
        if updated != self._block_engine_urls:
            self._tip_accounts_cache_by_endpoint = {}
            self._active_endpoint_index = 0
        self._block_engine_urls = updated

    def _iter_endpoints(self) -> list[str]:
        endpoints = self._block_engine_urls
        if not endpoints:
            return []
        active = self.block_engine_url
        if not active:
            return list(endpoints)
        return [active, *[endpoint for endpoint in endpoints if endpoint != active]]

    def _promote_endpoint(self, endpoint: str) -> None:
        if endpoint not in self._block_engine_urls:
            return
        self._active_endpoint_index = self._block_engine_urls.index(endpoint)

    def _log_failover(
        self,
        *,
        event: str,
        message: str,
        from_endpoint: str,
        to_endpoint: str,
        error: str,
    ) -> None:
        log_event(
            self._logger,
            level="warning",
            event=event,
            message=message,
            from_endpoint=_endpoint_for_log(from_endpoint),
            to_endpoint=_endpoint_for_log(to_endpoint),
            error=error,
        )

    async def fetch_tip_accounts(
        self,
        *,
        session: aiohttp.ClientSession,
        force_refresh: bool = False,
    ) -> list[str]:
        endpoints = self._iter_endpoints()
        if not endpoints:
            raise RuntimeError("JITO_BLOCK_ENGINE_URL is required for bundle mode.")

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTipAccounts",
            "params": [],
        }
        last_error: Exception | None = None

        for idx, endpoint in enumerate(endpoints, start=1):
            cached = self._tip_accounts_cache_by_endpoint.get(endpoint)
            if cached and not force_refresh:
                self._promote_endpoint(endpoint)
                return list(cached)

            try:
                async with session.post(endpoint, json=payload) as response:
                    status = response.status
                    retry_after_seconds = _parse_retry_after_seconds(response.headers.get("Retry-After"))
                    raw_text = await response.text()
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                last_error = RuntimeError(f"Jito getTipAccounts request failed: {error}")
                if idx < len(endpoints):
                    self._log_failover(
                        event="jito_tip_accounts_endpoint_failover",
                        message="Switching Jito endpoint after tip-account fetch transport failure",
                        from_endpoint=endpoint,
                        to_endpoint=endpoints[idx],
                        error=str(error),
                    )
                    continue
                raise last_error from error

            parsed: Any = None
            try:
                parsed = json.loads(raw_text) if raw_text else {}
            except json.JSONDecodeError:
                parsed = {"raw": raw_text}

            if status == 429:
                rate_error = JitoBundleRateLimitError(
                    f"Jito getTipAccounts failed: status={status} body={str(raw_text)[:240]!r}",
                    retry_after_seconds=retry_after_seconds,
                )
                last_error = rate_error
                if idx < len(endpoints):
                    self._log_failover(
                        event="jito_tip_accounts_endpoint_failover",
                        message="Switching Jito endpoint after tip-account rate limit",
                        from_endpoint=endpoint,
                        to_endpoint=endpoints[idx],
                        error=str(rate_error),
                    )
                    continue
                raise rate_error

            if status >= 400:
                error_message = str(raw_text)
                if isinstance(parsed, dict) and parsed.get("error") is not None:
                    error_message = _error_message_from_payload(parsed.get("error"))
                if _is_jito_rate_limit_message(error_message):
                    rate_error = JitoBundleRateLimitError(
                        f"Jito getTipAccounts rate-limited: status={status} error={error_message}",
                        retry_after_seconds=retry_after_seconds,
                    )
                    last_error = rate_error
                    if idx < len(endpoints):
                        self._log_failover(
                            event="jito_tip_accounts_endpoint_failover",
                            message="Switching Jito endpoint after tip-account rate limit",
                            from_endpoint=endpoint,
                            to_endpoint=endpoints[idx],
                            error=str(rate_error),
                        )
                        continue
                    raise rate_error

                last_error = RuntimeError(
                    f"Jito getTipAccounts failed: status={status} body={str(raw_text)[:240]!r}"
                )
                if idx < len(endpoints):
                    self._log_failover(
                        event="jito_tip_accounts_endpoint_failover",
                        message="Switching Jito endpoint after tip-account response error",
                        from_endpoint=endpoint,
                        to_endpoint=endpoints[idx],
                        error=str(last_error),
                    )
                    continue
                raise last_error

            if isinstance(parsed, dict) and parsed.get("error"):
                error_payload = parsed["error"]
                error_message = _error_message_from_payload(error_payload)
                if _is_jito_rate_limit_message(error_message):
                    rate_error = JitoBundleRateLimitError(
                        f"Jito getTipAccounts rate-limited: {error_message}",
                        retry_after_seconds=retry_after_seconds,
                    )
                    last_error = rate_error
                    if idx < len(endpoints):
                        self._log_failover(
                            event="jito_tip_accounts_endpoint_failover",
                            message="Switching Jito endpoint after tip-account RPC rate limit",
                            from_endpoint=endpoint,
                            to_endpoint=endpoints[idx],
                            error=str(rate_error),
                        )
                        continue
                    raise rate_error

                last_error = RuntimeError(f"Jito getTipAccounts failed: {error_payload}")
                if idx < len(endpoints):
                    self._log_failover(
                        event="jito_tip_accounts_endpoint_failover",
                        message="Switching Jito endpoint after tip-account RPC error",
                        from_endpoint=endpoint,
                        to_endpoint=endpoints[idx],
                        error=str(last_error),
                    )
                    continue
                raise last_error

            result = parsed.get("result") if isinstance(parsed, dict) else None
            if not isinstance(result, list):
                last_error = RuntimeError(f"Unexpected getTipAccounts response: {parsed}")
                if idx < len(endpoints):
                    self._log_failover(
                        event="jito_tip_accounts_endpoint_failover",
                        message="Switching Jito endpoint after malformed tip-account response",
                        from_endpoint=endpoint,
                        to_endpoint=endpoints[idx],
                        error=str(last_error),
                    )
                    continue
                raise last_error

            tip_accounts = [str(item).strip() for item in result if str(item).strip()]
            if not tip_accounts:
                last_error = RuntimeError("Jito getTipAccounts returned no tip accounts.")
                if idx < len(endpoints):
                    self._log_failover(
                        event="jito_tip_accounts_endpoint_failover",
                        message="Switching Jito endpoint after empty tip-account response",
                        from_endpoint=endpoint,
                        to_endpoint=endpoints[idx],
                        error=str(last_error),
                    )
                    continue
                raise last_error

            self._tip_accounts_cache_by_endpoint[endpoint] = tip_accounts
            self._promote_endpoint(endpoint)
            log_event(
                self._logger,
                level="info",
                event="jito_tip_accounts_loaded",
                message="Loaded Jito tip accounts",
                tip_account_count=len(tip_accounts),
                endpoint=_endpoint_for_log(endpoint),
            )
            return list(tip_accounts)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Jito getTipAccounts failed: no endpoint available.")

    async def select_tip_account(
        self,
        *,
        session: aiohttp.ClientSession,
        plan_id: str,
    ) -> str:
        tip_accounts = await self.fetch_tip_accounts(session=session)
        if len(tip_accounts) == 1:
            return tip_accounts[0]
        index = int(hashlib.sha256(plan_id.encode("utf-8")).hexdigest(), 16) % len(tip_accounts)
        return tip_accounts[index]

    async def send_bundle(
        self,
        *,
        session: aiohttp.ClientSession,
        signed_transactions: list[str],
        tip_lamports: int,
        plan_id: str,
    ) -> str | None:
        endpoints = self._iter_endpoints()
        if not endpoints:
            raise RuntimeError("JITO_BLOCK_ENGINE_URL is required for bundle mode.")

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendBundle",
            "params": [
                signed_transactions,
                {
                    "encoding": "base64",
                    "tipLamports": max(0, int(tip_lamports)),
                },
            ],
        }

        last_error: Exception | None = None

        for idx, endpoint in enumerate(endpoints, start=1):
            try:
                async with session.post(endpoint, json=payload) as response:
                    status = response.status
                    retry_after_seconds = _parse_retry_after_seconds(response.headers.get("Retry-After"))
                    raw_text = await response.text()
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                last_error = RuntimeError(f"Jito bundle submission request failed: {error}")
                if idx < len(endpoints):
                    self._log_failover(
                        event="jito_bundle_endpoint_failover",
                        message="Switching Jito endpoint after bundle submit transport failure",
                        from_endpoint=endpoint,
                        to_endpoint=endpoints[idx],
                        error=str(error),
                    )
                    continue
                raise last_error from error

            parsed: Any = None
            try:
                parsed = json.loads(raw_text) if raw_text else {}
            except json.JSONDecodeError:
                parsed = {"raw": raw_text}

            if status == 429:
                rate_error = JitoBundleRateLimitError(
                    f"Jito bundle submission failed: status={status} body={str(raw_text)[:240]!r}",
                    retry_after_seconds=retry_after_seconds,
                )
                last_error = rate_error
                if idx < len(endpoints):
                    self._log_failover(
                        event="jito_bundle_endpoint_failover",
                        message="Switching Jito endpoint after bundle submit rate limit",
                        from_endpoint=endpoint,
                        to_endpoint=endpoints[idx],
                        error=str(rate_error),
                    )
                    continue
                raise rate_error

            if status >= 400:
                error_message = str(raw_text)
                if isinstance(parsed, dict) and parsed.get("error") is not None:
                    error_message = _error_message_from_payload(parsed.get("error"))
                if _is_jito_rate_limit_message(error_message):
                    rate_error = JitoBundleRateLimitError(
                        f"Jito bundle submission rate-limited: status={status} error={error_message}",
                        retry_after_seconds=retry_after_seconds,
                    )
                    last_error = rate_error
                    if idx < len(endpoints):
                        self._log_failover(
                            event="jito_bundle_endpoint_failover",
                            message="Switching Jito endpoint after bundle submit rate limit",
                            from_endpoint=endpoint,
                            to_endpoint=endpoints[idx],
                            error=str(rate_error),
                        )
                        continue
                    raise rate_error
                last_error = RuntimeError(
                    f"Jito bundle submission failed: status={status} body={str(raw_text)[:240]!r}"
                )
                if idx < len(endpoints):
                    self._log_failover(
                        event="jito_bundle_endpoint_failover",
                        message="Switching Jito endpoint after bundle submit response error",
                        from_endpoint=endpoint,
                        to_endpoint=endpoints[idx],
                        error=str(last_error),
                    )
                    continue
                raise last_error

            if isinstance(parsed, dict) and parsed.get("error"):
                error_payload = parsed["error"]
                error_message = _error_message_from_payload(error_payload)
                if _is_jito_rate_limit_message(error_message):
                    rate_error = JitoBundleRateLimitError(
                        f"Jito bundle submission rate-limited: {error_message}",
                        retry_after_seconds=retry_after_seconds,
                    )
                    last_error = rate_error
                    if idx < len(endpoints):
                        self._log_failover(
                            event="jito_bundle_endpoint_failover",
                            message="Switching Jito endpoint after bundle submit RPC rate limit",
                            from_endpoint=endpoint,
                            to_endpoint=endpoints[idx],
                            error=str(rate_error),
                        )
                        continue
                    raise rate_error
                last_error = RuntimeError(f"Jito bundle submission failed: {error_payload}")
                if idx < len(endpoints):
                    self._log_failover(
                        event="jito_bundle_endpoint_failover",
                        message="Switching Jito endpoint after bundle submit RPC error",
                        from_endpoint=endpoint,
                        to_endpoint=endpoints[idx],
                        error=str(last_error),
                    )
                    continue
                raise last_error

            bundle_id = None
            if isinstance(parsed, dict):
                result = parsed.get("result")
                if isinstance(result, str):
                    bundle_id = result
                elif isinstance(result, dict):
                    bundle_id = str(result.get("bundleId") or result.get("id") or "") or None

            self._promote_endpoint(endpoint)
            log_event(
                self._logger,
                level="info",
                event="jito_bundle_submitted",
                message="Atomic bundle submitted to Jito Block Engine",
                plan_id=plan_id,
                tx_count=len(signed_transactions),
                tip_lamports=max(0, int(tip_lamports)),
                bundle_id=bundle_id,
                endpoint=_endpoint_for_log(endpoint),
            )
            return bundle_id

        if last_error is not None:
            raise last_error
        raise RuntimeError("Jito bundle submission failed: no endpoint available.")


class AtomicExecutionCoordinator:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        pending_manager: AtomicPendingManager,
        jito_client: JitoBlockEngineClient,
    ) -> None:
        self._logger = logger
        self.pending_manager = pending_manager
        self.jito_client = jito_client

    async def submit_bundle(
        self,
        *,
        session: aiohttp.ClientSession,
        plan: AtomicExecutionPlan,
        signed_transactions: list[str],
    ) -> str | None:
        return await self.jito_client.send_bundle(
            session=session,
            signed_transactions=signed_transactions,
            tip_lamports=plan.tip_lamports,
            plan_id=plan.plan_id,
        )

    async def recover_pending(
        self,
        *,
        fetch_signature_status: Callable[[str], Awaitable[dict[str, Any] | None]],
        ttl_seconds: int,
        limit: int,
        stale_after_seconds: float | None = None,
    ) -> AtomicRecoverySummary:
        summary = await self.pending_manager.recover(
            fetch_signature_status=fetch_signature_status,
            ttl_seconds=ttl_seconds,
            limit=limit,
            stale_after_seconds=stale_after_seconds,
        )
        if summary.scanned > 0:
            log_event(
                self._logger,
                level="info",
                event="atomic_recovery_completed",
                message="Recovered pending atomic plan states",
                scanned=summary.scanned,
                resolved=summary.resolved,
                unresolved=summary.unresolved,
            )
        return summary
