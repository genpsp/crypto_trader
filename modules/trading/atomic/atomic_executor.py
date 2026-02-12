from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

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

        for record in records:
            plan_id = str(record.get("plan_id") or "").strip()
            order_id = str(record.get("order_id") or "").strip()
            guard_key = str(record.get("guard_key") or "").strip()
            tx_signatures = record.get("tx_signatures")
            signatures = tx_signatures if isinstance(tx_signatures, list) else []
            signatures = [str(sig).strip() for sig in signatures if str(sig).strip()]

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
        self._block_engine_url = block_engine_url.strip()
        self._tip_accounts_cache: list[str] = []

    @property
    def block_engine_url(self) -> str:
        return self._block_engine_url

    def update_endpoint(self, block_engine_url: str) -> None:
        updated = (block_engine_url or "").strip()
        if updated != self._block_engine_url:
            self._tip_accounts_cache = []
        self._block_engine_url = updated

    async def fetch_tip_accounts(
        self,
        *,
        session: aiohttp.ClientSession,
        force_refresh: bool = False,
    ) -> list[str]:
        if self._tip_accounts_cache and not force_refresh:
            return list(self._tip_accounts_cache)

        if not self._block_engine_url:
            raise RuntimeError("JITO_BLOCK_ENGINE_URL is required for bundle mode.")

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTipAccounts",
            "params": [],
        }

        async with session.post(self._block_engine_url, json=payload) as response:
            status = response.status
            raw_text = await response.text()

        parsed: Any = None
        try:
            parsed = json.loads(raw_text) if raw_text else {}
        except json.JSONDecodeError:
            parsed = {"raw": raw_text}

        if status >= 400:
            raise RuntimeError(f"Jito getTipAccounts failed: status={status} body={str(raw_text)[:240]!r}")

        if isinstance(parsed, dict) and parsed.get("error"):
            raise RuntimeError(f"Jito getTipAccounts failed: {parsed['error']}")

        result = parsed.get("result") if isinstance(parsed, dict) else None
        if not isinstance(result, list):
            raise RuntimeError(f"Unexpected getTipAccounts response: {parsed}")

        tip_accounts = [str(item).strip() for item in result if str(item).strip()]
        if not tip_accounts:
            raise RuntimeError("Jito getTipAccounts returned no tip accounts.")

        self._tip_accounts_cache = tip_accounts
        log_event(
            self._logger,
            level="info",
            event="jito_tip_accounts_loaded",
            message="Loaded Jito tip accounts",
            tip_account_count=len(tip_accounts),
        )
        return list(tip_accounts)

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
        if not self._block_engine_url:
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

        async with session.post(self._block_engine_url, json=payload) as response:
            status = response.status
            retry_after_seconds = _parse_retry_after_seconds(response.headers.get("Retry-After"))
            raw_text = await response.text()

        parsed: Any = None
        try:
            parsed = json.loads(raw_text) if raw_text else {}
        except json.JSONDecodeError:
            parsed = {"raw": raw_text}

        if status == 429:
            raise JitoBundleRateLimitError(
                f"Jito bundle submission failed: status={status} body={str(raw_text)[:240]!r}",
                retry_after_seconds=retry_after_seconds,
            )

        if status >= 400:
            error_message = str(raw_text)
            if isinstance(parsed, dict) and parsed.get("error") is not None:
                error_message = _error_message_from_payload(parsed.get("error"))
            if _is_jito_rate_limit_message(error_message):
                raise JitoBundleRateLimitError(
                    f"Jito bundle submission rate-limited: status={status} error={error_message}",
                    retry_after_seconds=retry_after_seconds,
                )
            raise RuntimeError(
                f"Jito bundle submission failed: status={status} body={str(raw_text)[:240]!r}"
            )

        if isinstance(parsed, dict) and parsed.get("error"):
            error_payload = parsed["error"]
            error_message = _error_message_from_payload(error_payload)
            if _is_jito_rate_limit_message(error_message):
                raise JitoBundleRateLimitError(
                    f"Jito bundle submission rate-limited: {error_message}",
                    retry_after_seconds=retry_after_seconds,
                )
            raise RuntimeError(f"Jito bundle submission failed: {error_payload}")

        bundle_id = None
        if isinstance(parsed, dict):
            result = parsed.get("result")
            if isinstance(result, str):
                bundle_id = result
            elif isinstance(result, dict):
                bundle_id = str(result.get("bundleId") or result.get("id") or "") or None

        log_event(
            self._logger,
            level="info",
            event="jito_bundle_submitted",
            message="Atomic bundle submitted to Jito Block Engine",
            plan_id=plan_id,
            tx_count=len(signed_transactions),
            tip_lamports=max(0, int(tip_lamports)),
            bundle_id=bundle_id,
        )

        return bundle_id


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
    ) -> AtomicRecoverySummary:
        summary = await self.pending_manager.recover(
            fetch_signature_status=fetch_signature_status,
            ttl_seconds=ttl_seconds,
            limit=limit,
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
