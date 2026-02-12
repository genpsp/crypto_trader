from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
from typing import Any

import aiohttp
from solders.keypair import Keypair
from solders.message import to_bytes_versioned
from solders.transaction import VersionedTransaction

from modules.common import guarded_call, log_event

from .types import (
    ExecutionResult,
    OrderGuardStore,
    PriorityFeePlan,
    RuntimeConfig,
    TradeIntent,
    make_order_id,
    percentile_value,
    to_int,
)


class RpcMethodError(RuntimeError):
    def __init__(
        self,
        *,
        method: str,
        message: str,
        status: int | None = None,
        code: int | None = None,
        data: Any = None,
    ) -> None:
        super().__init__(message)
        self.method = method
        self.status = status
        self.code = code
        self.data = data


class TransactionExpiredError(RuntimeError):
    pass


class TransactionPendingConfirmationError(RuntimeError):
    def __init__(self, message: str, *, tx_signature: str | None = None) -> None:
        super().__init__(message)
        self.tx_signature = tx_signature


def _error_payload_to_message(payload: Any) -> str:
    if isinstance(payload, dict):
        message = payload.get("message")
        if isinstance(message, str) and message:
            return message
    return str(payload)


def _is_retryable_rpc_error(error: Exception) -> bool:
    if isinstance(error, (aiohttp.ClientError, asyncio.TimeoutError)):
        return True

    if isinstance(error, RpcMethodError):
        if error.status in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True
        if error.code in {-32005, -32004}:
            return True

    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "too many requests",
            "rate limit",
            "timeout",
            "temporarily unavailable",
            "connection reset",
            "connection aborted",
            "server error",
        )
    )


def _is_blockhash_related_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "blockhash not found",
            "block height exceeded",
            "blockheight exceeded",
            "transactionexpiredblockheightexceedederror",
            "transaction expired",
            "is no longer valid",
            "last valid block height",
        )
    )


def _normalize_swap_endpoint(url: str) -> str:
    normalized = url.strip()
    if not normalized:
        return "https://api.jup.ag/swap/v1/swap"

    if normalized.endswith("/quote"):
        return f"{normalized[:-6]}/swap"

    if "jup-proxy/swap/v1/quote" in normalized:
        return normalized.replace("jup-proxy/swap/v1/quote", "jup-proxy/swap/v1/swap")

    if normalized.endswith("/swap"):
        return normalized

    if "?" in normalized:
        head, query = normalized.split("?", 1)
        if head.endswith("/"):
            head = head[:-1]
        return f"{head}/swap?{query}"

    if normalized.endswith("/"):
        normalized = normalized[:-1]
    return f"{normalized}/swap"


def _digest_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _compact_quote_payload(raw_quote: Any) -> dict[str, Any] | None:
    if not isinstance(raw_quote, dict):
        return None

    route_plan = raw_quote.get("routePlan")
    return {
        "inAmount": raw_quote.get("inAmount"),
        "outAmount": raw_quote.get("outAmount"),
        "otherAmountThreshold": raw_quote.get("otherAmountThreshold"),
        "priceImpactPct": raw_quote.get("priceImpactPct"),
        "slippageBps": raw_quote.get("slippageBps"),
        "contextSlot": raw_quote.get("contextSlot"),
        "timeTaken": raw_quote.get("timeTaken"),
        "routeHopCount": len(route_plan) if isinstance(route_plan, list) else 0,
    }


def _compact_execution_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}

    compact = {key: value for key, value in metadata.items() if key not in {"forward_quote", "reverse_quote"}}

    forward_quote = metadata.get("forward_quote")
    reverse_quote = metadata.get("reverse_quote")

    forward_compact = _compact_quote_payload(forward_quote)
    if forward_compact is not None:
        compact["forward_quote"] = forward_compact
        compact["forward_quote_hash"] = _digest_payload(forward_quote)

    reverse_compact = _compact_quote_payload(reverse_quote)
    if reverse_compact is not None:
        compact["reverse_quote"] = reverse_compact
        compact["reverse_quote_hash"] = _digest_payload(reverse_quote)

    return compact


async def _cancel_task(
    task: asyncio.Task[None] | None,
    *,
    logger: logging.Logger,
    event: str,
) -> None:
    if task is None:
        return

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return
    except Exception as error:
        log_event(
            logger,
            level="warning",
            event=event,
            message="Background task shutdown failed",
            error=str(error),
        )


async def _record_order_state(
    *,
    order_store: OrderGuardStore | None,
    order_id: str,
    status: str,
    ttl_seconds: int,
    guard_key: str | None = None,
    payload: dict[str, Any] | None = None,
    logger: logging.Logger,
    event: str,
) -> None:
    if order_store is None:
        return

    await guarded_call(
        lambda: order_store.record_order_state(
            order_id=order_id,
            status=status,
            ttl_seconds=ttl_seconds,
            guard_key=guard_key,
            payload=payload,
        ),
        logger=logger,
        event=event,
        message="Failed to record order state",
        level="warning",
        order_id=order_id,
        status=status,
        guard_key=guard_key,
    )


async def _guard_refresh_loop(
    *,
    logger: logging.Logger,
    order_store: OrderGuardStore,
    guard_key: str,
    order_id: str,
    ttl_seconds: int,
) -> None:
    interval_seconds = max(1.0, min(5.0, ttl_seconds / 2))
    while True:
        await asyncio.sleep(interval_seconds)
        refreshed = await guarded_call(
            lambda: order_store.refresh_order_guard(
                guard_key=guard_key,
                order_id=order_id,
                ttl_seconds=ttl_seconds,
            ),
            logger=logger,
            event="order_guard_refresh_error",
            message="Order guard refresh failed with an exception",
            level="warning",
            default=None,
            guard_key=guard_key,
            order_id=order_id,
        )
        if refreshed is None:
            return
        if not refreshed:
            log_event(
                logger,
                level="warning",
                event="order_guard_refresh_failed",
                message="Order guard refresh failed; lock may have been lost",
                guard_key=guard_key,
                order_id=order_id,
            )
            return


class DryRunOrderExecutor:
    def __init__(self, logger: logging.Logger, order_store: OrderGuardStore | None = None) -> None:
        self._logger = logger
        self._order_store = order_store

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def healthcheck(self) -> None:
        return None

    async def resolve_priority_fee(self, *, runtime_config: RuntimeConfig) -> PriorityFeePlan:
        recommended = max(0, runtime_config.priority_fee_micro_lamports)
        max_fee = max(0, runtime_config.max_fee_micro_lamports)
        exceeds_max = recommended > max_fee
        selected = recommended if not exceeds_max else max_fee

        return PriorityFeePlan(
            selected_micro_lamports=selected,
            recommended_micro_lamports=recommended,
            max_fee_micro_lamports=max_fee,
            sample_size=0,
            source="dry_run_default",
            exceeds_max=exceeds_max,
        )

    async def execute(
        self,
        *,
        intent: TradeIntent,
        idempotency_key: str,
        lock_ttl_seconds: int,
        priority_fee_micro_lamports: int,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        order_id = make_order_id(idempotency_key)
        compact_metadata = _compact_execution_metadata(metadata)
        acquired_guard = False
        record_ttl_seconds = max(lock_ttl_seconds * 30, 300)
        guard_refresh_task: asyncio.Task[None] | None = None

        if self._order_store is not None:
            acquired_guard = await self._order_store.acquire_order_guard(
                guard_key=idempotency_key,
                order_id=order_id,
                ttl_seconds=lock_ttl_seconds,
            )
            if not acquired_guard:
                existing = await self._order_store.get_order_guard(guard_key=idempotency_key)
                existing_order_id = str(existing.get("order_id")) if existing else order_id
                return ExecutionResult(
                    status="skipped_duplicate",
                    tx_signature=None,
                    priority_fee_micro_lamports=priority_fee_micro_lamports,
                    reason="idempotency guard is active",
                    order_id=existing_order_id,
                    idempotency_key=idempotency_key,
                    metadata={"existing_guard": existing or {}},
                )

            guard_refresh_task = asyncio.create_task(
                _guard_refresh_loop(
                    logger=self._logger,
                    order_store=self._order_store,
                    guard_key=idempotency_key,
                    order_id=order_id,
                    ttl_seconds=lock_ttl_seconds,
                )
            )

            await _record_order_state(
                order_store=self._order_store,
                order_id=order_id,
                status="pending",
                ttl_seconds=record_ttl_seconds,
                guard_key=idempotency_key,
                payload=compact_metadata,
                logger=self._logger,
                event="dry_run_record_pending_failed",
            )

        try:
            log_event(
                self._logger,
                level="info",
                event="order_dry_run",
                message="Dry-run execution",
                pair=intent.pair,
                priority_fee_micro_lamports=priority_fee_micro_lamports,
                idempotency_key=idempotency_key,
                order_id=order_id,
            )

            result = ExecutionResult(
                status="dry_run",
                tx_signature=None,
                priority_fee_micro_lamports=priority_fee_micro_lamports,
                reason="DRY_RUN is enabled",
                order_id=order_id,
                idempotency_key=idempotency_key,
                metadata=compact_metadata,
            )

            await _record_order_state(
                order_store=self._order_store,
                order_id=order_id,
                status="dry_run",
                ttl_seconds=record_ttl_seconds,
                guard_key=idempotency_key,
                payload=result.to_dict(),
                logger=self._logger,
                event="dry_run_record_result_failed",
            )

            return result
        finally:
            await _cancel_task(
                guard_refresh_task,
                logger=self._logger,
                event="dry_run_guard_refresh_cancel_failed",
            )
            if acquired_guard and self._order_store is not None:
                await guarded_call(
                    lambda: self._order_store.release_order_guard(
                        guard_key=idempotency_key,
                        order_id=order_id,
                    ),
                    logger=self._logger,
                    event="dry_run_release_guard_failed",
                    message="Failed to release order guard",
                    guard_key=idempotency_key,
                    order_id=order_id,
                )


class LiveOrderExecutor:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        rpc_url: str,
        private_key: str,
        order_store: OrderGuardStore | None = None,
        swap_api_url: str = "https://api.jup.ag/swap/v1/swap",
        jupiter_api_key: str | None = None,
        send_max_attempts: int = 3,
        send_retry_backoff_seconds: float = 0.8,
        confirm_timeout_seconds: float = 45.0,
        confirm_poll_interval_seconds: float = 1.0,
        rebuild_max_attempts: int = 2,
        pending_guard_ttl_seconds: int = 180,
        pending_recovery_limit: int = 50,
        min_balance_lamports: int = 0,
    ) -> None:
        self._logger = logger
        self._rpc_url = rpc_url
        self._private_key = private_key
        self._order_store = order_store
        self._swap_api_url = _normalize_swap_endpoint(swap_api_url)
        self._jupiter_api_key = (jupiter_api_key or "").strip()
        self._send_max_attempts = max(1, send_max_attempts)
        self._send_retry_backoff_seconds = max(0.1, send_retry_backoff_seconds)
        self._confirm_timeout_seconds = max(5.0, confirm_timeout_seconds)
        self._confirm_poll_interval_seconds = max(0.25, confirm_poll_interval_seconds)
        self._rebuild_max_attempts = max(1, rebuild_max_attempts)
        self._pending_guard_ttl_seconds = max(30, pending_guard_ttl_seconds)
        self._pending_recovery_limit = max(1, pending_recovery_limit)
        self._min_balance_lamports = max(0, min_balance_lamports)
        self._signer: Keypair | None = None
        self._http_session: aiohttp.ClientSession | None = None

    async def connect(self) -> None:
        if not self._rpc_url:
            raise ValueError("SOLANA_RPC_URL is required when DRY_RUN is false.")
        if not self._private_key:
            raise ValueError("PRIVATE_KEY is required when DRY_RUN is false.")

        if self._http_session is None:
            timeout = aiohttp.ClientTimeout(total=8)
            self._http_session = aiohttp.ClientSession(timeout=timeout)

        self._signer = self._parse_private_key(self._private_key)
        await self._recover_pending_transactions()

    async def close(self) -> None:
        if self._http_session is not None:
            await self._http_session.close()
            self._http_session = None

    async def healthcheck(self) -> None:
        await self._fetch_latest_blockhash()

    async def resolve_priority_fee(self, *, runtime_config: RuntimeConfig) -> PriorityFeePlan:
        recommended = max(0, runtime_config.priority_fee_micro_lamports)
        source = "static"
        sample_size = 0

        try:
            recent_fees = await self._fetch_recent_prioritization_fees()
            sample_size = len(recent_fees)
            if recent_fees:
                percentile_fee = percentile_value(recent_fees, runtime_config.priority_fee_percentile)
                dynamic_candidate = int(percentile_fee * runtime_config.priority_fee_multiplier)
                recommended = max(recommended, dynamic_candidate)
                source = "recent_prioritization_fees"
        except Exception as error:
            source = "fallback_static"
            log_event(
                self._logger,
                level="warning",
                event="priority_fee_fallback",
                message="Falling back to static priority fee",
                error=str(error),
            )

        max_fee = max(0, runtime_config.max_fee_micro_lamports)
        exceeds_max = recommended > max_fee
        selected = recommended if not exceeds_max else max_fee

        return PriorityFeePlan(
            selected_micro_lamports=selected,
            recommended_micro_lamports=recommended,
            max_fee_micro_lamports=max_fee,
            sample_size=sample_size,
            source=source,
            exceeds_max=exceeds_max,
        )

    async def execute(
        self,
        *,
        intent: TradeIntent,
        idempotency_key: str,
        lock_ttl_seconds: int,
        priority_fee_micro_lamports: int,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        if self._order_store is None:
            raise RuntimeError("Order guard store is required for live execution.")

        if self._signer is None:
            await self.connect()
        if self._signer is None:
            raise RuntimeError("Signer is not initialized.")

        if self._min_balance_lamports > 0:
            wallet_balance = await self._fetch_wallet_balance()
            if wallet_balance < self._min_balance_lamports:
                reason = (
                    f"Wallet balance ({wallet_balance}) is below minimum "
                    f"required ({self._min_balance_lamports})"
                )
                log_event(
                    self._logger,
                    level="warning",
                    event="live_balance_guard_triggered",
                    message="Execution skipped due to low wallet balance",
                    wallet_balance_lamports=wallet_balance,
                    min_balance_lamports=self._min_balance_lamports,
                )
                return ExecutionResult(
                    status="skipped_low_balance",
                    tx_signature=None,
                    priority_fee_micro_lamports=priority_fee_micro_lamports,
                    reason=reason,
                    order_id=make_order_id(idempotency_key),
                    idempotency_key=idempotency_key,
                    metadata={
                        "wallet_balance_lamports": wallet_balance,
                        "min_balance_lamports": self._min_balance_lamports,
                    },
                )

        order_id = make_order_id(idempotency_key)
        compact_metadata = _compact_execution_metadata(metadata)
        guard_refresh_task: asyncio.Task[None] | None = None
        release_guard = True
        acquired_guard = await self._order_store.acquire_order_guard(
            guard_key=idempotency_key,
            order_id=order_id,
            ttl_seconds=lock_ttl_seconds,
        )

        if not acquired_guard:
            existing = await self._order_store.get_order_guard(guard_key=idempotency_key)
            existing_order_id = str(existing.get("order_id")) if existing else order_id
            return ExecutionResult(
                status="skipped_duplicate",
                tx_signature=None,
                priority_fee_micro_lamports=priority_fee_micro_lamports,
                reason="idempotency guard is active",
                order_id=existing_order_id,
                idempotency_key=idempotency_key,
                metadata={"existing_guard": existing or {}},
            )

        record_ttl_seconds = max(lock_ttl_seconds * 30, 300)
        pending_payload: dict[str, Any] = {
            "pair": intent.pair,
            "amount_in": intent.amount_in,
            "expected_amount_out": intent.expected_amount_out,
            "priority_fee_micro_lamports": priority_fee_micro_lamports,
            "metadata": compact_metadata,
        }

        try:
            guard_refresh_task = asyncio.create_task(
                _guard_refresh_loop(
                    logger=self._logger,
                    order_store=self._order_store,
                    guard_key=idempotency_key,
                    order_id=order_id,
                    ttl_seconds=lock_ttl_seconds,
                )
            )

            await _record_order_state(
                order_store=self._order_store,
                order_id=order_id,
                status="pending",
                ttl_seconds=record_ttl_seconds,
                guard_key=idempotency_key,
                payload=pending_payload,
                logger=self._logger,
                event="live_record_pending_failed",
            )

            quote_response = self._extract_quote_response(metadata)

            for build_attempt in range(1, self._rebuild_max_attempts + 1):
                try:
                    build_result = await self._build_signed_swap_transaction(
                        quote_response=quote_response,
                        priority_fee_micro_lamports=priority_fee_micro_lamports,
                    )
                    tx_signature = await self._send_transaction_with_retry(
                        signed_tx_base64=build_result["signed_tx_base64"],
                    )

                    await _record_order_state(
                        order_store=self._order_store,
                        order_id=order_id,
                        status="submitted",
                        ttl_seconds=record_ttl_seconds,
                        guard_key=idempotency_key,
                        payload={
                            **pending_payload,
                            "tx_signature": tx_signature,
                            "build_attempt": build_attempt,
                            "last_valid_block_height": build_result["last_valid_block_height"],
                            "latest_blockhash": build_result["latest_blockhash"],
                        },
                        logger=self._logger,
                        event="live_record_submitted_failed",
                    )

                    confirmation = await self._wait_for_confirmation(
                        tx_signature=tx_signature,
                        last_valid_block_height=build_result["last_valid_block_height"],
                    )

                    result_metadata = dict(compact_metadata)
                    result_metadata.update(
                        {
                            "tx_signature": tx_signature,
                            "build_attempt": build_attempt,
                            "latest_blockhash": build_result["latest_blockhash"],
                            "last_valid_block_height": build_result["last_valid_block_height"],
                            "confirmation": confirmation,
                        }
                    )

                    result = ExecutionResult(
                        status="filled",
                        tx_signature=tx_signature,
                        priority_fee_micro_lamports=priority_fee_micro_lamports,
                        reason="transaction confirmed",
                        order_id=order_id,
                        idempotency_key=idempotency_key,
                        metadata=result_metadata,
                    )

                    await _record_order_state(
                        order_store=self._order_store,
                        order_id=order_id,
                        status="confirmed",
                        ttl_seconds=record_ttl_seconds,
                        guard_key=idempotency_key,
                        payload=result.to_dict(),
                        logger=self._logger,
                        event="live_record_confirmed_failed",
                    )
                    return result
                except TransactionExpiredError as error:
                    if build_attempt < self._rebuild_max_attempts:
                        log_event(
                            self._logger,
                            level="warning",
                            event="live_blockhash_rebuild_retry",
                            message="Transaction expired before confirmation; rebuilding with a fresh blockhash",
                            order_id=order_id,
                            build_attempt=build_attempt,
                            max_attempts=self._rebuild_max_attempts,
                            error=str(error),
                        )
                        continue
                    raise
                except TransactionPendingConfirmationError as error:
                    release_guard = False
                    await _record_order_state(
                        order_store=self._order_store,
                        order_id=order_id,
                        status="pending_confirmation",
                        ttl_seconds=record_ttl_seconds,
                        guard_key=idempotency_key,
                        payload={
                            **pending_payload,
                            "tx_signature": error.tx_signature,
                            "reason": str(error),
                        },
                        logger=self._logger,
                        event="live_record_pending_confirmation_failed",
                    )
                    await guarded_call(
                        lambda: self._order_store.refresh_order_guard(
                            guard_key=idempotency_key,
                            order_id=order_id,
                            ttl_seconds=max(lock_ttl_seconds, self._pending_guard_ttl_seconds),
                        ),
                        logger=self._logger,
                        event="live_pending_guard_extend_failed",
                        message="Failed to extend order guard for pending confirmation",
                        level="warning",
                    )
                    raise
                except Exception as error:
                    retryable = _is_retryable_rpc_error(error)
                    if retryable and build_attempt < self._rebuild_max_attempts:
                        backoff = self._send_retry_backoff_seconds * build_attempt
                        log_event(
                            self._logger,
                            level="warning",
                            event="live_execution_retry",
                            message="Live execution attempt failed with a retryable error; retrying",
                            order_id=order_id,
                            build_attempt=build_attempt,
                            max_attempts=self._rebuild_max_attempts,
                            backoff_seconds=backoff,
                            error=str(error),
                        )
                        await asyncio.sleep(backoff)
                        continue
                    raise
        except TransactionPendingConfirmationError:
            raise
        except Exception as error:
            await _record_order_state(
                order_store=self._order_store,
                order_id=order_id,
                status="failed",
                ttl_seconds=record_ttl_seconds,
                guard_key=idempotency_key,
                payload={"error": str(error)},
                logger=self._logger,
                event="live_record_failed_state_failed",
            )
            raise
        finally:
            await _cancel_task(
                guard_refresh_task,
                logger=self._logger,
                event="live_guard_refresh_cancel_failed",
            )
            if release_guard:
                await guarded_call(
                    lambda: self._order_store.release_order_guard(
                        guard_key=idempotency_key,
                        order_id=order_id,
                    ),
                    logger=self._logger,
                    event="live_release_guard_failed",
                    message="Failed to release order guard",
                    guard_key=idempotency_key,
                    order_id=order_id,
                )

    def _extract_quote_response(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        candidate = (metadata or {}).get("forward_quote")
        if not isinstance(candidate, dict) or not candidate:
            raise ValueError("forward_quote is required in metadata for live execution.")
        return candidate

    def _swap_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._jupiter_api_key:
            headers["x-api-key"] = self._jupiter_api_key
        return headers

    async def _build_signed_swap_transaction(
        self,
        *,
        quote_response: dict[str, Any],
        priority_fee_micro_lamports: int,
    ) -> dict[str, Any]:
        if self._signer is None:
            await self.connect()
        if self._signer is None:
            raise RuntimeError("Signer is not initialized.")

        if self._http_session is None:
            await self.connect()
        if self._http_session is None:
            raise RuntimeError("HTTP session is not initialized.")

        swap_request_payload = {
            "quoteResponse": quote_response,
            "userPublicKey": str(self._signer.pubkey()),
            "dynamicComputeUnitLimit": True,
            "prioritizationFeeLamports": max(0, int(priority_fee_micro_lamports)),
        }

        try:
            async with self._http_session.post(
                self._swap_api_url,
                json=swap_request_payload,
                headers=self._swap_headers(),
            ) as response:
                status_code = response.status
                raw_text = await response.text()
        except Exception as error:
            if _is_retryable_rpc_error(error):
                raise RpcMethodError(method="swap", message=f"Swap API network error: {error}") from error
            raise

        parsed: Any = None
        try:
            parsed = json.loads(raw_text) if raw_text else {}
        except json.JSONDecodeError:
            parsed = {"raw_text": raw_text}

        if not isinstance(parsed, dict):
            raise RuntimeError(f"Unexpected swap API response: {parsed}")

        if status_code >= 400:
            error_payload = parsed.get("error")
            code = None
            if isinstance(error_payload, dict):
                code = to_int(error_payload.get("code"), 0) or None
            message = _error_payload_to_message(error_payload) if error_payload else str(parsed)
            raise RpcMethodError(
                method="swap",
                status=status_code,
                code=code,
                data=parsed,
                message=f"Swap API request failed: status={status_code} error={message}",
            )

        error_payload = parsed.get("error")
        if error_payload:
            message = _error_payload_to_message(error_payload)
            code = None
            if isinstance(error_payload, dict):
                code = to_int(error_payload.get("code"), 0) or None
            raise RpcMethodError(
                method="swap",
                message=f"Swap API error: {message}",
                status=None,
                code=code,
                data=parsed,
            )

        swap_tx_base64 = str(parsed.get("swapTransaction") or "").strip()
        if not swap_tx_base64:
            raise RuntimeError(f"swapTransaction is missing in swap API response: {parsed}")

        latest_blockhash = str(parsed.get("lastValidBlockhash") or "").strip()
        last_valid_block_height = to_int(parsed.get("lastValidBlockHeight"), -1)

        raw_tx = VersionedTransaction.from_bytes(base64.b64decode(swap_tx_base64))
        user_signature = self._signer.sign_message(to_bytes_versioned(raw_tx.message))
        signed_tx = VersionedTransaction.populate(raw_tx.message, [user_signature])
        signed_tx_base64 = base64.b64encode(bytes(signed_tx)).decode("ascii")

        return {
            "signed_tx_base64": signed_tx_base64,
            "latest_blockhash": latest_blockhash,
            "last_valid_block_height": last_valid_block_height if last_valid_block_height >= 0 else None,
        }

    async def _send_transaction_with_retry(self, *, signed_tx_base64: str) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self._send_max_attempts + 1):
            try:
                return await self._send_transaction_once(signed_tx_base64=signed_tx_base64)
            except Exception as error:
                last_error = error
                if _is_blockhash_related_error(error):
                    raise TransactionExpiredError(str(error)) from error

                retryable = _is_retryable_rpc_error(error)
                if retryable and attempt < self._send_max_attempts:
                    backoff = self._send_retry_backoff_seconds * attempt
                    log_event(
                        self._logger,
                        level="warning",
                        event="live_send_retry",
                        message="sendTransaction failed; retrying",
                        attempt=attempt,
                        max_attempts=self._send_max_attempts,
                        backoff_seconds=backoff,
                        error=str(error),
                    )
                    await asyncio.sleep(backoff)
                    continue

                raise

        raise RuntimeError(f"sendTransaction exhausted retries: {last_error}")

    async def _send_transaction_once(self, *, signed_tx_base64: str) -> str:
        options = {
            "encoding": "base64",
            "skipPreflight": False,
            "preflightCommitment": "processed",
            "maxRetries": 0,
        }
        result = await self._rpc_call("sendTransaction", [signed_tx_base64, options])
        if not isinstance(result, str) or not result:
            raise RuntimeError(f"sendTransaction returned unexpected payload: {result}")
        return result

    async def _wait_for_confirmation(
        self,
        *,
        tx_signature: str,
        last_valid_block_height: int | None,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._confirm_timeout_seconds
        next_block_height_check = loop.time()

        while loop.time() < deadline:
            try:
                status = await self._fetch_signature_status(tx_signature=tx_signature)
            except Exception as error:
                if _is_blockhash_related_error(error):
                    raise TransactionExpiredError(str(error)) from error
                if _is_retryable_rpc_error(error):
                    await asyncio.sleep(self._confirm_poll_interval_seconds)
                    continue
                raise

            if status is not None:
                err = status.get("err")
                if err is not None:
                    message = f"transaction failed: {err}"
                    if _is_blockhash_related_error(RuntimeError(message)):
                        raise TransactionExpiredError(message)
                    raise RuntimeError(message)

                confirmation_status = str(status.get("confirmationStatus") or "")
                if confirmation_status in {"confirmed", "finalized"}:
                    return status

            if last_valid_block_height is not None and loop.time() >= next_block_height_check:
                current_block_height = await self._fetch_block_height()
                if current_block_height > last_valid_block_height:
                    raise TransactionExpiredError(
                        "Transaction expired before confirmation "
                        f"(current_block_height={current_block_height}, "
                        f"last_valid_block_height={last_valid_block_height})"
                    )
                next_block_height_check = loop.time() + max(self._confirm_poll_interval_seconds, 1.0)

            await asyncio.sleep(self._confirm_poll_interval_seconds)

        raise TransactionPendingConfirmationError(
            "Transaction confirmation timed out; keeping guard active for recovery.",
            tx_signature=tx_signature,
        )

    async def _fetch_signature_status(self, *, tx_signature: str) -> dict[str, Any] | None:
        result = await self._rpc_call(
            "getSignatureStatuses",
            [[tx_signature], {"searchTransactionHistory": True}],
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected getSignatureStatuses response: {result}")

        value = result.get("value")
        if not isinstance(value, list) or not value:
            return None

        entry = value[0]
        if entry is None:
            return None
        if not isinstance(entry, dict):
            raise RuntimeError(f"Unexpected signature status entry: {entry}")
        return entry

    async def _fetch_block_height(self) -> int:
        result = await self._rpc_call("getBlockHeight", [{"commitment": "processed"}])
        block_height = to_int(result, -1)
        if block_height < 0:
            raise RuntimeError(f"Unexpected getBlockHeight response: {result}")
        return block_height

    async def _fetch_wallet_balance(self) -> int:
        if self._signer is None:
            await self.connect()
        if self._signer is None:
            raise RuntimeError("Signer is not initialized.")

        result = await self._rpc_call("getBalance", [str(self._signer.pubkey()), {"commitment": "processed"}])
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected getBalance response: {result}")

        value = to_int(result.get("value"), -1)
        if value < 0:
            raise RuntimeError(f"Unexpected getBalance payload: {result}")
        return value

    async def _recover_pending_transactions(self) -> None:
        if self._order_store is None:
            return

        list_records = getattr(self._order_store, "list_order_records", None)
        if not callable(list_records):
            return

        records = await list_records(
            statuses={"submitted", "pending_confirmation", "confirming"},
            limit=self._pending_recovery_limit,
        )
        if not records:
            return

        resolved_count = 0
        unresolved_count = 0

        for record in records:
            order_id = str(record.get("order_id", "")).strip()
            guard_key = str(record.get("guard_key", "")).strip()
            payload = record.get("payload")
            payload_dict = payload if isinstance(payload, dict) else {}
            tx_signature = str(payload_dict.get("tx_signature", "")).strip()
            if not order_id or not guard_key or not tx_signature:
                continue

            try:
                status = await self._fetch_signature_status(tx_signature=tx_signature)
            except Exception as error:
                log_event(
                    self._logger,
                    level="warning",
                    event="live_recovery_status_check_failed",
                    message="Failed to check pending transaction during recovery",
                    order_id=order_id,
                    tx_signature=tx_signature,
                    error=str(error),
                )
                continue

            if status is None:
                unresolved_count += 1
                await guarded_call(
                    lambda: self._order_store.refresh_order_guard(
                        guard_key=guard_key,
                        order_id=order_id,
                        ttl_seconds=self._pending_guard_ttl_seconds,
                    ),
                    logger=self._logger,
                    event="live_recovery_guard_refresh_failed",
                    message="Failed to refresh guard during pending transaction recovery",
                    level="warning",
                )
                continue

            err = status.get("err")
            if err is None:
                confirmation_status = str(status.get("confirmationStatus") or "")
                if confirmation_status in {"confirmed", "finalized"}:
                    resolved_count += 1
                    await _record_order_state(
                        order_store=self._order_store,
                        order_id=order_id,
                        status="confirmed",
                        ttl_seconds=self._pending_guard_ttl_seconds * 2,
                        guard_key=guard_key,
                        payload={"tx_signature": tx_signature, "confirmation": status, "recovered": True},
                        logger=self._logger,
                        event="live_recovery_record_confirmed_failed",
                    )
                    await guarded_call(
                        lambda: self._order_store.release_order_guard(
                            guard_key=guard_key,
                            order_id=order_id,
                        ),
                        logger=self._logger,
                        event="live_recovery_release_guard_failed",
                        message="Failed to release guard for recovered confirmed transaction",
                        level="warning",
                    )
                    continue

            if err is not None:
                resolved_count += 1
                await _record_order_state(
                    order_store=self._order_store,
                    order_id=order_id,
                    status="failed",
                    ttl_seconds=self._pending_guard_ttl_seconds * 2,
                    guard_key=guard_key,
                    payload={"tx_signature": tx_signature, "error": err, "recovered": True},
                    logger=self._logger,
                    event="live_recovery_record_failed_state_failed",
                )
                await guarded_call(
                    lambda: self._order_store.release_order_guard(
                        guard_key=guard_key,
                        order_id=order_id,
                    ),
                    logger=self._logger,
                    event="live_recovery_release_failed_guard_failed",
                    message="Failed to release guard for recovered failed transaction",
                    level="warning",
                )
                continue

            unresolved_count += 1
            await guarded_call(
                lambda: self._order_store.refresh_order_guard(
                    guard_key=guard_key,
                    order_id=order_id,
                    ttl_seconds=self._pending_guard_ttl_seconds,
                ),
                logger=self._logger,
                event="live_recovery_guard_extend_failed",
                message="Failed to extend guard for unresolved recovered transaction",
                level="warning",
            )

        log_event(
            self._logger,
            level="info",
            event="live_recovery_completed",
            message="Recovered pending transaction states",
            recovered_records=len(records),
            resolved_count=resolved_count,
            unresolved_count=unresolved_count,
        )

    async def _rpc_call(self, method: str, params: list[Any] | None = None) -> Any:
        if self._http_session is None:
            await self.connect()
        if self._http_session is None:
            raise RuntimeError("RPC HTTP session is not initialized.")

        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or [],
        }

        async with self._http_session.post(self._rpc_url, json=payload) as response:
            raw_text = await response.text()

        parsed: Any = None
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(raw_text) if raw_text else {}
        if parsed is None:
            parsed = {"raw_text": raw_text}

        if not isinstance(parsed, dict):
            raise RpcMethodError(method=method, message=f"Invalid RPC response for {method}: {parsed}")

        if response.status >= 400:
            error_payload = parsed.get("error")
            code = None
            if isinstance(error_payload, dict):
                code = to_int(error_payload.get("code"), 0) or None
            message = _error_payload_to_message(error_payload) if error_payload else str(parsed)
            raise RpcMethodError(
                method=method,
                status=response.status,
                code=code,
                data=parsed,
                message=f"RPC call failed: method={method} status={response.status} error={message}",
            )

        if parsed.get("error"):
            error_payload = parsed.get("error")
            code = to_int(error_payload.get("code"), 0) if isinstance(error_payload, dict) else None
            message = _error_payload_to_message(error_payload)
            raise RpcMethodError(
                method=method,
                status=response.status,
                code=code,
                data=parsed,
                message=f"RPC error for {method}: {message}",
            )

        return parsed.get("result")

    async def _fetch_latest_blockhash(self) -> str:
        result = await self._rpc_call("getLatestBlockhash", [{"commitment": "processed"}])
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected getLatestBlockhash response: {result}")

        value = result.get("value")
        if not isinstance(value, dict):
            raise RuntimeError(f"Unexpected getLatestBlockhash payload: {result}")

        blockhash = value.get("blockhash")
        if not isinstance(blockhash, str) or not blockhash:
            raise RuntimeError(f"Missing blockhash in RPC response: {result}")

        return blockhash

    async def _fetch_recent_prioritization_fees(self) -> list[int]:
        result = await self._rpc_call("getRecentPrioritizationFees", [[]])
        if not isinstance(result, list):
            raise RuntimeError(f"Unexpected getRecentPrioritizationFees response: {result}")

        fees: list[int] = []
        for item in result:
            if not isinstance(item, dict):
                continue
            fee = to_int(item.get("prioritizationFee"), 0)
            if fee >= 0:
                fees.append(fee)

        return fees

    def _parse_private_key(self, raw: str) -> Keypair:
        value = raw.strip()

        if value.startswith("["):
            arr = json.loads(value)
            if not isinstance(arr, list):
                raise ValueError("PRIVATE_KEY JSON must be an integer array.")
            return Keypair.from_bytes(bytes(arr))

        with contextlib.suppress(Exception):
            return Keypair.from_base58_string(value)

        raise ValueError("Unsupported PRIVATE_KEY format.")
