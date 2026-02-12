from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any

import aiohttp
from solana.rpc.async_api import AsyncClient
from solders.compute_budget import set_compute_unit_price
from solders.keypair import Keypair

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
                payload=metadata,
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
                metadata=metadata or {},
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
    ) -> None:
        self._logger = logger
        self._rpc_url = rpc_url
        self._private_key = private_key
        self._order_store = order_store
        self._rpc_client: AsyncClient | None = None
        self._signer: Keypair | None = None
        self._http_session: aiohttp.ClientSession | None = None

    async def connect(self) -> None:
        if not self._rpc_url:
            raise ValueError("SOLANA_RPC_URL is required when DRY_RUN is false.")
        if not self._private_key:
            raise ValueError("PRIVATE_KEY is required when DRY_RUN is false.")

        if self._rpc_client is None:
            self._rpc_client = AsyncClient(self._rpc_url)
        if self._http_session is None:
            timeout = aiohttp.ClientTimeout(total=8)
            self._http_session = aiohttp.ClientSession(timeout=timeout)

        self._signer = self._parse_private_key(self._private_key)

    async def close(self) -> None:
        if self._rpc_client is not None:
            await self._rpc_client.close()
            self._rpc_client = None

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

        latest_blockhash = await self._fetch_latest_blockhash()
        order_id = make_order_id(idempotency_key, latest_blockhash)
        guard_refresh_task: asyncio.Task[None] | None = None
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
                payload={
                    "pair": intent.pair,
                    "amount_in": intent.amount_in,
                    "expected_amount_out": intent.expected_amount_out,
                    "latest_blockhash": latest_blockhash,
                    "priority_fee_micro_lamports": priority_fee_micro_lamports,
                    "metadata": metadata or {},
                },
                logger=self._logger,
                event="live_record_pending_failed",
            )

            _ = set_compute_unit_price(priority_fee_micro_lamports)

            log_event(
                self._logger,
                level="warning",
                event="order_live_placeholder",
                message="Live execution path is a placeholder",
                pair=intent.pair,
                order_id=order_id,
                idempotency_key=idempotency_key,
            )

            result_metadata = metadata.copy() if metadata else {}
            result_metadata["latest_blockhash"] = latest_blockhash

            result = ExecutionResult(
                status="live_placeholder",
                tx_signature=None,
                priority_fee_micro_lamports=priority_fee_micro_lamports,
                reason="Swap transaction builder is not implemented yet",
                order_id=order_id,
                idempotency_key=idempotency_key,
                metadata=result_metadata,
            )

            await _record_order_state(
                order_store=self._order_store,
                order_id=order_id,
                status="live_placeholder",
                ttl_seconds=record_ttl_seconds,
                guard_key=idempotency_key,
                payload=result.to_dict(),
                logger=self._logger,
                event="live_record_placeholder_failed",
            )

            return result
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
            body = await response.json(content_type=None)
            if response.status >= 400:
                raise RuntimeError(f"RPC call failed: method={method} status={response.status} body={body}")

        if not isinstance(body, dict):
            raise RuntimeError(f"Invalid RPC response for {method}: {body}")

        if body.get("error"):
            raise RuntimeError(f"RPC error for {method}: {body['error']}")

        return body.get("result")

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
