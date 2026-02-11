from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import aiohttp
from solana.rpc.async_api import AsyncClient
from solders.compute_budget import set_compute_unit_price
from solders.keypair import Keypair

SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _to_int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp_percentile(value: float) -> float:
    return max(0.0, min(1.0, value))


def _percentile_value(values: list[int], percentile: float) -> int:
    if not values:
        return 0

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    p = _clamp_percentile(percentile)
    index = max(0, min(len(sorted_values) - 1, math.ceil(len(sorted_values) * p) - 1))
    return sorted_values[index]


def _make_order_id(idempotency_key: str, blockhash: str | None = None) -> str:
    tail = blockhash[:16] if blockhash else datetime.now(timezone.utc).strftime("%H%M%S%f")
    return f"ord-{idempotency_key[:18]}-{tail}"


@dataclass(slots=True, frozen=True)
class PairConfig:
    symbol: str
    base_mint: str
    quote_mint: str
    base_decimals: int
    quote_decimals: int
    base_amount: int
    slippage_bps: int

    @classmethod
    def from_env(cls) -> "PairConfig":
        return cls(
            symbol=os.getenv("PAIR_SYMBOL", "SOL/USDC"),
            base_mint=os.getenv("PAIR_BASE_MINT", SOL_MINT),
            quote_mint=os.getenv("PAIR_QUOTE_MINT", USDC_MINT),
            base_decimals=_to_int(os.getenv("PAIR_BASE_DECIMALS"), 9),
            quote_decimals=_to_int(os.getenv("PAIR_QUOTE_DECIMALS"), 6),
            base_amount=_to_int(os.getenv("PAIR_BASE_AMOUNT"), 1_000_000_000),
            slippage_bps=_to_int(os.getenv("PAIR_SLIPPAGE_BPS"), 20),
        )


@dataclass(slots=True, frozen=True)
class SpreadObservation:
    pair: str
    timestamp: str
    forward_out_amount: int
    reverse_out_amount: int
    forward_price: float
    spread_bps: float
    forward_quote: dict[str, Any]
    reverse_quote: dict[str, Any]


@dataclass(slots=True, frozen=True)
class RuntimeConfig:
    min_spread_bps: float
    dex_fee_bps: float
    priority_fee_micro_lamports: int
    priority_compute_units: int
    priority_fee_percentile: float
    priority_fee_multiplier: float
    max_fee_micro_lamports: int
    trade_enabled: bool
    order_guard_ttl_seconds: int

    @classmethod
    def from_env_defaults(cls) -> "RuntimeConfig":
        return cls(
            min_spread_bps=_to_float(os.getenv("MIN_SPREAD_BPS"), 5.0),
            dex_fee_bps=_to_float(os.getenv("DEX_FEE_BPS"), 4.0),
            priority_fee_micro_lamports=_to_int(os.getenv("PRIORITY_FEE_MICRO_LAMPORTS"), 10_000),
            priority_compute_units=_to_int(os.getenv("PRIORITY_COMPUTE_UNITS"), 200_000),
            priority_fee_percentile=_clamp_percentile(_to_float(os.getenv("PRIORITY_FEE_PERCENTILE"), 0.75)),
            priority_fee_multiplier=max(0.0, _to_float(os.getenv("PRIORITY_FEE_MULTIPLIER"), 1.15)),
            max_fee_micro_lamports=_to_int(os.getenv("MAX_FEE_MICRO_LAMPORTS"), 80_000),
            trade_enabled=_to_bool(os.getenv("TRADE_ENABLED"), False),
            order_guard_ttl_seconds=max(1, _to_int(os.getenv("ORDER_GUARD_TTL_SECONDS"), 20)),
        )

    @classmethod
    def from_redis(cls, redis_config: dict[str, str], defaults: "RuntimeConfig") -> "RuntimeConfig":
        max_fee_raw = redis_config.get("max_fee_micro_lamports") or redis_config.get("max_fee")

        return cls(
            min_spread_bps=_to_float(
                redis_config.get("min_spread_bps") or redis_config.get("min_spread"),
                defaults.min_spread_bps,
            ),
            dex_fee_bps=_to_float(redis_config.get("dex_fee_bps"), defaults.dex_fee_bps),
            priority_fee_micro_lamports=_to_int(
                redis_config.get("priority_fee_micro_lamports"),
                defaults.priority_fee_micro_lamports,
            ),
            priority_compute_units=_to_int(
                redis_config.get("priority_compute_units"),
                defaults.priority_compute_units,
            ),
            priority_fee_percentile=_clamp_percentile(
                _to_float(redis_config.get("priority_fee_percentile"), defaults.priority_fee_percentile)
            ),
            priority_fee_multiplier=max(
                0.0,
                _to_float(redis_config.get("priority_fee_multiplier"), defaults.priority_fee_multiplier),
            ),
            max_fee_micro_lamports=_to_int(max_fee_raw, defaults.max_fee_micro_lamports),
            trade_enabled=_to_bool(redis_config.get("trade_enabled"), defaults.trade_enabled),
            order_guard_ttl_seconds=max(
                1,
                _to_int(
                    redis_config.get("order_guard_ttl_seconds"),
                    defaults.order_guard_ttl_seconds,
                ),
            ),
        )


@dataclass(slots=True, frozen=True)
class PriorityFeePlan:
    selected_micro_lamports: int
    recommended_micro_lamports: int
    max_fee_micro_lamports: int
    sample_size: int
    source: str
    exceeds_max: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class TradeDecision:
    profitable: bool
    should_execute: bool
    spread_bps: float
    required_spread_bps: float
    total_fee_bps: float
    reason: str
    blocked_by_fee_cap: bool
    priority_fee_micro_lamports: int


@dataclass(slots=True, frozen=True)
class TradeIntent:
    pair: str
    input_mint: str
    output_mint: str
    amount_in: int
    expected_amount_out: int


@dataclass(slots=True, frozen=True)
class ExecutionResult:
    status: str
    tx_signature: str | None
    priority_fee_micro_lamports: int
    reason: str
    order_id: str
    idempotency_key: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OrderGuardStore(Protocol):
    async def acquire_order_guard(
        self,
        *,
        guard_key: str,
        order_id: str,
        ttl_seconds: int,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        ...

    async def get_order_guard(self, *, guard_key: str) -> dict[str, Any] | None:
        ...

    async def release_order_guard(self, *, guard_key: str) -> None:
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


class OrderExecutor(Protocol):
    async def connect(self) -> None:
        ...

    async def close(self) -> None:
        ...

    async def healthcheck(self) -> None:
        ...

    async def resolve_priority_fee(self, *, runtime_config: RuntimeConfig) -> PriorityFeePlan:
        ...

    async def execute(
        self,
        *,
        intent: TradeIntent,
        idempotency_key: str,
        lock_ttl_seconds: int,
        priority_fee_micro_lamports: int,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        ...


class JupiterWatcher:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        api_base_url: str,
        timeout_seconds: float = 8.0,
    ) -> None:
        self._logger = logger
        self._api_base_url = api_base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._session: aiohttp.ClientSession | None = None

    async def connect(self) -> None:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def healthcheck(self) -> None:
        await self.connect()

    async def quote(
        self,
        *,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int,
    ) -> dict[str, Any]:
        if self._session is None:
            await self.connect()
        if self._session is None:
            raise RuntimeError("Jupiter HTTP session is not initialized.")

        endpoint = f"{self._api_base_url}/quote"
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageBps": str(slippage_bps),
        }

        async with self._session.get(endpoint, params=params) as response:
            data = await response.json(content_type=None)
            if response.status >= 400:
                raise RuntimeError(f"Jupiter quote failed: status={response.status} body={data}")

        if "outAmount" not in data:
            raise RuntimeError(f"Unexpected Jupiter response: {data}")

        return data

    async def fetch_spread(self, pair: PairConfig) -> SpreadObservation:
        forward_quote = await self.quote(
            input_mint=pair.base_mint,
            output_mint=pair.quote_mint,
            amount=pair.base_amount,
            slippage_bps=pair.slippage_bps,
        )
        forward_out = int(forward_quote["outAmount"])

        reverse_quote = await self.quote(
            input_mint=pair.quote_mint,
            output_mint=pair.base_mint,
            amount=forward_out,
            slippage_bps=pair.slippage_bps,
        )
        reverse_out = int(reverse_quote["outAmount"])

        base_units = pair.base_amount / (10**pair.base_decimals)
        quote_units = forward_out / (10**pair.quote_decimals)
        forward_price = quote_units / base_units if base_units else 0.0

        spread_bps = ((reverse_out - pair.base_amount) / pair.base_amount) * 10_000

        return SpreadObservation(
            pair=pair.symbol,
            timestamp=_now_iso(),
            forward_out_amount=forward_out,
            reverse_out_amount=reverse_out,
            forward_price=forward_price,
            spread_bps=spread_bps,
            forward_quote=forward_quote,
            reverse_quote=reverse_quote,
        )


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
        order_id = _make_order_id(idempotency_key)
        acquired_guard = False
        record_ttl_seconds = max(lock_ttl_seconds * 30, 300)

        if self._order_store is not None:
            acquired_guard = await self._order_store.acquire_order_guard(
                guard_key=idempotency_key,
                order_id=order_id,
                ttl_seconds=lock_ttl_seconds,
                payload={
                    "pair": intent.pair,
                    "priority_fee_micro_lamports": priority_fee_micro_lamports,
                    "mode": "dry_run",
                },
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

            await self._order_store.record_order_state(
                order_id=order_id,
                status="pending",
                ttl_seconds=record_ttl_seconds,
                guard_key=idempotency_key,
                payload=metadata,
            )

        try:
            self._logger.info(
                "Dry-run execution",
                extra={
                    "event": "order_dry_run",
                    "pair": intent.pair,
                    "priority_fee_micro_lamports": priority_fee_micro_lamports,
                    "idempotency_key": idempotency_key,
                    "order_id": order_id,
                },
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

            if self._order_store is not None:
                await self._order_store.record_order_state(
                    order_id=order_id,
                    status="dry_run",
                    ttl_seconds=record_ttl_seconds,
                    guard_key=idempotency_key,
                    payload=result.to_dict(),
                )

            return result
        finally:
            if acquired_guard and self._order_store is not None:
                await self._order_store.release_order_guard(guard_key=idempotency_key)


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
            raise ValueError("RPC_URL is required when DRY_RUN is false.")
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
                percentile_fee = _percentile_value(recent_fees, runtime_config.priority_fee_percentile)
                dynamic_candidate = int(percentile_fee * runtime_config.priority_fee_multiplier)
                recommended = max(recommended, dynamic_candidate)
                source = "recent_prioritization_fees"
        except Exception as error:
            source = "fallback_static"
            self._logger.warning(
                "Falling back to static priority fee",
                extra={"event": "priority_fee_fallback", "error": str(error)},
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
        order_id = _make_order_id(idempotency_key, latest_blockhash)
        acquired_guard = await self._order_store.acquire_order_guard(
            guard_key=idempotency_key,
            order_id=order_id,
            ttl_seconds=lock_ttl_seconds,
            payload={
                "pair": intent.pair,
                "latest_blockhash": latest_blockhash,
                "priority_fee_micro_lamports": priority_fee_micro_lamports,
                "mode": "live",
            },
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
            await self._order_store.record_order_state(
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
            )

            _ = set_compute_unit_price(priority_fee_micro_lamports)
            jupiter_sdk_available = False
            with contextlib.suppress(Exception):
                importlib.import_module("jupiter_python_sdk")
                jupiter_sdk_available = True

            self._logger.warning(
                "Live execution path is a placeholder",
                extra={
                    "event": "order_live_placeholder",
                    "pair": intent.pair,
                    "order_id": order_id,
                    "idempotency_key": idempotency_key,
                },
            )

            result_metadata = metadata.copy() if metadata else {}
            result_metadata["jupiter_sdk_available"] = jupiter_sdk_available
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

            await self._order_store.record_order_state(
                order_id=order_id,
                status="live_placeholder",
                ttl_seconds=record_ttl_seconds,
                guard_key=idempotency_key,
                payload=result.to_dict(),
            )

            return result
        except Exception as error:
            await self._order_store.record_order_state(
                order_id=order_id,
                status="failed",
                ttl_seconds=record_ttl_seconds,
                guard_key=idempotency_key,
                payload={"error": str(error)},
            )
            raise
        finally:
            await self._order_store.release_order_guard(guard_key=idempotency_key)

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
            fee = _to_int(item.get("prioritizationFee"), 0)
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


class TraderEngine:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        watcher: JupiterWatcher,
        executor: OrderExecutor,
    ) -> None:
        self._logger = logger
        self.watcher = watcher
        self.executor = executor

    async def healthcheck(self) -> None:
        await self.watcher.healthcheck()
        await self.executor.healthcheck()

    @staticmethod
    def build_idempotency_key(*, pair: PairConfig, observation: SpreadObservation) -> str:
        fingerprint = {
            "pair": pair.symbol,
            "input_mint": pair.base_mint,
            "output_mint": pair.quote_mint,
            "amount_in": pair.base_amount,
            "slippage_bps": pair.slippage_bps,
            "forward_out_amount": observation.forward_out_amount,
            "reverse_out_amount": observation.reverse_out_amount,
            "spread_bps": round(observation.spread_bps, 4),
            "forward_context_slot": observation.forward_quote.get("contextSlot"),
            "reverse_context_slot": observation.reverse_quote.get("contextSlot"),
        }
        encoded = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    async def resolve_priority_fee(self, *, runtime_config: RuntimeConfig) -> PriorityFeePlan:
        return await self.executor.resolve_priority_fee(runtime_config=runtime_config)

    @staticmethod
    def _priority_fee_bps(
        *,
        priority_fee_micro_lamports: int,
        compute_units: int,
        notional_lamports: int,
    ) -> float:
        if notional_lamports <= 0:
            return 0.0
        priority_fee_lamports = (priority_fee_micro_lamports * compute_units) / 1_000_000
        return (priority_fee_lamports / notional_lamports) * 10_000

    def evaluate(
        self,
        *,
        observation: SpreadObservation,
        runtime_config: RuntimeConfig,
        pair: PairConfig,
        priority_fee_plan: PriorityFeePlan,
    ) -> TradeDecision:
        effective_priority_fee = priority_fee_plan.recommended_micro_lamports
        priority_fee_bps = self._priority_fee_bps(
            priority_fee_micro_lamports=effective_priority_fee,
            compute_units=runtime_config.priority_compute_units,
            notional_lamports=pair.base_amount,
        )
        total_fee_bps = runtime_config.dex_fee_bps + priority_fee_bps
        required_spread_bps = runtime_config.min_spread_bps + total_fee_bps

        profitable = observation.spread_bps >= required_spread_bps
        blocked_by_fee_cap = priority_fee_plan.exceeds_max
        should_execute = profitable and runtime_config.trade_enabled and not blocked_by_fee_cap

        if blocked_by_fee_cap:
            reason = "priority fee exceeds max_fee"
        elif should_execute:
            reason = "spread threshold exceeded"
        elif profitable and not runtime_config.trade_enabled:
            reason = "trade disabled"
        else:
            reason = "spread below threshold"

        return TradeDecision(
            profitable=profitable,
            should_execute=should_execute,
            spread_bps=observation.spread_bps,
            required_spread_bps=required_spread_bps,
            total_fee_bps=total_fee_bps,
            reason=reason,
            blocked_by_fee_cap=blocked_by_fee_cap,
            priority_fee_micro_lamports=priority_fee_plan.selected_micro_lamports,
        )

    async def execute(
        self,
        *,
        pair: PairConfig,
        observation: SpreadObservation,
        runtime_config: RuntimeConfig,
        priority_fee_plan: PriorityFeePlan,
        idempotency_key: str,
    ) -> ExecutionResult:
        if priority_fee_plan.exceeds_max:
            order_id = _make_order_id(idempotency_key)
            return ExecutionResult(
                status="skipped_max_fee",
                tx_signature=None,
                priority_fee_micro_lamports=priority_fee_plan.recommended_micro_lamports,
                reason="Priority fee exceeds max_fee",
                order_id=order_id,
                idempotency_key=idempotency_key,
                metadata=priority_fee_plan.to_dict(),
            )

        intent = TradeIntent(
            pair=pair.symbol,
            input_mint=pair.base_mint,
            output_mint=pair.quote_mint,
            amount_in=pair.base_amount,
            expected_amount_out=observation.forward_out_amount,
        )

        return await self.executor.execute(
            intent=intent,
            idempotency_key=idempotency_key,
            lock_ttl_seconds=runtime_config.order_guard_ttl_seconds,
            priority_fee_micro_lamports=priority_fee_plan.selected_micro_lamports,
            metadata={
                "spread_bps": observation.spread_bps,
                "forward_out_amount": observation.forward_out_amount,
                "reverse_out_amount": observation.reverse_out_amount,
                "priority_fee_plan": priority_fee_plan.to_dict(),
            },
        )
