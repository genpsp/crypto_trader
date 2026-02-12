from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


def to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp_percentile(value: float) -> float:
    return max(0.0, min(1.0, value))


def percentile_value(values: list[int], percentile: float) -> int:
    if not values:
        return 0

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    p = clamp_percentile(percentile)
    index = max(0, min(len(sorted_values) - 1, math.ceil(len(sorted_values) * p) - 1))
    return sorted_values[index]


def make_order_id(idempotency_key: str, blockhash: str | None = None) -> str:
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
            base_decimals=to_int(os.getenv("PAIR_BASE_DECIMALS"), 9),
            quote_decimals=to_int(os.getenv("PAIR_QUOTE_DECIMALS"), 6),
            base_amount=to_int(os.getenv("PAIR_BASE_AMOUNT"), 1_000_000_000),
            slippage_bps=to_int(os.getenv("PAIR_SLIPPAGE_BPS"), 20),
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
    config_schema_version: int
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
            config_schema_version=max(1, to_int(os.getenv("CONFIG_SCHEMA_VERSION"), 1)),
            min_spread_bps=to_float(os.getenv("MIN_SPREAD_BPS"), 5.0),
            dex_fee_bps=to_float(os.getenv("DEX_FEE_BPS"), 4.0),
            priority_fee_micro_lamports=to_int(os.getenv("PRIORITY_FEE_MICRO_LAMPORTS"), 10_000),
            priority_compute_units=to_int(os.getenv("PRIORITY_COMPUTE_UNITS"), 200_000),
            priority_fee_percentile=clamp_percentile(to_float(os.getenv("PRIORITY_FEE_PERCENTILE"), 0.75)),
            priority_fee_multiplier=max(0.0, to_float(os.getenv("PRIORITY_FEE_MULTIPLIER"), 1.15)),
            max_fee_micro_lamports=to_int(os.getenv("MAX_FEE_MICRO_LAMPORTS"), 80_000),
            trade_enabled=to_bool(os.getenv("TRADE_ENABLED"), False),
            order_guard_ttl_seconds=max(1, to_int(os.getenv("ORDER_GUARD_TTL_SECONDS"), 20)),
        )

    @classmethod
    def from_redis(cls, redis_config: dict[str, str], defaults: "RuntimeConfig") -> "RuntimeConfig":
        schema_raw = redis_config.get("schema_version") or redis_config.get("config_schema_version")
        max_fee_raw = redis_config.get("max_fee_micro_lamports") or redis_config.get("max_fee")

        return cls(
            config_schema_version=max(
                1,
                to_int(schema_raw, defaults.config_schema_version),
            ),
            min_spread_bps=to_float(
                redis_config.get("min_spread_bps") or redis_config.get("min_spread"),
                defaults.min_spread_bps,
            ),
            dex_fee_bps=to_float(redis_config.get("dex_fee_bps"), defaults.dex_fee_bps),
            priority_fee_micro_lamports=to_int(
                redis_config.get("priority_fee_micro_lamports"),
                defaults.priority_fee_micro_lamports,
            ),
            priority_compute_units=to_int(
                redis_config.get("priority_compute_units"),
                defaults.priority_compute_units,
            ),
            priority_fee_percentile=clamp_percentile(
                to_float(redis_config.get("priority_fee_percentile"), defaults.priority_fee_percentile)
            ),
            priority_fee_multiplier=max(
                0.0,
                to_float(redis_config.get("priority_fee_multiplier"), defaults.priority_fee_multiplier),
            ),
            max_fee_micro_lamports=to_int(max_fee_raw, defaults.max_fee_micro_lamports),
            trade_enabled=to_bool(redis_config.get("trade_enabled"), defaults.trade_enabled),
            order_guard_ttl_seconds=max(
                1,
                to_int(
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
    ) -> bool:
        ...

    async def get_order_guard(self, *, guard_key: str) -> dict[str, Any] | None:
        ...

    async def refresh_order_guard(
        self,
        *,
        guard_key: str,
        order_id: str,
        ttl_seconds: int,
    ) -> bool:
        ...

    async def release_order_guard(
        self,
        *,
        guard_key: str,
        order_id: str,
    ) -> bool:
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

    async def list_order_records(
        self,
        *,
        statuses: set[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
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
