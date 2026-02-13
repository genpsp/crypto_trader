from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal

AtomicSendMode = Literal["single_tx", "bundle", "auto"]
AtomicResolvedMode = Literal["single_tx", "bundle"]


def now_epoch_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def make_plan_id(
    *,
    idempotency_key: str,
    pair: str,
    forward_context_slot: Any,
    reverse_context_slot: Any,
    created_at_ms: int,
) -> str:
    payload = {
        "idempotency_key": idempotency_key,
        "pair": pair,
        "forward_context_slot": forward_context_slot,
        "reverse_context_slot": reverse_context_slot,
        "created_at_ms": created_at_ms,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:40]


@dataclass(slots=True, frozen=True)
class LegQuote:
    leg: str
    input_mint: str
    output_mint: str
    amount_in: int
    amount_out: int
    slippage_bps: int
    quote_response: dict[str, Any]

    def compact_quote(self) -> dict[str, Any]:
        route_plan = self.quote_response.get("routePlan")
        return {
            "inAmount": self.quote_response.get("inAmount"),
            "outAmount": self.quote_response.get("outAmount"),
            "otherAmountThreshold": self.quote_response.get("otherAmountThreshold"),
            "priceImpactPct": self.quote_response.get("priceImpactPct"),
            "contextSlot": self.quote_response.get("contextSlot"),
            "slippageBps": self.quote_response.get("slippageBps"),
            "routeHopCount": len(route_plan) if isinstance(route_plan, list) else 0,
        }


@dataclass(slots=True, frozen=True)
class AtomicExecutionPlan:
    plan_id: str
    idempotency_key: str
    pair: str
    created_at_ms: int
    expires_at_ms: int
    send_mode: AtomicSendMode
    resolved_mode: AtomicResolvedMode
    forward_leg: LegQuote
    reverse_leg: LegQuote
    expected_spread_bps: float
    required_spread_bps: float
    expected_fee_bps: float
    expected_net_bps: float
    expected_net_lamports: int
    priority_fee_micro_lamports: int
    priority_fee_lamports: int
    tip_lamports: int
    dex_fee_bps: float
    atomic_margin_bps: float
    metadata: dict[str, Any]

    def is_expired(self, *, now_ms: int | None = None) -> bool:
        current = now_epoch_ms() if now_ms is None else now_ms
        return current > self.expires_at_ms

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["forward_leg"]["quote_response"] = self.forward_leg.compact_quote()
        payload["reverse_leg"]["quote_response"] = self.reverse_leg.compact_quote()
        return payload


@dataclass(slots=True, frozen=True)
class BuiltAtomicLeg:
    leg: str
    signed_tx_base64: str
    tx_signature: str
    latest_blockhash: str
    last_valid_block_height: int | None


@dataclass(slots=True, frozen=True)
class AtomicBuildArtifact:
    mode: AtomicResolvedMode
    legs: list[BuiltAtomicLeg]

    def tx_signatures(self) -> list[str]:
        return [leg.tx_signature for leg in self.legs]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class AtomicExecutionResult:
    status: str
    reason: str
    plan_id: str
    tx_signatures: list[str]
    confirmed: bool
    bundle_id: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
