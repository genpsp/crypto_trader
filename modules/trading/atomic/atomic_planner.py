from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, Awaitable, Callable

from .atomic_types import AtomicExecutionPlan, AtomicResolvedMode, AtomicSendMode, LegQuote, make_plan_id, now_epoch_ms
from ..types import PairConfig, RuntimeConfig, SpreadObservation, now_iso, to_int


def normalize_send_mode(mode: str) -> AtomicSendMode:
    normalized = (mode or "").strip().lower()
    if normalized in {"single_tx", "bundle", "auto"}:
        return normalized  # type: ignore[return-value]
    return "auto"


def resolve_tip_lamports(runtime_config: RuntimeConfig, *, send_mode: AtomicSendMode) -> int:
    # Tip is applied only right before network submission in executor.
    return 0


def estimate_priority_fee_lamports(*, priority_fee_micro_lamports: int, compute_units: int) -> int:
    return max(0, int((priority_fee_micro_lamports * compute_units) / 1_000_000))


def lamports_to_bps(*, lamports: int, notional_lamports: int) -> float:
    if notional_lamports <= 0:
        return 0.0
    return (max(0, lamports) / notional_lamports) * 10_000


def resolve_atomic_margin_single_bps(*, observed_spread_bps: float, runtime_config: RuntimeConfig) -> float:
    if runtime_config.execution_mode != "atomic":
        return 0.0
    return max(0.0, runtime_config.atomic_margin_bps)


def safe_forward_output_amount(quote: dict[str, Any]) -> int:
    out_amount = to_int(quote.get("outAmount"), 0)
    min_out_amount = to_int(quote.get("otherAmountThreshold"), 0)
    if out_amount <= 0:
        return 0
    if min_out_amount <= 0:
        return out_amount
    return max(1, min(out_amount, min_out_amount))


def _extract_route_dexes(quote: dict[str, Any]) -> tuple[str, ...]:
    route_plan = quote.get("routePlan")
    if not isinstance(route_plan, list):
        return ()

    dexes: list[str] = []
    for hop in route_plan:
        if not isinstance(hop, dict):
            continue
        swap_info = hop.get("swapInfo")
        if not isinstance(swap_info, dict):
            continue
        label = str(swap_info.get("label") or "").strip()
        if not label:
            label = str(swap_info.get("ammKey") or "").strip()
        label = label.replace(",", " ")
        if label and label not in dexes:
            dexes.append(label)
    return tuple(dexes)


def _route_locked_quote_params(base_params: dict[str, Any], route_dexes: tuple[str, ...]) -> dict[str, Any]:
    params = dict(base_params)
    if route_dexes and "dexes" not in params:
        params["dexes"] = ",".join(route_dexes)
    return params


def _route_hash_from_dexes(route_dexes: tuple[str, ...]) -> str:
    if not route_dexes:
        return "unknown"
    payload = "|".join(route_dexes)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


class AtomicPlanner:
    async def refresh_observation(
        self,
        *,
        quote_callable: Callable[..., Awaitable[dict[str, Any]]],
        pair: PairConfig,
        seed_observation: SpreadObservation | None = None,
        override_quote_params: dict[str, Any] | None = None,
    ) -> SpreadObservation:
        if override_quote_params is not None:
            base_quote_params: dict[str, Any] = dict(override_quote_params)
        else:
            base_quote_params = dict(seed_observation.quote_params) if seed_observation else {}
        forward_quote_params = _route_locked_quote_params(
            base_quote_params,
            seed_observation.forward_route_dexes if seed_observation else (),
        )
        reverse_quote_params = _route_locked_quote_params(
            base_quote_params,
            seed_observation.reverse_route_dexes if seed_observation else (),
        )

        async def execute_quote(**kwargs: Any) -> dict[str, Any]:
            try:
                return await quote_callable(
                    **kwargs,
                    request_purpose="execution",
                    allow_during_pause=True,
                )
            except TypeError:
                return await quote_callable(**kwargs)

        forward_quote = await execute_quote(
            input_mint=pair.base_mint,
            output_mint=pair.quote_mint,
            amount=pair.base_amount,
            slippage_bps=pair.slippage_bps,
            extra_params=forward_quote_params,
        )
        forward_out = to_int(forward_quote.get("outAmount"), 0)
        if forward_out <= 0:
            raise RuntimeError("Atomic planner failed: forward quote outAmount is missing or invalid.")

        reverse_quote = await execute_quote(
            input_mint=pair.quote_mint,
            output_mint=pair.base_mint,
            amount=forward_out,
            slippage_bps=pair.slippage_bps,
            extra_params=reverse_quote_params,
        )
        reverse_out = to_int(reverse_quote.get("outAmount"), 0)
        if reverse_out <= 0:
            raise RuntimeError("Atomic planner failed: reverse quote outAmount is missing or invalid.")

        base_units = pair.base_amount / (10**pair.base_decimals)
        quote_units = forward_out / (10**pair.quote_decimals)
        forward_price = quote_units / base_units if base_units else 0.0

        spread_bps = ((reverse_out - pair.base_amount) / pair.base_amount) * 10_000
        forward_route_dexes = _extract_route_dexes(forward_quote)
        reverse_route_dexes = _extract_route_dexes(reverse_quote)

        return SpreadObservation(
            pair=pair.symbol,
            timestamp=now_iso(),
            forward_out_amount=forward_out,
            reverse_out_amount=reverse_out,
            forward_price=forward_price,
            spread_bps=spread_bps,
            forward_quote=forward_quote,
            reverse_quote=reverse_quote,
            quote_params=base_quote_params,
            forward_route_dexes=forward_route_dexes,
            reverse_route_dexes=reverse_route_dexes,
            forward_route_hash=_route_hash_from_dexes(forward_route_dexes),
            reverse_route_hash=_route_hash_from_dexes(reverse_route_dexes),
        )

    def build_plan(
        self,
        *,
        idempotency_key: str,
        pair: PairConfig,
        observation: SpreadObservation,
        runtime_config: RuntimeConfig,
        priority_fee_micro_lamports: int,
    ) -> AtomicExecutionPlan:
        created_at_ms = now_epoch_ms()
        expiry_ms = max(250, runtime_config.atomic_expiry_ms)

        send_mode = normalize_send_mode(runtime_config.atomic_send_mode)
        resolved_mode: AtomicResolvedMode = "bundle" if send_mode == "bundle" else "single_tx"

        priority_fee_lamports = estimate_priority_fee_lamports(
            priority_fee_micro_lamports=priority_fee_micro_lamports,
            compute_units=runtime_config.priority_compute_units,
        )
        tip_lamports = resolve_tip_lamports(runtime_config, send_mode=send_mode)

        priority_fee_bps = lamports_to_bps(
            lamports=priority_fee_lamports,
            notional_lamports=pair.base_amount,
        )

        atomic_margin_single_bps = resolve_atomic_margin_single_bps(
            observed_spread_bps=observation.spread_bps,
            runtime_config=runtime_config,
        )
        expected_fee_bps = (
            runtime_config.dex_fee_bps
            + priority_fee_bps
            + atomic_margin_single_bps
        )
        required_spread_bps = runtime_config.min_spread_bps + expected_fee_bps
        expected_net_bps = observation.spread_bps - required_spread_bps
        expected_net_lamports = int((pair.base_amount * expected_net_bps) / 10_000)

        forward_leg = LegQuote(
            leg="forward",
            input_mint=pair.base_mint,
            output_mint=pair.quote_mint,
            amount_in=pair.base_amount,
            amount_out=observation.forward_out_amount,
            slippage_bps=pair.slippage_bps,
            quote_response=observation.forward_quote,
        )
        reverse_leg = LegQuote(
            leg="reverse",
            input_mint=pair.quote_mint,
            output_mint=pair.base_mint,
            amount_in=observation.forward_out_amount,
            amount_out=observation.reverse_out_amount,
            slippage_bps=pair.slippage_bps,
            quote_response=observation.reverse_quote,
        )

        plan_id = make_plan_id(
            idempotency_key=idempotency_key,
            pair=pair.symbol,
            forward_context_slot=observation.forward_quote.get("contextSlot"),
            reverse_context_slot=observation.reverse_quote.get("contextSlot"),
            created_at_ms=created_at_ms,
        )

        return AtomicExecutionPlan(
            plan_id=plan_id,
            idempotency_key=idempotency_key,
            pair=pair.symbol,
            created_at_ms=created_at_ms,
            expires_at_ms=created_at_ms + expiry_ms,
            send_mode=send_mode,
            resolved_mode=resolved_mode,
            forward_leg=forward_leg,
            reverse_leg=reverse_leg,
            expected_spread_bps=observation.spread_bps,
            required_spread_bps=required_spread_bps,
            expected_fee_bps=expected_fee_bps,
            expected_net_bps=expected_net_bps,
            expected_net_lamports=expected_net_lamports,
            priority_fee_micro_lamports=max(0, priority_fee_micro_lamports),
            priority_fee_lamports=priority_fee_lamports,
            tip_lamports=tip_lamports,
            dex_fee_bps=runtime_config.dex_fee_bps,
            atomic_margin_bps=atomic_margin_single_bps,
            metadata={
                "forward_context_slot": observation.forward_quote.get("contextSlot"),
                "reverse_context_slot": observation.reverse_quote.get("contextSlot"),
                "quote_params": dict(observation.quote_params),
                "forward_route_dexes": list(observation.forward_route_dexes),
                "reverse_route_dexes": list(observation.reverse_route_dexes),
                "forward_route_hash": observation.forward_route_hash,
                "reverse_route_hash": observation.reverse_route_hash,
                "atomic_margin_single_bps": atomic_margin_single_bps,
            },
        )

    @staticmethod
    def force_bundle(plan: AtomicExecutionPlan) -> AtomicExecutionPlan:
        return replace(plan, resolved_mode="bundle")
