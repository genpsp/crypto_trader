from __future__ import annotations

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
    if send_mode == "single_tx":
        return 0

    recommended = max(0, runtime_config.jito_tip_lamports_recommended)
    max_tip = max(0, runtime_config.jito_tip_lamports_max)
    if max_tip == 0:
        return recommended
    return min(recommended, max_tip)


def estimate_priority_fee_lamports(*, priority_fee_micro_lamports: int, compute_units: int) -> int:
    return max(0, int((priority_fee_micro_lamports * compute_units) / 1_000_000))


def lamports_to_bps(*, lamports: int, notional_lamports: int) -> float:
    if notional_lamports <= 0:
        return 0.0
    return (max(0, lamports) / notional_lamports) * 10_000


class AtomicPlanner:
    async def refresh_observation(
        self,
        *,
        quote_callable: Callable[..., Awaitable[dict[str, Any]]],
        pair: PairConfig,
    ) -> SpreadObservation:
        forward_quote = await quote_callable(
            input_mint=pair.base_mint,
            output_mint=pair.quote_mint,
            amount=pair.base_amount,
            slippage_bps=pair.slippage_bps,
        )
        forward_out = to_int(forward_quote.get("outAmount"), 0)
        if forward_out <= 0:
            raise RuntimeError("Atomic planner failed: forward quote outAmount is missing or invalid.")

        reverse_quote = await quote_callable(
            input_mint=pair.quote_mint,
            output_mint=pair.base_mint,
            amount=forward_out,
            slippage_bps=pair.slippage_bps,
        )
        reverse_out = to_int(reverse_quote.get("outAmount"), 0)
        if reverse_out <= 0:
            raise RuntimeError("Atomic planner failed: reverse quote outAmount is missing or invalid.")

        base_units = pair.base_amount / (10**pair.base_decimals)
        quote_units = forward_out / (10**pair.quote_decimals)
        forward_price = quote_units / base_units if base_units else 0.0

        spread_bps = ((reverse_out - pair.base_amount) / pair.base_amount) * 10_000

        return SpreadObservation(
            pair=pair.symbol,
            timestamp=now_iso(),
            forward_out_amount=forward_out,
            reverse_out_amount=reverse_out,
            forward_price=forward_price,
            spread_bps=spread_bps,
            forward_quote=forward_quote,
            reverse_quote=reverse_quote,
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
        tip_fee_bps = lamports_to_bps(
            lamports=tip_lamports,
            notional_lamports=pair.base_amount,
        )

        expected_fee_bps = (
            runtime_config.dex_fee_bps
            + priority_fee_bps
            + runtime_config.atomic_margin_bps
            + tip_fee_bps
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
            atomic_margin_bps=runtime_config.atomic_margin_bps,
            metadata={
                "forward_context_slot": observation.forward_quote.get("contextSlot"),
                "reverse_context_slot": observation.reverse_quote.get("contextSlot"),
            },
        )

    @staticmethod
    def force_bundle(plan: AtomicExecutionPlan) -> AtomicExecutionPlan:
        return replace(plan, resolved_mode="bundle")
