from __future__ import annotations

import logging

from .types import (
    FAIL_REASON_ROUTE_UNSTABLE_REQUOTE_DROPPED,
    ExecutionResult,
    OrderExecutor,
    PairConfig,
    PriorityFeePlan,
    RuntimeConfig,
    SpreadObservation,
    TradeDecision,
    TradeIntent,
    build_idempotency_key,
    make_order_id,
)
from .watcher import HeliusQuoteWatcher


class TraderEngine:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        watcher: HeliusQuoteWatcher,
        executor: OrderExecutor,
    ) -> None:
        self._logger = logger
        self.watcher = watcher
        self.executor = executor

    async def healthcheck(self) -> None:
        await self.watcher.healthcheck()
        await self.executor.healthcheck()

    async def resolve_priority_fee(self, *, runtime_config: RuntimeConfig) -> PriorityFeePlan:
        return await self.executor.resolve_priority_fee(runtime_config=runtime_config)

    @staticmethod
    def build_idempotency_key(*, pair: PairConfig, observation: SpreadObservation) -> str:
        return build_idempotency_key(pair=pair, observation=observation)

    @staticmethod
    def _lamports_to_bps(*, lamports: int, notional_lamports: int) -> float:
        if notional_lamports <= 0:
            return 0.0
        return (max(0, lamports) / notional_lamports) * 10_000

    @staticmethod
    def _priority_fee_bps(
        *,
        priority_fee_micro_lamports: int,
        compute_units: int,
        notional_lamports: int,
    ) -> float:
        priority_fee_lamports = int((priority_fee_micro_lamports * compute_units) / 1_000_000)
        return TraderEngine._lamports_to_bps(
            lamports=priority_fee_lamports,
            notional_lamports=notional_lamports,
        )

    @staticmethod
    def _resolve_atomic_margin_single_bps(
        *,
        observed_spread_bps: float,
        base_margin_bps: float,
    ) -> float:
        base_margin = max(0.0, float(base_margin_bps))
        if observed_spread_bps < 2.0:
            return base_margin
        if observed_spread_bps < 6.0:
            return min(base_margin, 1.5)
        return min(base_margin, 1.0)

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

        # Initial exploration/gating excludes Jito tip and applies tip only at plan->send.
        tip_lamports = 0
        tip_fee_bps = 0.0
        initial_min_spread_bps = max(0.0, float(runtime_config.initial_min_spread_bps))

        atomic_margin_single_bps = self._resolve_atomic_margin_single_bps(
            observed_spread_bps=observation.spread_bps,
            base_margin_bps=runtime_config.initial_atomic_margin_bps,
        )
        atomic_margin_bundle_bps = atomic_margin_single_bps
        atomic_margin_bps = atomic_margin_single_bps

        total_fee_bps_single = runtime_config.dex_fee_bps + priority_fee_bps + atomic_margin_single_bps
        required_spread_bps_single = initial_min_spread_bps + total_fee_bps_single

        total_fee_bps_bundle = runtime_config.dex_fee_bps + priority_fee_bps + atomic_margin_bundle_bps
        required_spread_bps_bundle = initial_min_spread_bps + total_fee_bps_bundle

        expected_net_bps_single = observation.spread_bps - required_spread_bps_single
        expected_net_bps_bundle = observation.spread_bps - required_spread_bps_bundle
        expected_net_lamports_single = int((pair.base_amount * expected_net_bps_single) / 10_000)
        expected_net_lamports_bundle = int((pair.base_amount * expected_net_bps_bundle) / 10_000)

        profitable = expected_net_bps_single >= 0
        blocked_by_fee_cap = priority_fee_plan.exceeds_max
        blocked_by_route_instability = (
            (observation.fail_reason or "").strip().upper() == FAIL_REASON_ROUTE_UNSTABLE_REQUOTE_DROPPED
        )
        should_execute = (
            profitable
            and runtime_config.trade_enabled
            and not blocked_by_fee_cap
            and not blocked_by_route_instability
        )
        expected_net_bps = expected_net_bps_single

        if blocked_by_fee_cap:
            reason = "priority fee exceeds max_fee"
        elif blocked_by_route_instability:
            reason = "route rejected by median requote stability guard"
        elif should_execute:
            reason = "spread exceeded initial gate threshold"
        elif profitable and not runtime_config.trade_enabled:
            reason = "trade disabled"
        else:
            reason = "spread below initial gate threshold"

        return TradeDecision(
            profitable=profitable,
            should_execute=should_execute,
            spread_bps=observation.spread_bps,
            required_spread_bps=required_spread_bps_single,
            total_fee_bps=total_fee_bps_single,
            required_spread_bps_single=required_spread_bps_single,
            required_spread_bps_bundle=required_spread_bps_bundle,
            total_fee_bps_single=total_fee_bps_single,
            total_fee_bps_bundle=total_fee_bps_bundle,
            reason=reason,
            blocked_by_fee_cap=blocked_by_fee_cap,
            priority_fee_micro_lamports=priority_fee_plan.selected_micro_lamports,
            tip_lamports=tip_lamports,
            tip_fee_bps=tip_fee_bps,
            atomic_margin_bps=atomic_margin_bps,
            atomic_margin_single_bps=atomic_margin_single_bps,
            atomic_margin_bundle_bps=atomic_margin_bundle_bps,
            expected_net_bps=expected_net_bps,
            expected_net_bps_single=expected_net_bps_single,
            expected_net_bps_bundle=expected_net_bps_bundle,
            expected_net_lamports_single=expected_net_lamports_single,
            expected_net_lamports_bundle=expected_net_lamports_bundle,
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
            order_id = make_order_id(idempotency_key)
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
            runtime_config=runtime_config,
            metadata={
                "spread_bps": observation.spread_bps,
                "forward_out_amount": observation.forward_out_amount,
                "reverse_out_amount": observation.reverse_out_amount,
                "forward_quote": observation.forward_quote,
                "reverse_quote": observation.reverse_quote,
                "quote_params": observation.quote_params,
                "forward_route_dexes": list(observation.forward_route_dexes),
                "reverse_route_dexes": list(observation.reverse_route_dexes),
                "forward_route_hash": observation.forward_route_hash,
                "reverse_route_hash": observation.reverse_route_hash,
                "stage_a_pass": observation.stage_a_pass,
                "stage_b_pass": observation.stage_b_pass,
                "is_probe_trade": observation.is_probe_trade,
                "requote_samples_route_hashes_forward": list(
                    observation.requote_samples_route_hashes_forward
                ),
                "requote_samples_route_hashes_reverse": list(
                    observation.requote_samples_route_hashes_reverse
                ),
                "pair_slippage_bps": pair.slippage_bps,
                "base_mint": pair.base_mint,
                "quote_mint": pair.quote_mint,
                "base_amount": pair.base_amount,
                "base_decimals": pair.base_decimals,
                "quote_decimals": pair.quote_decimals,
                "execution_mode": runtime_config.execution_mode,
                "atomic_send_mode": runtime_config.atomic_send_mode,
                "atomic_expiry_ms": runtime_config.atomic_expiry_ms,
                "atomic_margin_bps": runtime_config.atomic_margin_bps,
                "allow_stageb_fail_probe": runtime_config.allow_stageb_fail_probe,
                "priority_fee_plan": priority_fee_plan.to_dict(),
            },
        )
