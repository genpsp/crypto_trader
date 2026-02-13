from __future__ import annotations

import logging

from .types import (
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
    def _resolve_tip_lamports(*, runtime_config: RuntimeConfig) -> int:
        if runtime_config.execution_mode != "atomic":
            return 0

        send_mode = runtime_config.atomic_send_mode
        if send_mode == "single_tx":
            return 0

        recommended = max(0, runtime_config.jito_tip_lamports_recommended)
        max_tip = max(0, runtime_config.jito_tip_lamports_max)
        if max_tip <= 0:
            return recommended
        return min(recommended, max_tip)

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

        tip_lamports = self._resolve_tip_lamports(runtime_config=runtime_config)
        tip_fee_bps = self._lamports_to_bps(lamports=tip_lamports, notional_lamports=pair.base_amount)

        atomic_margin_bps = runtime_config.atomic_margin_bps if runtime_config.execution_mode == "atomic" else 0.0

        total_fee_bps = runtime_config.dex_fee_bps + priority_fee_bps + tip_fee_bps + atomic_margin_bps
        required_spread_bps = runtime_config.min_spread_bps + total_fee_bps

        profitable = observation.spread_bps >= required_spread_bps
        blocked_by_fee_cap = priority_fee_plan.exceeds_max
        should_execute = profitable and runtime_config.trade_enabled and not blocked_by_fee_cap
        expected_net_bps = observation.spread_bps - required_spread_bps

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
            tip_lamports=tip_lamports,
            atomic_margin_bps=atomic_margin_bps,
            expected_net_bps=expected_net_bps,
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
                "priority_fee_plan": priority_fee_plan.to_dict(),
            },
        )
