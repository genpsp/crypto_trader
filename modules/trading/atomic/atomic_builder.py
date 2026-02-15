from __future__ import annotations

import base64
import logging
from typing import Any, Awaitable, Callable

from solders.transaction import VersionedTransaction

from modules.common import log_event

from .atomic_types import AtomicBuildArtifact, AtomicExecutionPlan, BuiltAtomicLeg


class AtomicBuildUnavailableError(RuntimeError):
    pass


class AtomicTransactionBuilder:
    def __init__(self, *, logger: logging.Logger) -> None:
        self._logger = logger

    @staticmethod
    def _extract_signature_from_signed_tx(signed_tx_base64: str) -> str:
        decoded = base64.b64decode(signed_tx_base64)
        tx = VersionedTransaction.from_bytes(decoded)
        if not tx.signatures:
            raise RuntimeError("Atomic build failed: signed transaction has no signatures.")
        return str(tx.signatures[0])

    async def build(
        self,
        *,
        plan: AtomicExecutionPlan,
        build_leg_tx: Callable[[dict[str, Any], int], Awaitable[dict[str, Any]]],
        build_single_tx: Callable[[AtomicExecutionPlan], Awaitable[AtomicBuildArtifact]] | None = None,
    ) -> AtomicBuildArtifact:
        if plan.resolved_mode == "single_tx":
            try:
                if build_single_tx is not None:
                    return await build_single_tx(plan)
                return await self._build_single_tx(plan=plan)
            except AtomicBuildUnavailableError as error:
                if plan.send_mode != "auto":
                    raise
                log_event(
                    self._logger,
                    level="warning",
                    event="atomic_single_tx_unavailable",
                    message="single_tx build was unavailable; falling back to bundle mode",
                    plan_id=plan.plan_id,
                    reason=str(error),
                )
                log_event(
                    self._logger,
                    level="warning",
                    event="atomic_fallback_to_bundle",
                    message="Falling back to bundle mode because single_tx build was unavailable",
                    plan_id=plan.plan_id,
                    send_mode_requested=plan.send_mode,
                    reason=str(error),
                )

        return await self._build_bundle(plan=plan, build_leg_tx=build_leg_tx)

    async def _build_single_tx(self, *, plan: AtomicExecutionPlan) -> AtomicBuildArtifact:
        raise AtomicBuildUnavailableError(
            "single_tx composition is unavailable with the current Jupiter swap API integration."
        )

    async def _build_bundle(
        self,
        *,
        plan: AtomicExecutionPlan,
        build_leg_tx: Callable[[dict[str, Any], int], Awaitable[dict[str, Any]]],
    ) -> AtomicBuildArtifact:
        forward = await build_leg_tx(plan.forward_leg.quote_response, plan.priority_fee_micro_lamports)
        reverse = await build_leg_tx(plan.reverse_leg.quote_response, plan.priority_fee_micro_lamports)

        forward_signed = str(forward.get("signed_tx_base64") or "").strip()
        reverse_signed = str(reverse.get("signed_tx_base64") or "").strip()
        if not forward_signed or not reverse_signed:
            raise RuntimeError("Atomic bundle build failed: signed transaction payload is missing.")

        forward_leg = BuiltAtomicLeg(
            leg="forward",
            signed_tx_base64=forward_signed,
            tx_signature=self._extract_signature_from_signed_tx(forward_signed),
            latest_blockhash=str(forward.get("latest_blockhash") or ""),
            last_valid_block_height=forward.get("last_valid_block_height"),
        )
        reverse_leg = BuiltAtomicLeg(
            leg="reverse",
            signed_tx_base64=reverse_signed,
            tx_signature=self._extract_signature_from_signed_tx(reverse_signed),
            latest_blockhash=str(reverse.get("latest_blockhash") or ""),
            last_valid_block_height=reverse.get("last_valid_block_height"),
        )

        return AtomicBuildArtifact(mode="bundle", legs=[forward_leg, reverse_leg])
