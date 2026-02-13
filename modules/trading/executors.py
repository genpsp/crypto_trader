from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
from dataclasses import replace
from typing import Any

from solders.address_lookup_table_account import AddressLookupTableAccount
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.message import MessageV0, to_bytes_versioned
from solders.pubkey import Pubkey
from solders.system_program import TransferParams, transfer
from solders.transaction import VersionedTransaction

from modules.common import guarded_call, log_event

from .atomic import (
    AtomicBuildArtifact,
    AtomicBuildUnavailableError,
    AtomicExecutionCoordinator,
    AtomicExecutionPlan,
    AtomicPendingManager,
    AtomicPlanner,
    AtomicTransactionBuilder,
    BuiltAtomicLeg,
    JitoBlockEngineClient,
    JitoBundleRateLimitError,
)
from .base_executors import (
    DryRunOrderExecutor,
    LiveOrderExecutor,
    RpcMethodError,
    TransactionPendingConfirmationError,
    _error_payload_to_message,
    _is_retryable_rpc_error,
    _cancel_task,
    _compact_execution_metadata,
    _guard_refresh_loop,
    _record_order_state,
)
from .types import (
    ExecutionResult,
    OrderGuardStore,
    PairConfig,
    RuntimeConfig,
    TradeIntent,
    make_order_id,
    normalize_atomic_send_mode,
    to_int,
)
from .watcher import HeliusQuoteWatcher


class LiveAtomicArbExecutor(LiveOrderExecutor):
    def __init__(
        self,
        *,
        logger: logging.Logger,
        rpc_url: str,
        private_key: str,
        watcher: HeliusQuoteWatcher,
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
        atomic_send_mode: str = "auto",
        atomic_expiry_ms: int = 5000,
        atomic_margin_bps: float = 20.0,
        jito_block_engine_url: str = "",
        jito_tip_lamports_max: int = 100_000,
        jito_tip_lamports_recommended: int = 20_000,
    ) -> None:
        super().__init__(
            logger=logger,
            rpc_url=rpc_url,
            private_key=private_key,
            order_store=order_store,
            swap_api_url=swap_api_url,
            jupiter_api_key=jupiter_api_key,
            send_max_attempts=send_max_attempts,
            send_retry_backoff_seconds=send_retry_backoff_seconds,
            confirm_timeout_seconds=confirm_timeout_seconds,
            confirm_poll_interval_seconds=confirm_poll_interval_seconds,
            rebuild_max_attempts=rebuild_max_attempts,
            pending_guard_ttl_seconds=pending_guard_ttl_seconds,
            pending_recovery_limit=pending_recovery_limit,
            min_balance_lamports=min_balance_lamports,
        )
        self._watcher = watcher
        self._default_atomic_send_mode = normalize_atomic_send_mode(atomic_send_mode)
        self._default_atomic_expiry_ms = max(250, int(atomic_expiry_ms))
        self._default_atomic_margin_bps = max(0.0, float(atomic_margin_bps))
        self._default_jito_block_engine_url = (jito_block_engine_url or "").strip()
        self._default_jito_tip_lamports_max = max(0, int(jito_tip_lamports_max))
        self._default_jito_tip_lamports_recommended = max(0, int(jito_tip_lamports_recommended))
        self._atomic_planner = AtomicPlanner()
        self._atomic_builder = AtomicTransactionBuilder(logger=logger)
        self._bundle_rate_limited_count = 0
        self._bundle_backoff_until = 0.0
        pending_manager = AtomicPendingManager(logger=logger, store=order_store)
        jito_client = JitoBlockEngineClient(logger=logger, block_engine_url=self._default_jito_block_engine_url)
        self._atomic_coordinator = AtomicExecutionCoordinator(
            logger=logger,
            pending_manager=pending_manager,
            jito_client=jito_client,
        )

    async def connect(self) -> None:
        await super().connect()
        await self.recover_pending()

    async def recover_pending(self) -> None:
        stale_after_seconds = min(
            float(self._pending_guard_ttl_seconds),
            max(60.0, float(self._confirm_timeout_seconds) + 30.0),
        )
        await self._atomic_coordinator.recover_pending(
            fetch_signature_status=lambda tx_signature: self._fetch_signature_status(
                tx_signature=tx_signature,
                search_transaction_history=True,
            ),
            ttl_seconds=self._pending_guard_ttl_seconds,
            limit=self._pending_recovery_limit,
            stale_after_seconds=stale_after_seconds,
        )

    def _compute_bundle_backoff_seconds(self, *, retry_after_seconds: float | None) -> float:
        exponential = min(30.0, float(2 ** max(0, self._bundle_rate_limited_count - 1)))
        jitter = random.uniform(0.0, max(0.1, exponential * 0.25))
        adaptive = min(30.0, exponential + jitter)
        if retry_after_seconds is None:
            return adaptive
        return min(30.0, max(adaptive, retry_after_seconds))

    def _apply_bundle_rate_limit_backoff(self, *, retry_after_seconds: float | None) -> float:
        self._bundle_rate_limited_count += 1
        backoff_seconds = self._compute_bundle_backoff_seconds(retry_after_seconds=retry_after_seconds)
        now = asyncio.get_running_loop().time()
        self._bundle_backoff_until = max(self._bundle_backoff_until, now + backoff_seconds)
        return backoff_seconds

    def _reset_bundle_rate_limit_backoff(self) -> None:
        self._bundle_rate_limited_count = 0
        self._bundle_backoff_until = 0.0

    async def _submit_bundle_with_retry(
        self,
        *,
        plan: Any,
        signed_transactions: list[str],
    ) -> str | None:
        if self._http_session is None:
            await self.connect()
        if self._http_session is None:
            raise RuntimeError("HTTP session is not initialized for bundle submission.")

        last_error: Exception | None = None
        for attempt in range(1, self._send_max_attempts + 1):
            try:
                return await self._atomic_coordinator.submit_bundle(
                    session=self._http_session,
                    plan=plan,
                    signed_transactions=signed_transactions,
                )
            except JitoBundleRateLimitError:
                raise
            except Exception as error:
                last_error = error
                if attempt >= self._send_max_attempts:
                    break
                jitter = random.uniform(0.0, self._send_retry_backoff_seconds * 0.25)
                backoff = min(3.0, self._send_retry_backoff_seconds * attempt + jitter)
                log_event(
                    self._logger,
                    level="warning",
                    event="jito_bundle_submit_retry",
                    message="Jito bundle submission failed with a retryable error; retrying",
                    plan_id=plan.plan_id,
                    attempt=attempt,
                    max_attempts=self._send_max_attempts,
                    backoff_seconds=round(backoff, 3),
                    error=str(error),
                )
                await asyncio.sleep(backoff)

        raise RuntimeError(f"Jito bundle submission exhausted retries: {last_error}")

    async def _wait_for_atomic_confirmations(
        self,
        *,
        tx_signatures: list[str],
        last_valid_block_height_by_signature: dict[str, int | None],
    ) -> list[dict[str, Any]]:
        tasks: list[asyncio.Task[dict[str, Any]]] = []
        for tx_signature in tx_signatures:
            tasks.append(
                asyncio.create_task(
                    self._wait_for_confirmation(
                        tx_signature=tx_signature,
                        last_valid_block_height=last_valid_block_height_by_signature.get(tx_signature),
                    )
                )
            )

        try:
            return list(await asyncio.gather(*tasks))
        except Exception:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    def _effective_runtime_config(self, runtime_config: RuntimeConfig) -> RuntimeConfig:
        atomic_send_mode = normalize_atomic_send_mode(
            runtime_config.atomic_send_mode or self._default_atomic_send_mode
        )
        atomic_expiry_ms = (
            runtime_config.atomic_expiry_ms
            if runtime_config.atomic_expiry_ms > 0
            else self._default_atomic_expiry_ms
        )
        atomic_margin_bps = (
            runtime_config.atomic_margin_bps
            if runtime_config.atomic_margin_bps >= 0
            else self._default_atomic_margin_bps
        )
        jito_block_engine_url = (
            runtime_config.jito_block_engine_url.strip()
            if runtime_config.jito_block_engine_url.strip()
            else self._default_jito_block_engine_url
        )
        jito_tip_lamports_max = (
            runtime_config.jito_tip_lamports_max
            if runtime_config.jito_tip_lamports_max > 0
            else self._default_jito_tip_lamports_max
        )
        jito_tip_lamports_recommended = (
            runtime_config.jito_tip_lamports_recommended
            if runtime_config.jito_tip_lamports_recommended > 0
            else self._default_jito_tip_lamports_recommended
        )

        return replace(
            runtime_config,
            atomic_send_mode=atomic_send_mode,
            atomic_expiry_ms=max(250, int(atomic_expiry_ms)),
            atomic_margin_bps=max(0.0, float(atomic_margin_bps)),
            jito_block_engine_url=jito_block_engine_url,
            jito_tip_lamports_max=max(0, int(jito_tip_lamports_max)),
            jito_tip_lamports_recommended=max(0, int(jito_tip_lamports_recommended)),
        )

    @staticmethod
    def _pair_from_metadata(intent: TradeIntent, metadata: dict[str, Any] | None) -> PairConfig:
        payload = metadata or {}
        return PairConfig(
            symbol=intent.pair,
            base_mint=str(payload.get("base_mint") or intent.input_mint),
            quote_mint=str(payload.get("quote_mint") or intent.output_mint),
            base_decimals=max(0, to_int(payload.get("base_decimals"), 9)),
            quote_decimals=max(0, to_int(payload.get("quote_decimals"), 6)),
            base_amount=max(1, to_int(payload.get("base_amount"), intent.amount_in)),
            slippage_bps=max(1, to_int(payload.get("pair_slippage_bps"), 20)),
        )

    async def _build_leg_transaction(
        self,
        quote_response: dict[str, Any],
        priority_fee_micro_lamports: int,
    ) -> dict[str, Any]:
        return await self._build_signed_swap_transaction(
            quote_response=quote_response,
            priority_fee_micro_lamports=priority_fee_micro_lamports,
        )

    @staticmethod
    def _swap_instructions_api_url(swap_api_url: str) -> str:
        url = (swap_api_url or "").strip()
        if not url:
            return "https://api.jup.ag/swap/v1/swap-instructions"

        if "/swap/v1/swap" in url:
            return url.replace("/swap/v1/swap", "/swap/v1/swap-instructions")
        if url.endswith("/swap"):
            return f"{url[:-5]}/swap-instructions"

        if "?" in url:
            head, query = url.split("?", 1)
            if head.endswith("/"):
                head = head[:-1]
            if head.endswith("/swap"):
                head = head[:-5]
            return f"{head}/swap-instructions?{query}"

        if url.endswith("/"):
            url = url[:-1]
        return f"{url}/swap-instructions"

    @staticmethod
    def _decode_instruction(raw: Any, *, section: str) -> Instruction:
        if not isinstance(raw, dict):
            raise RuntimeError(f"Invalid instruction payload in {section}: {raw}")

        program_id = str(raw.get("programId") or "").strip()
        if not program_id:
            raise RuntimeError(f"Instruction programId is missing in {section}")

        raw_accounts = raw.get("accounts")
        if not isinstance(raw_accounts, list):
            raise RuntimeError(f"Instruction accounts are missing in {section}")

        metas: list[AccountMeta] = []
        for idx, account in enumerate(raw_accounts):
            if not isinstance(account, dict):
                raise RuntimeError(f"Instruction account[{idx}] is invalid in {section}: {account}")
            pubkey = str(account.get("pubkey") or "").strip()
            if not pubkey:
                raise RuntimeError(f"Instruction account[{idx}] pubkey is missing in {section}")
            metas.append(
                AccountMeta(
                    pubkey=Pubkey.from_string(pubkey),
                    is_signer=bool(account.get("isSigner")),
                    is_writable=bool(account.get("isWritable")),
                )
            )

        encoded_data = str(raw.get("data") or "")
        try:
            data = base64.b64decode(encoded_data)
        except Exception as error:  # pragma: no cover - malformed upstream payload
            raise RuntimeError(f"Instruction data decode failed in {section}: {error}") from error

        return Instruction(Pubkey.from_string(program_id), data, metas)

    @classmethod
    def _decode_instruction_list(cls, raw: Any, *, section: str) -> list[Instruction]:
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise RuntimeError(f"Instruction list is invalid in {section}: {raw}")
        return [cls._decode_instruction(item, section=f"{section}[{index}]") for index, item in enumerate(raw)]

    @classmethod
    def _extract_leg_instructions(
        cls,
        *,
        payload: dict[str, Any],
        leg: str,
    ) -> tuple[list[Instruction], list[str]]:
        instructions: list[Instruction] = []
        token_ledger = payload.get("tokenLedgerInstruction")
        if token_ledger:
            instructions.append(cls._decode_instruction(token_ledger, section=f"{leg}.tokenLedgerInstruction"))
        instructions.extend(
            cls._decode_instruction_list(
                payload.get("setupInstructions"),
                section=f"{leg}.setupInstructions",
            )
        )
        instructions.extend(
            cls._decode_instruction_list(
                payload.get("otherInstructions"),
                section=f"{leg}.otherInstructions",
            )
        )

        swap_instruction = payload.get("swapInstruction")
        if not isinstance(swap_instruction, dict):
            raise RuntimeError(f"swapInstruction is missing for {leg} leg.")
        instructions.append(cls._decode_instruction(swap_instruction, section=f"{leg}.swapInstruction"))

        cleanup_instruction = payload.get("cleanupInstruction")
        if cleanup_instruction:
            instructions.append(cls._decode_instruction(cleanup_instruction, section=f"{leg}.cleanupInstruction"))

        lookup_addresses: list[str] = []
        raw_lookup_addresses = payload.get("addressLookupTableAddresses")
        if isinstance(raw_lookup_addresses, list):
            for raw_address in raw_lookup_addresses:
                address = str(raw_address or "").strip()
                if address:
                    lookup_addresses.append(address)

        return instructions, lookup_addresses

    @staticmethod
    def _dedupe_instructions(instructions: list[Instruction]) -> list[Instruction]:
        deduped: list[Instruction] = []
        seen: set[bytes] = set()
        for instruction in instructions:
            fingerprint = bytes(instruction)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            deduped.append(instruction)
        return deduped

    async def _fetch_swap_instructions_payload(
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

        endpoint = self._swap_instructions_api_url(self._swap_api_url)
        payload = {
            "quoteResponse": quote_response,
            "userPublicKey": str(self._signer.pubkey()),
            "dynamicComputeUnitLimit": False,
            "prioritizationFeeLamports": max(0, int(priority_fee_micro_lamports)),
        }

        try:
            async with self._http_session.post(
                endpoint,
                json=payload,
                headers=self._swap_headers(),
            ) as response:
                status_code = response.status
                raw_text = await response.text()
        except Exception as error:
            if _is_retryable_rpc_error(error):
                raise RpcMethodError(
                    method="swap-instructions",
                    message=f"Swap-instructions API network error: {error}",
                ) from error
            raise

        parsed: Any = None
        try:
            parsed = json.loads(raw_text) if raw_text else {}
        except json.JSONDecodeError:
            parsed = {"raw_text": raw_text}

        if not isinstance(parsed, dict):
            raise RuntimeError(f"Unexpected swap-instructions response: {parsed}")

        if status_code >= 400:
            error_payload = parsed.get("error")
            error_code = None
            if isinstance(error_payload, dict):
                error_code = to_int(error_payload.get("code"), 0) or None
            error_message = _error_payload_to_message(error_payload) if error_payload else str(parsed)
            raise RpcMethodError(
                method="swap-instructions",
                status=status_code,
                code=error_code,
                data=parsed,
                message=(
                    "Swap-instructions API request failed: "
                    f"status={status_code} error={error_message}"
                ),
            )

        error_payload = parsed.get("error")
        if error_payload:
            error_code = to_int(error_payload.get("code"), 0) if isinstance(error_payload, dict) else None
            error_message = _error_payload_to_message(error_payload)
            raise RpcMethodError(
                method="swap-instructions",
                code=error_code,
                data=parsed,
                message=f"Swap-instructions API error: {error_message}",
            )

        if not isinstance(parsed.get("swapInstruction"), dict):
            raise RuntimeError(f"swapInstruction is missing in swap-instructions response: {parsed}")

        return parsed

    async def _fetch_latest_blockhash_with_height(self) -> tuple[str, int | None]:
        result = await self._rpc_call("getLatestBlockhash", [{"commitment": "processed"}])
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected getLatestBlockhash response: {result}")

        value = result.get("value")
        if not isinstance(value, dict):
            raise RuntimeError(f"Unexpected getLatestBlockhash payload: {result}")

        blockhash = str(value.get("blockhash") or "").strip()
        if not blockhash:
            raise RuntimeError(f"Missing blockhash in RPC response: {result}")
        last_valid_block_height_raw = to_int(value.get("lastValidBlockHeight"), -1)
        last_valid_block_height = (
            last_valid_block_height_raw if last_valid_block_height_raw >= 0 else None
        )
        return blockhash, last_valid_block_height

    async def _fetch_lookup_table_accounts(
        self,
        *,
        lookup_table_addresses: list[str],
    ) -> list[AddressLookupTableAccount]:
        lookup_table_accounts: list[AddressLookupTableAccount] = []
        for lookup_table_address in lookup_table_addresses:
            result = await self._rpc_call(
                "getAddressLookupTable",
                [lookup_table_address, {"commitment": "processed"}],
            )
            if not isinstance(result, dict):
                raise RuntimeError(
                    "Unexpected getAddressLookupTable response "
                    f"for {lookup_table_address}: {result}"
                )
            value = result.get("value")
            if not isinstance(value, dict):
                raise RuntimeError(f"Address lookup table not found: {lookup_table_address}")
            raw_addresses = value.get("addresses")
            if not isinstance(raw_addresses, list):
                raise RuntimeError(
                    "Address lookup table addresses are missing "
                    f"for {lookup_table_address}: {value}"
                )

            lookup_table_accounts.append(
                AddressLookupTableAccount(
                    Pubkey.from_string(lookup_table_address),
                    [
                        Pubkey.from_string(str(raw_address))
                        for raw_address in raw_addresses
                        if str(raw_address or "").strip()
                    ],
                )
            )

        return lookup_table_accounts

    @staticmethod
    def _single_tx_compact_quote_strategies() -> list[tuple[str, dict[str, str]]]:
        return [
            (
                "direct_route_24",
                {
                    "onlyDirectRoutes": "true",
                    "restrictIntermediateTokens": "true",
                    "maxAccounts": "24",
                },
            ),
            (
                "restricted_28",
                {
                    "restrictIntermediateTokens": "true",
                    "maxAccounts": "28",
                },
            ),
            ("max_accounts_24", {"maxAccounts": "24"}),
            ("max_accounts_20", {"maxAccounts": "20"}),
        ]

    @staticmethod
    def _safe_forward_output_amount(quote: dict[str, Any]) -> int:
        out_amount = to_int(quote.get("outAmount"), 0)
        min_out_amount = to_int(quote.get("otherAmountThreshold"), 0)
        if out_amount <= 0:
            return 0
        if min_out_amount <= 0:
            return out_amount
        return max(1, min(out_amount, min_out_amount))

    async def _build_single_tx_from_quotes(
        self,
        *,
        plan: AtomicExecutionPlan,
        forward_quote: dict[str, Any],
        reverse_quote: dict[str, Any],
        source: str,
        quote_strategy: dict[str, str] | None = None,
    ) -> AtomicBuildArtifact:
        forward_payload = await self._fetch_swap_instructions_payload(
            quote_response=forward_quote,
            priority_fee_micro_lamports=plan.priority_fee_micro_lamports,
        )
        reverse_payload = await self._fetch_swap_instructions_payload(
            quote_response=reverse_quote,
            priority_fee_micro_lamports=plan.priority_fee_micro_lamports,
        )

        forward_instructions, forward_lookup_addresses = self._extract_leg_instructions(
            payload=forward_payload,
            leg="forward",
        )
        reverse_instructions, reverse_lookup_addresses = self._extract_leg_instructions(
            payload=reverse_payload,
            leg="reverse",
        )

        execution_instructions = self._dedupe_instructions(
            [*forward_instructions, *reverse_instructions]
        )
        if not execution_instructions:
            raise RuntimeError("single_tx build failed: no executable swap instructions were generated.")

        lookup_addresses: list[str] = []
        for lookup_table_address in [*forward_lookup_addresses, *reverse_lookup_addresses]:
            if lookup_table_address not in lookup_addresses:
                lookup_addresses.append(lookup_table_address)

        lookup_table_accounts: list[AddressLookupTableAccount] = []
        if lookup_addresses:
            try:
                lookup_table_accounts = await self._fetch_lookup_table_accounts(
                    lookup_table_addresses=lookup_addresses,
                )
            except RpcMethodError as error:
                if (
                    error.method == "getAddressLookupTable"
                    and "method not found" in str(error).lower()
                ):
                    log_event(
                        self._logger,
                        level="warning",
                        event="atomic_lookup_table_rpc_unsupported",
                        message=(
                            "RPC does not support getAddressLookupTable; "
                            "retrying single_tx build without lookup tables"
                        ),
                        plan_id=plan.plan_id,
                        lookup_table_count=len(lookup_addresses),
                        rpc_status=error.status,
                        source=source,
                    )
                    lookup_table_accounts = []
                else:
                    raise
        latest_blockhash, last_valid_block_height = await self._fetch_latest_blockhash_with_height()

        compute_unit_limit = max(600_000, min(1_400_000, len(execution_instructions) * 50_000))
        prelude_instructions = [
            set_compute_unit_limit(compute_unit_limit),
            set_compute_unit_price(max(0, int(plan.priority_fee_micro_lamports))),
        ]

        message = MessageV0.try_compile(
            self._signer.pubkey(),
            [*prelude_instructions, *execution_instructions],
            lookup_table_accounts,
            Hash.from_string(latest_blockhash),
        )
        signer_signature = self._signer.sign_message(to_bytes_versioned(message))
        signed_tx = VersionedTransaction.populate(message, [signer_signature])
        if not signed_tx.signatures:
            raise RuntimeError("single_tx build failed: signed transaction has no signatures.")

        signed_tx_raw = bytes(signed_tx)
        if len(signed_tx_raw) > 1232:
            raise RuntimeError(
                "single_tx build produced an oversized transaction; "
                f"size={len(signed_tx_raw)} bytes"
            )

        tx_signature = str(signed_tx.signatures[0])
        signed_tx_base64 = base64.b64encode(signed_tx_raw).decode("ascii")
        log_event(
            self._logger,
            level="info",
            event="atomic_single_tx_built",
            message="Atomic single transaction was built successfully",
            plan_id=plan.plan_id,
            source=source,
            quote_strategy=quote_strategy or {},
            instruction_count=len(execution_instructions),
            lookup_table_count=len(lookup_table_accounts),
            lookup_table_requested_count=len(lookup_addresses),
            tx_size_bytes=len(signed_tx_raw),
            tx_signature=tx_signature,
        )
        return AtomicBuildArtifact(
            mode="single_tx",
            legs=[
                BuiltAtomicLeg(
                    leg="atomic",
                    signed_tx_base64=signed_tx_base64,
                    tx_signature=tx_signature,
                    latest_blockhash=latest_blockhash,
                    last_valid_block_height=last_valid_block_height,
                )
            ],
        )

    async def _build_single_tx_with_compact_requote(
        self,
        *,
        plan: AtomicExecutionPlan,
    ) -> AtomicBuildArtifact:
        last_error: Exception | None = None
        for strategy_name, strategy_params in self._single_tx_compact_quote_strategies():
            try:
                forward_quote = await self._watcher.quote(
                    input_mint=plan.forward_leg.input_mint,
                    output_mint=plan.forward_leg.output_mint,
                    amount=plan.forward_leg.amount_in,
                    slippage_bps=plan.forward_leg.slippage_bps,
                    extra_params=strategy_params,
                )
                reusable_forward_out = self._safe_forward_output_amount(forward_quote)
                if reusable_forward_out <= 0:
                    raise RuntimeError("compact requote forward outAmount is missing.")

                reverse_quote = await self._watcher.quote(
                    input_mint=plan.reverse_leg.input_mint,
                    output_mint=plan.reverse_leg.output_mint,
                    amount=reusable_forward_out,
                    slippage_bps=plan.reverse_leg.slippage_bps,
                    extra_params=strategy_params,
                )
                reverse_out = to_int(reverse_quote.get("outAmount"), 0)
                if reverse_out <= 0:
                    raise RuntimeError("compact requote reverse outAmount is missing.")

                spread_bps = ((reverse_out - plan.forward_leg.amount_in) / plan.forward_leg.amount_in) * 10_000
                if spread_bps < plan.required_spread_bps:
                    log_event(
                        self._logger,
                        level="info",
                        event="atomic_single_tx_compact_requote_unprofitable",
                        message="Compact requote did not satisfy atomic threshold",
                        plan_id=plan.plan_id,
                        strategy=strategy_name,
                        observed_spread_bps=round(spread_bps, 6),
                        required_spread_bps=round(plan.required_spread_bps, 6),
                    )
                    continue

                return await self._build_single_tx_from_quotes(
                    plan=plan,
                    forward_quote=forward_quote,
                    reverse_quote=reverse_quote,
                    source=f"compact_requote:{strategy_name}",
                    quote_strategy=strategy_params,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                last_error = error
                log_event(
                    self._logger,
                    level="warning",
                    event="atomic_single_tx_compact_requote_failed",
                    message="Compact requote strategy failed for single_tx build",
                    plan_id=plan.plan_id,
                    strategy=strategy_name,
                    reason=str(error),
                )

        if last_error is not None:
            raise RuntimeError(
                "single_tx build produced an oversized transaction even after compact requote attempts: "
                f"{last_error}"
            )
        raise RuntimeError("single_tx compact requote attempts were exhausted without a valid build.")

    async def _build_single_tx_artifact(self, plan: AtomicExecutionPlan) -> AtomicBuildArtifact:
        if self._signer is None:
            await self.connect()
        if self._signer is None:
            raise RuntimeError("Signer is not initialized.")

        try:
            try:
                return await self._build_single_tx_from_quotes(
                    plan=plan,
                    forward_quote=plan.forward_leg.quote_response,
                    reverse_quote=plan.reverse_leg.quote_response,
                    source="plan_quote",
                    quote_strategy=None,
                )
            except Exception as error:
                if "oversized transaction" not in str(error).lower():
                    raise
                log_event(
                    self._logger,
                    level="warning",
                    event="atomic_single_tx_oversized_requote_start",
                    message="single_tx is oversized; retrying with compact requote strategies",
                    plan_id=plan.plan_id,
                    reason=str(error),
                )
                return await self._build_single_tx_with_compact_requote(
                    plan=plan,
                )
        except asyncio.CancelledError:
            raise
        except (RpcMethodError, ValueError) as error:
            raise AtomicBuildUnavailableError(str(error)) from error
        except Exception as error:
            raise AtomicBuildUnavailableError(str(error)) from error

    async def _build_jito_tip_transaction(
        self,
        *,
        tip_lamports: int,
        plan_id: str,
    ) -> dict[str, Any]:
        if tip_lamports <= 0:
            raise RuntimeError("tip_lamports must be greater than zero in bundle mode.")

        if self._signer is None:
            await self.connect()
        if self._signer is None:
            raise RuntimeError("Signer is not initialized.")

        if self._http_session is None:
            await self.connect()
        if self._http_session is None:
            raise RuntimeError("HTTP session is not initialized.")

        tip_account_raw = await self._atomic_coordinator.jito_client.select_tip_account(
            session=self._http_session,
            plan_id=plan_id,
        )
        try:
            tip_account = Pubkey.from_string(tip_account_raw)
        except Exception as error:
            raise RuntimeError(f"Invalid Jito tip account returned: {tip_account_raw}") from error

        latest_blockhash = await self._fetch_latest_blockhash()
        instruction = transfer(
            TransferParams(
                from_pubkey=self._signer.pubkey(),
                to_pubkey=tip_account,
                lamports=max(0, int(tip_lamports)),
            )
        )
        message = MessageV0.try_compile(
            self._signer.pubkey(),
            [instruction],
            [],
            Hash.from_string(latest_blockhash),
        )
        signed_tip_tx = VersionedTransaction(message, [self._signer])
        if not signed_tip_tx.signatures:
            raise RuntimeError("Failed to build Jito tip transaction signature.")

        return {
            "signed_tx_base64": base64.b64encode(bytes(signed_tip_tx)).decode("ascii"),
            "tx_signature": str(signed_tip_tx.signatures[0]),
            "tip_account": str(tip_account),
            "latest_blockhash": latest_blockhash,
            "last_valid_block_height": None,
            "tip_lamports": max(0, int(tip_lamports)),
        }

    async def execute(
        self,
        *,
        intent: TradeIntent,
        idempotency_key: str,
        lock_ttl_seconds: int,
        priority_fee_micro_lamports: int,
        runtime_config: RuntimeConfig,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        runtime_config = self._effective_runtime_config(runtime_config)
        if runtime_config.execution_mode != "atomic":
            return await super().execute(
                intent=intent,
                idempotency_key=idempotency_key,
                lock_ttl_seconds=lock_ttl_seconds,
                priority_fee_micro_lamports=priority_fee_micro_lamports,
                runtime_config=runtime_config,
                metadata=metadata,
            )

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
        current_plan_id = "unknown"
        submitted_tx_signatures: list[str] = []

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

            pair = self._pair_from_metadata(intent, metadata)
            refreshed_observation = await self._atomic_planner.refresh_observation(
                quote_callable=self._watcher.quote,
                pair=pair,
            )
            plan = self._atomic_planner.build_plan(
                idempotency_key=idempotency_key,
                pair=pair,
                observation=refreshed_observation,
                runtime_config=runtime_config,
                priority_fee_micro_lamports=priority_fee_micro_lamports,
            )

            current_plan_id = plan.plan_id

            if plan.send_mode == "bundle":
                now = asyncio.get_running_loop().time()
                remaining_bundle_backoff = self._bundle_backoff_until - now
                if remaining_bundle_backoff > 0:
                    retry_after_seconds = round(max(0.0, remaining_bundle_backoff), 3)
                    skip_reason = "Bundle submission is cooling down after Jito rate limit"
                    await _record_order_state(
                        order_store=self._order_store,
                        order_id=order_id,
                        status="rate_limited",
                        ttl_seconds=record_ttl_seconds,
                        guard_key=idempotency_key,
                        payload={
                            "plan_id": plan.plan_id,
                            "error": skip_reason,
                            "retry_after_seconds": retry_after_seconds,
                            "bundle_backoff_seconds": retry_after_seconds,
                        },
                        logger=self._logger,
                        event="atomic_record_rate_limited_failed",
                    )
                    return ExecutionResult(
                        status="rate_limited",
                        tx_signature=None,
                        priority_fee_micro_lamports=priority_fee_micro_lamports,
                        reason=skip_reason,
                        order_id=order_id,
                        idempotency_key=idempotency_key,
                        metadata={
                            **compact_metadata,
                            "plan_id": plan.plan_id,
                            "rate_limited": True,
                            "retry_after_seconds": retry_after_seconds,
                            "bundle_backoff_seconds": retry_after_seconds,
                        },
                    )

            if plan.expected_spread_bps < plan.required_spread_bps:
                skip_reason = "Re-quote spread is below atomic threshold"
                await _record_order_state(
                    order_store=self._order_store,
                    order_id=order_id,
                    status="skipped_requote_unprofitable",
                    ttl_seconds=record_ttl_seconds,
                    guard_key=idempotency_key,
                    payload={
                        "plan_id": plan.plan_id,
                        "expected_spread_bps": plan.expected_spread_bps,
                        "required_spread_bps": plan.required_spread_bps,
                    },
                    logger=self._logger,
                    event="atomic_record_skip_failed",
                )
                return ExecutionResult(
                    status="skipped_requote_unprofitable",
                    tx_signature=None,
                    priority_fee_micro_lamports=priority_fee_micro_lamports,
                    reason=skip_reason,
                    order_id=order_id,
                    idempotency_key=idempotency_key,
                    metadata={
                        **compact_metadata,
                        "plan_id": plan.plan_id,
                        "expected_spread_bps": plan.expected_spread_bps,
                        "required_spread_bps": plan.required_spread_bps,
                        "expected_net_bps": plan.expected_net_bps,
                    },
                )

            if plan.is_expired():
                skip_reason = "Atomic plan expired before submission"
                await _record_order_state(
                    order_store=self._order_store,
                    order_id=order_id,
                    status="skipped_expired_plan",
                    ttl_seconds=record_ttl_seconds,
                    guard_key=idempotency_key,
                    payload={"plan_id": plan.plan_id, "expires_at_ms": plan.expires_at_ms},
                    logger=self._logger,
                    event="atomic_record_skip_expired_failed",
                )
                return ExecutionResult(
                    status="skipped_expired_plan",
                    tx_signature=None,
                    priority_fee_micro_lamports=priority_fee_micro_lamports,
                    reason=skip_reason,
                    order_id=order_id,
                    idempotency_key=idempotency_key,
                    metadata={
                        **compact_metadata,
                        "plan_id": plan.plan_id,
                        "expires_at_ms": plan.expires_at_ms,
                    },
                )

            artifact = await self._atomic_builder.build(
                plan=plan,
                build_leg_tx=self._build_leg_transaction,
                build_single_tx=self._build_single_tx_artifact,
            )

            if plan.is_expired():
                skip_reason = "Atomic plan expired before network submission"
                await _record_order_state(
                    order_store=self._order_store,
                    order_id=order_id,
                    status="skipped_expired_plan",
                    ttl_seconds=record_ttl_seconds,
                    guard_key=idempotency_key,
                    payload={
                        "plan_id": plan.plan_id,
                        "expires_at_ms": plan.expires_at_ms,
                        "stage": "post_build",
                    },
                    logger=self._logger,
                    event="atomic_record_skip_expired_failed",
                )
                return ExecutionResult(
                    status="skipped_expired_plan",
                    tx_signature=None,
                    priority_fee_micro_lamports=priority_fee_micro_lamports,
                    reason=skip_reason,
                    order_id=order_id,
                    idempotency_key=idempotency_key,
                    metadata={
                        **compact_metadata,
                        "plan_id": plan.plan_id,
                        "expires_at_ms": plan.expires_at_ms,
                        "stage": "post_build",
                    },
                )

            if (
                artifact.mode == "bundle"
                and plan.expected_net_bps < runtime_config.atomic_bundle_min_expected_net_bps
            ):
                skip_reason = "Expected net spread is too thin for bundle submission"
                log_event(
                    self._logger,
                    level="info",
                    event="atomic_bundle_thin_opportunity_skipped",
                    message="Atomic bundle submission skipped due to thin expected net spread",
                    plan_id=plan.plan_id,
                    expected_net_bps=round(plan.expected_net_bps, 6),
                    required_min_net_bps=round(runtime_config.atomic_bundle_min_expected_net_bps, 6),
                    send_mode_requested=plan.send_mode,
                )
                await _record_order_state(
                    order_store=self._order_store,
                    order_id=order_id,
                    status="skipped_bundle_thin_opportunity",
                    ttl_seconds=record_ttl_seconds,
                    guard_key=idempotency_key,
                    payload={
                        "plan_id": plan.plan_id,
                        "expected_net_bps": plan.expected_net_bps,
                        "required_min_net_bps": runtime_config.atomic_bundle_min_expected_net_bps,
                    },
                    logger=self._logger,
                    event="atomic_record_skip_bundle_thin_failed",
                )
                return ExecutionResult(
                    status="skipped_bundle_thin_opportunity",
                    tx_signature=None,
                    priority_fee_micro_lamports=priority_fee_micro_lamports,
                    reason=skip_reason,
                    order_id=order_id,
                    idempotency_key=idempotency_key,
                    metadata={
                        **compact_metadata,
                        "plan_id": plan.plan_id,
                        "expected_net_bps": plan.expected_net_bps,
                        "required_min_net_bps": runtime_config.atomic_bundle_min_expected_net_bps,
                        "resolved_mode": artifact.mode,
                    },
                )

            self._atomic_coordinator.jito_client.update_endpoint(runtime_config.jito_block_engine_url)
            tx_signatures = artifact.tx_signatures()
            last_valid_block_height_by_signature = {
                leg.tx_signature: leg.last_valid_block_height for leg in artifact.legs
            }
            signed_transactions: list[str] = []
            tip_tx_signature: str | None = None
            tip_account: str | None = None

            if artifact.mode == "bundle":
                if self._http_session is None:
                    await self.connect()
                if self._http_session is None:
                    raise RuntimeError("HTTP session is not initialized for bundle submission.")

                signed_transactions = [leg.signed_tx_base64 for leg in artifact.legs]

                tip_tx = await self._build_jito_tip_transaction(
                    tip_lamports=plan.tip_lamports,
                    plan_id=plan.plan_id,
                )
                tip_tx_signature = str(tip_tx.get("tx_signature") or "").strip()
                if not tip_tx_signature:
                    raise RuntimeError("Failed to build Jito tip transaction signature.")
                tip_account = str(tip_tx.get("tip_account") or "").strip() or None

                signed_transactions.append(str(tip_tx["signed_tx_base64"]))
                tx_signatures.append(tip_tx_signature)
                last_valid_block_height_by_signature[tip_tx_signature] = tip_tx.get("last_valid_block_height")

            submitted_tx_signatures = list(tx_signatures)

            await self._atomic_coordinator.pending_manager.mark_submitted(
                plan=plan,
                order_id=order_id,
                guard_key=idempotency_key,
                tx_signatures=tx_signatures,
                ttl_seconds=record_ttl_seconds,
                extra_payload={
                    "builder_mode": artifact.mode,
                    "tip_tx_signature": tip_tx_signature,
                    "tip_account": tip_account,
                },
            )

            await _record_order_state(
                order_store=self._order_store,
                order_id=order_id,
                status="submitted",
                ttl_seconds=record_ttl_seconds,
                guard_key=idempotency_key,
                payload={
                    "plan_id": plan.plan_id,
                    "mode": artifact.mode,
                    "tx_signatures": tx_signatures,
                    "expected_net_bps": plan.expected_net_bps,
                    "expected_fee_bps": plan.expected_fee_bps,
                    "tip_lamports": plan.tip_lamports,
                    "tip_tx_signature": tip_tx_signature,
                    "tip_account": tip_account,
                },
                logger=self._logger,
                event="atomic_record_submitted_failed",
            )

            log_event(
                self._logger,
                level="info",
                event="atomic_plan_submitted",
                message="Atomic plan submitted",
                plan_id=plan.plan_id,
                mode=artifact.mode,
                send_mode_requested=plan.send_mode,
                expected_net_bps=round(plan.expected_net_bps, 6),
                expected_fee_bps=round(plan.expected_fee_bps, 6),
                tip_lamports=plan.tip_lamports,
                tip_tx_signature=tip_tx_signature,
                tip_account=tip_account,
                tx_signatures=tx_signatures,
            )

            bundle_id: str | None = None
            confirmations: list[dict[str, Any]] = []

            if artifact.mode == "single_tx":
                if not artifact.legs:
                    raise RuntimeError("Atomic single_tx build returned no transactions.")
                leg = artifact.legs[0]
                submitted_sig = await self._send_transaction_with_retry(
                    signed_tx_base64=leg.signed_tx_base64,
                )
                if submitted_sig and submitted_sig != tx_signatures[0]:
                    expected_signature = tx_signatures[0]
                    expected_lvh = last_valid_block_height_by_signature.pop(
                        expected_signature,
                        leg.last_valid_block_height,
                    )
                    tx_signatures = [submitted_sig]
                    last_valid_block_height_by_signature[submitted_sig] = expected_lvh
                    submitted_tx_signatures = list(tx_signatures)
            else:
                bundle_id = await self._submit_bundle_with_retry(
                    plan=plan,
                    signed_transactions=signed_transactions,
                )
                self._reset_bundle_rate_limit_backoff()

            confirmations = await self._wait_for_atomic_confirmations(
                tx_signatures=tx_signatures,
                last_valid_block_height_by_signature=last_valid_block_height_by_signature,
            )

            await self._atomic_coordinator.pending_manager.mark_confirmed(
                plan_id=plan.plan_id,
                order_id=order_id,
                guard_key=idempotency_key,
                tx_signatures=tx_signatures,
                ttl_seconds=record_ttl_seconds,
                payload={
                    "bundle_id": bundle_id,
                    "confirmations": confirmations,
                    "expected_net_bps": plan.expected_net_bps,
                    "expected_fee_bps": plan.expected_fee_bps,
                    "tip_lamports": plan.tip_lamports,
                    "tip_tx_signature": tip_tx_signature,
                    "tip_account": tip_account,
                },
            )

            result_metadata = {
                **compact_metadata,
                "plan_id": plan.plan_id,
                "mode": artifact.mode,
                "send_mode_requested": plan.send_mode,
                "tx_signatures": tx_signatures,
                "bundle_id": bundle_id,
                "confirmations": confirmations,
                "expected_net_bps": plan.expected_net_bps,
                "expected_net_lamports": plan.expected_net_lamports,
                "expected_fee_bps": plan.expected_fee_bps,
                "tip_lamports": plan.tip_lamports,
                "tip_tx_signature": tip_tx_signature,
                "tip_account": tip_account,
                "required_spread_bps": plan.required_spread_bps,
                "observed_spread_bps": plan.expected_spread_bps,
                "expires_at_ms": plan.expires_at_ms,
                "confirmed": True,
            }

            log_event(
                self._logger,
                level="info",
                event="atomic_plan_confirmed",
                message="Atomic plan confirmed",
                plan_id=plan.plan_id,
                mode=artifact.mode,
                bundle_id=bundle_id,
                tx_signatures=tx_signatures,
                expected_net_bps=round(plan.expected_net_bps, 6),
                expected_fee_bps=round(plan.expected_fee_bps, 6),
                tip_lamports=plan.tip_lamports,
                tip_tx_signature=tip_tx_signature,
                tip_account=tip_account,
                confirmed=True,
            )

            return ExecutionResult(
                status="filled",
                tx_signature=tx_signatures[0] if tx_signatures else None,
                priority_fee_micro_lamports=priority_fee_micro_lamports,
                reason="atomic plan confirmed",
                order_id=order_id,
                idempotency_key=idempotency_key,
                metadata=result_metadata,
            )
        except JitoBundleRateLimitError as error:
            backoff_seconds = self._apply_bundle_rate_limit_backoff(
                retry_after_seconds=error.retry_after_seconds,
            )
            if current_plan_id != "unknown":
                await self._atomic_coordinator.pending_manager.mark_failed(
                    plan_id=current_plan_id,
                    order_id=order_id,
                    guard_key=idempotency_key,
                    tx_signatures=submitted_tx_signatures,
                    ttl_seconds=record_ttl_seconds,
                    payload={
                        "error": str(error),
                        "rate_limited": True,
                        "retry_after_seconds": error.retry_after_seconds,
                        "bundle_backoff_seconds": backoff_seconds,
                    },
                )
            await _record_order_state(
                order_store=self._order_store,
                order_id=order_id,
                status="rate_limited",
                ttl_seconds=record_ttl_seconds,
                guard_key=idempotency_key,
                payload={
                    "plan_id": current_plan_id,
                    "retry_after_seconds": error.retry_after_seconds,
                    "error": str(error),
                    "bundle_backoff_seconds": backoff_seconds,
                },
                logger=self._logger,
                event="atomic_record_rate_limited_failed",
            )
            return ExecutionResult(
                status="rate_limited",
                tx_signature=None,
                priority_fee_micro_lamports=priority_fee_micro_lamports,
                reason=str(error),
                order_id=order_id,
                idempotency_key=idempotency_key,
                metadata={
                    **compact_metadata,
                    "plan_id": current_plan_id,
                    "rate_limited": True,
                    "retry_after_seconds": max(
                        error.retry_after_seconds or 0.0,
                        backoff_seconds,
                    ),
                    "bundle_backoff_seconds": backoff_seconds,
                    "tx_signatures": submitted_tx_signatures,
                },
            )
        except TransactionPendingConfirmationError as error:
            release_guard = False
            if current_plan_id != "unknown":
                await self._atomic_coordinator.pending_manager.mark_pending_confirmation(
                    plan_id=current_plan_id,
                    order_id=order_id,
                    guard_key=idempotency_key,
                    tx_signatures=submitted_tx_signatures,
                    ttl_seconds=record_ttl_seconds,
                    payload={"error": str(error)},
                )
            await guarded_call(
                lambda: self._order_store.refresh_order_guard(
                    guard_key=idempotency_key,
                    order_id=order_id,
                    ttl_seconds=max(lock_ttl_seconds, self._pending_guard_ttl_seconds),
                ),
                logger=self._logger,
                event="atomic_pending_guard_extend_failed",
                message="Failed to extend order guard for pending atomic confirmation",
                level="warning",
            )
            raise
        except AtomicBuildUnavailableError as error:
            if current_plan_id != "unknown":
                await self._atomic_coordinator.pending_manager.mark_failed(
                    plan_id=current_plan_id,
                    order_id=order_id,
                    guard_key=idempotency_key,
                    tx_signatures=submitted_tx_signatures,
                    ttl_seconds=record_ttl_seconds,
                    payload={"error": str(error)},
                )
            await _record_order_state(
                order_store=self._order_store,
                order_id=order_id,
                status="failed",
                ttl_seconds=record_ttl_seconds,
                guard_key=idempotency_key,
                payload={"error": str(error)},
                logger=self._logger,
                event="atomic_record_single_tx_unavailable_failed",
            )
            raise RuntimeError(str(error)) from error
        except Exception as error:
            if current_plan_id != "unknown":
                await self._atomic_coordinator.pending_manager.mark_failed(
                    plan_id=current_plan_id,
                    order_id=order_id,
                    guard_key=idempotency_key,
                    tx_signatures=submitted_tx_signatures,
                    ttl_seconds=record_ttl_seconds,
                    payload={"error": str(error)},
                )
            await _record_order_state(
                order_store=self._order_store,
                order_id=order_id,
                status="failed",
                ttl_seconds=record_ttl_seconds,
                guard_key=idempotency_key,
                payload={"error": str(error)},
                logger=self._logger,
                event="atomic_record_failed_state_failed",
            )
            raise
        finally:
            await _cancel_task(
                guard_refresh_task,
                logger=self._logger,
                event="atomic_guard_refresh_cancel_failed",
            )
            if release_guard:
                await guarded_call(
                    lambda: self._order_store.release_order_guard(
                        guard_key=idempotency_key,
                        order_id=order_id,
                    ),
                    logger=self._logger,
                    event="atomic_release_guard_failed",
                    message="Failed to release order guard",
                    level="warning",
                )
