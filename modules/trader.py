from __future__ import annotations

import contextlib
import importlib
import json
import logging
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
    trade_enabled: bool

    @classmethod
    def from_env_defaults(cls) -> "RuntimeConfig":
        return cls(
            min_spread_bps=_to_float(os.getenv("MIN_SPREAD_BPS"), 5.0),
            dex_fee_bps=_to_float(os.getenv("DEX_FEE_BPS"), 4.0),
            priority_fee_micro_lamports=_to_int(os.getenv("PRIORITY_FEE_MICRO_LAMPORTS"), 10_000),
            priority_compute_units=_to_int(os.getenv("PRIORITY_COMPUTE_UNITS"), 200_000),
            trade_enabled=_to_bool(os.getenv("TRADE_ENABLED"), False),
        )

    @classmethod
    def from_redis(cls, redis_config: dict[str, str], defaults: "RuntimeConfig") -> "RuntimeConfig":
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
            trade_enabled=_to_bool(redis_config.get("trade_enabled"), defaults.trade_enabled),
        )


@dataclass(slots=True, frozen=True)
class TradeDecision:
    profitable: bool
    should_execute: bool
    spread_bps: float
    required_spread_bps: float
    total_fee_bps: float
    reason: str


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
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OrderExecutor(Protocol):
    async def connect(self) -> None:
        ...

    async def close(self) -> None:
        ...

    async def execute(
        self,
        *,
        intent: TradeIntent,
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
            timestamp=datetime.now(timezone.utc).isoformat(),
            forward_out_amount=forward_out,
            reverse_out_amount=reverse_out,
            forward_price=forward_price,
            spread_bps=spread_bps,
            forward_quote=forward_quote,
            reverse_quote=reverse_quote,
        )


class DryRunOrderExecutor:
    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def execute(
        self,
        *,
        intent: TradeIntent,
        priority_fee_micro_lamports: int,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        self._logger.info(
            "Dry-run execution",
            extra={
                "event": "order_dry_run",
                "pair": intent.pair,
                "priority_fee_micro_lamports": priority_fee_micro_lamports,
            },
        )
        return ExecutionResult(
            status="dry_run",
            tx_signature=None,
            priority_fee_micro_lamports=priority_fee_micro_lamports,
            reason="DRY_RUN is enabled",
            metadata=metadata or {},
        )


class LiveOrderExecutor:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        rpc_url: str,
        private_key: str,
    ) -> None:
        self._logger = logger
        self._rpc_url = rpc_url
        self._private_key = private_key
        self._rpc_client: AsyncClient | None = None
        self._signer: Keypair | None = None

    async def connect(self) -> None:
        if not self._rpc_url:
            raise ValueError("RPC_URL is required when DRY_RUN is false.")
        if not self._private_key:
            raise ValueError("PRIVATE_KEY is required when DRY_RUN is false.")

        self._rpc_client = AsyncClient(self._rpc_url)
        self._signer = self._parse_private_key(self._private_key)

    async def close(self) -> None:
        if self._rpc_client is not None:
            await self._rpc_client.close()
            self._rpc_client = None

    async def execute(
        self,
        *,
        intent: TradeIntent,
        priority_fee_micro_lamports: int,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        _ = set_compute_unit_price(priority_fee_micro_lamports)
        jupiter_sdk_available = False
        with contextlib.suppress(Exception):
            importlib.import_module("jupiter_python_sdk")
            jupiter_sdk_available = True

        self._logger.warning(
            "Live execution path is a placeholder",
            extra={"event": "order_live_placeholder", "pair": intent.pair},
        )

        result_metadata = metadata.copy() if metadata else {}
        result_metadata["jupiter_sdk_available"] = jupiter_sdk_available

        return ExecutionResult(
            status="live_placeholder",
            tx_signature=None,
            priority_fee_micro_lamports=priority_fee_micro_lamports,
            reason="Swap transaction builder is not implemented yet",
            metadata=result_metadata,
        )

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
    ) -> TradeDecision:
        priority_fee_bps = self._priority_fee_bps(
            priority_fee_micro_lamports=runtime_config.priority_fee_micro_lamports,
            compute_units=runtime_config.priority_compute_units,
            notional_lamports=pair.base_amount,
        )
        total_fee_bps = runtime_config.dex_fee_bps + priority_fee_bps
        required_spread_bps = runtime_config.min_spread_bps + total_fee_bps

        profitable = observation.spread_bps >= required_spread_bps
        should_execute = profitable and runtime_config.trade_enabled

        if should_execute:
            reason = "spread threshold exceeded"
        elif profitable:
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
        )

    async def execute(
        self,
        *,
        pair: PairConfig,
        observation: SpreadObservation,
        runtime_config: RuntimeConfig,
    ) -> ExecutionResult:
        intent = TradeIntent(
            pair=pair.symbol,
            input_mint=pair.base_mint,
            output_mint=pair.quote_mint,
            amount_in=pair.base_amount,
            expected_amount_out=observation.forward_out_amount,
        )

        return await self.executor.execute(
            intent=intent,
            priority_fee_micro_lamports=runtime_config.priority_fee_micro_lamports,
            metadata={
                "spread_bps": observation.spread_bps,
                "forward_out_amount": observation.forward_out_amount,
                "reverse_out_amount": observation.reverse_out_amount,
            },
        )
