from __future__ import annotations

import logging
from typing import Any

import aiohttp

from .types import PairConfig, SpreadObservation, now_iso


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
            timestamp=now_iso(),
            forward_out_amount=forward_out,
            reverse_out_amount=reverse_out,
            forward_price=forward_price,
            spread_bps=spread_bps,
            forward_quote=forward_quote,
            reverse_quote=reverse_quote,
        )
