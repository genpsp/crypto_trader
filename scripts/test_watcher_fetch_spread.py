from __future__ import annotations

import logging
import unittest
from unittest.mock import AsyncMock

from modules.trading.types import PairConfig
from modules.trading.watcher import HeliusQuoteWatcher


def _make_pair() -> PairConfig:
    return PairConfig(
        symbol="SOL/USDC",
        base_mint="So11111111111111111111111111111111111111112",
        quote_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        base_decimals=9,
        quote_decimals=6,
        base_amount=10_000_000,
        slippage_bps=10,
    )


def _make_combo(*, spread_bps: float) -> dict[str, object]:
    base_amount = _make_pair().base_amount
    reverse_out = int(base_amount * (1 + (spread_bps / 10_000)))
    forward_quote = {"outAmount": "1000000", "routePlan": [{"swapInfo": {"label": "Whirlpool"}}]}
    reverse_quote = {"outAmount": str(reverse_out), "routePlan": [{"swapInfo": {"label": "Whirlpool"}}]}
    return {
        "spread_bps": spread_bps,
        "forward": {
            "strategy": "test",
            "params": {},
            "quote": forward_quote,
            "out_amount": int(forward_quote["outAmount"]),
            "route_fingerprint": "Whirlpool",
            "route_dexes": ("Whirlpool",),
        },
        "reverse": {
            "strategy": "test",
            "params": {},
            "quote": reverse_quote,
            "out_amount": int(reverse_quote["outAmount"]),
            "route_fingerprint": "Whirlpool",
            "route_dexes": ("Whirlpool",),
        },
    }


class WatcherFetchSpreadTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.watcher = HeliusQuoteWatcher(
            logger=logging.getLogger("test.watcher"),
            api_base_url="https://api.jup.ag/swap/v1/quote",
            send_api_key_header=False,
        )
        self.watcher._refresh_dynamic_dex_labels = AsyncMock(return_value=())  # type: ignore[method-assign]
        self.watcher._quote_rps_effective = lambda: 0.0  # type: ignore[method-assign]
        self.watcher._quote_cache_hit_rate = lambda: 0.0  # type: ignore[method-assign]

    async def test_fetch_spread_median_off_path_keeps_defaults_safe(self) -> None:
        pair = _make_pair()
        combo = _make_combo(spread_bps=1.0)
        self.watcher._find_best_combo = AsyncMock(  # type: ignore[method-assign]
            return_value=(combo, 1, 1, [combo["forward"]], [combo["reverse"]], 0)
        )
        self.watcher._requote_best_combo_median = AsyncMock(return_value=(None, []))  # type: ignore[method-assign]

        observation = await self.watcher.fetch_spread(
            pair,
            enable_stagea_relaxed_gate=False,
            stage_a_required_bps=1.0,
            stage_a_min_margin_bps=0.5,
            stage_b_required_bps_without_tip=1.0,
        )

        self.assertFalse(observation.stage_a_pass)
        self.assertEqual(observation.fail_reason, "BELOW_STAGEA_REQUIRED")
        self.assertFalse(observation.median_requote_applied)

    async def test_fetch_spread_no_combo_raises_runtime_error_without_unboundlocal(self) -> None:
        pair = _make_pair()
        self.watcher._find_best_combo = AsyncMock(return_value=(None, 0, 0, [], [], 0))  # type: ignore[method-assign]

        with self.assertRaises(RuntimeError) as ctx:
            await self.watcher.fetch_spread(pair)
        self.assertIn("did not produce any forward/reverse combination", str(ctx.exception))

    async def test_fetch_spread_combo_exception_path_is_safe(self) -> None:
        pair = _make_pair()
        self.watcher._find_best_combo = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

        with self.assertRaises(RuntimeError) as ctx:
            await self.watcher.fetch_spread(pair)
        self.assertIn("boom", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
