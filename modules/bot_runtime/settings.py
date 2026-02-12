from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class AppSettings:
    watch_interval_seconds: float
    error_backoff_seconds: float
    helius_quote_api: str
    helius_api_key: str
    jupiter_api_key: str
    solana_rpc_url: str
    private_key: str
    dry_run: bool

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            watch_interval_seconds=max(0.05, float(os.getenv("WATCH_INTERVAL_SECONDS", "1.0"))),
            error_backoff_seconds=max(0.2, float(os.getenv("ERROR_BACKOFF_SECONDS", "2.0"))),
            helius_quote_api=os.getenv(
                "HELIUS_QUOTE_API",
                "https://mainnet.helius-rpc.com/v0/jup-proxy/swap/v1/quote?api-key=YOUR_API_KEY",
            ),
            helius_api_key=os.getenv("HELIUS_API_KEY", "").strip(),
            jupiter_api_key=os.getenv("JUPITER_API_KEY", "").strip(),
            solana_rpc_url=os.getenv("SOLANA_RPC_URL") or os.getenv("RPC_URL", ""),
            private_key=os.getenv("PRIVATE_KEY", ""),
            dry_run=to_bool(os.getenv("DRY_RUN"), True),
        )
