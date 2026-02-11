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
    jupiter_quote_api: str
    rpc_url: str
    private_key: str
    dry_run: bool

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            watch_interval_seconds=max(0.05, float(os.getenv("WATCH_INTERVAL_SECONDS", "1.0"))),
            error_backoff_seconds=max(0.2, float(os.getenv("ERROR_BACKOFF_SECONDS", "2.0"))),
            jupiter_quote_api=os.getenv("JUPITER_QUOTE_API", "https://quote-api.jup.ag/v6"),
            rpc_url=os.getenv("RPC_URL", ""),
            private_key=os.getenv("PRIVATE_KEY", ""),
            dry_run=to_bool(os.getenv("DRY_RUN"), True),
        )
