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


def to_int(value: Any, default: int) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def normalize_execution_mode(value: str) -> str:
    mode = (value or "").strip().lower()
    if mode in {"atomic", "legacy"}:
        return mode
    return "atomic"


def normalize_atomic_send_mode(value: str) -> str:
    mode = (value or "").strip().lower()
    if mode in {"single_tx", "bundle", "auto"}:
        return mode
    return "auto"


@dataclass(slots=True)
class AppSettings:
    watch_interval_seconds: float
    error_backoff_seconds: float
    helius_quote_api: str
    helius_api_key: str
    jupiter_api_key: str
    helius_jup_proxy_enabled: bool
    jupiter_swap_api: str
    solana_rpc_url: str
    private_key: str
    dry_run: bool
    execution_mode: str
    atomic_send_mode: str
    atomic_expiry_ms: int
    atomic_margin_bps: float
    jito_block_engine_url: str
    jito_tip_lamports_max: int
    jito_tip_lamports_recommended: int
    live_send_max_attempts: int
    live_send_retry_backoff_seconds: float
    live_confirm_timeout_seconds: float
    live_confirm_poll_interval_seconds: float
    live_rebuild_max_attempts: int
    live_pending_guard_ttl_seconds: int
    live_pending_recovery_limit: int
    live_execution_cooldown_seconds: float
    live_execution_window_seconds: float
    live_max_executions_per_window: int
    live_max_pending_orders: int
    live_min_balance_lamports: int
    live_fee_budget_window_seconds: float
    live_max_estimated_fee_lamports_per_window: int
    live_estimated_base_fee_lamports: int
    live_max_consecutive_execution_errors: int
    live_execution_circuit_breaker_seconds: float
    live_drawdown_window_seconds: float
    live_max_drawdown_lamports: float
    live_max_drawdown_pct: float
    live_final_stop_equity_usd: float
    live_drawdown_circuit_breaker_seconds: float

    @classmethod
    def from_env(cls) -> "AppSettings":
        return cls(
            watch_interval_seconds=max(0.05, to_float(os.getenv("WATCH_INTERVAL_SECONDS"), 1.0)),
            error_backoff_seconds=max(0.2, to_float(os.getenv("ERROR_BACKOFF_SECONDS"), 2.0)),
            helius_quote_api=os.getenv(
                "HELIUS_QUOTE_API",
                "https://api.jup.ag/swap/v1/quote",
            ),
            helius_api_key=os.getenv("HELIUS_API_KEY", "").strip(),
            jupiter_api_key=os.getenv("JUPITER_API_KEY", "").strip(),
            helius_jup_proxy_enabled=to_bool(os.getenv("HELIUS_JUP_PROXY_ENABLED"), False),
            jupiter_swap_api=os.getenv("JUPITER_SWAP_API", "https://api.jup.ag/swap/v1/swap").strip(),
            solana_rpc_url=os.getenv("SOLANA_RPC_URL", "").strip(),
            private_key=os.getenv("PRIVATE_KEY", ""),
            dry_run=to_bool(os.getenv("DRY_RUN"), True),
            execution_mode=normalize_execution_mode(os.getenv("EXECUTION_MODE", "atomic")),
            atomic_send_mode=normalize_atomic_send_mode(os.getenv("ATOMIC_SEND_MODE", "auto")),
            atomic_expiry_ms=max(250, to_int(os.getenv("ATOMIC_EXPIRY_MS"), 5_000)),
            atomic_margin_bps=max(0.0, to_float(os.getenv("ATOMIC_MARGIN_BPS"), 20.0)),
            jito_block_engine_url=os.getenv("JITO_BLOCK_ENGINE_URL", "").strip(),
            jito_tip_lamports_max=max(0, to_int(os.getenv("JITO_TIP_LAMPORTS_MAX"), 100_000)),
            jito_tip_lamports_recommended=max(
                0,
                to_int(os.getenv("JITO_TIP_LAMPORTS_RECOMMENDED"), 20_000),
            ),
            live_send_max_attempts=max(1, to_int(os.getenv("LIVE_SEND_MAX_ATTEMPTS"), 3)),
            live_send_retry_backoff_seconds=max(
                0.1,
                to_float(os.getenv("LIVE_SEND_RETRY_BACKOFF_SECONDS"), 0.8),
            ),
            live_confirm_timeout_seconds=max(
                5.0,
                to_float(os.getenv("LIVE_CONFIRM_TIMEOUT_SECONDS"), 45.0),
            ),
            live_confirm_poll_interval_seconds=max(
                0.25,
                to_float(os.getenv("LIVE_CONFIRM_POLL_INTERVAL_SECONDS"), 1.0),
            ),
            live_rebuild_max_attempts=max(1, to_int(os.getenv("LIVE_REBUILD_MAX_ATTEMPTS"), 2)),
            live_pending_guard_ttl_seconds=max(
                30,
                to_int(os.getenv("LIVE_PENDING_GUARD_TTL_SECONDS"), 180),
            ),
            live_pending_recovery_limit=max(1, to_int(os.getenv("LIVE_PENDING_RECOVERY_LIMIT"), 50)),
            live_execution_cooldown_seconds=max(
                0.0,
                to_float(os.getenv("LIVE_EXECUTION_COOLDOWN_SECONDS"), 5.0),
            ),
            live_execution_window_seconds=max(
                1.0,
                to_float(os.getenv("LIVE_EXECUTION_WINDOW_SECONDS"), 60.0),
            ),
            live_max_executions_per_window=max(
                1,
                to_int(os.getenv("LIVE_MAX_EXECUTIONS_PER_WINDOW"), 5),
            ),
            live_max_pending_orders=max(1, to_int(os.getenv("LIVE_MAX_PENDING_ORDERS"), 1)),
            live_min_balance_lamports=max(0, to_int(os.getenv("LIVE_MIN_BALANCE_LAMPORTS"), 20_000_000)),
            live_fee_budget_window_seconds=max(
                1.0,
                to_float(os.getenv("LIVE_FEE_BUDGET_WINDOW_SECONDS"), 3600.0),
            ),
            live_max_estimated_fee_lamports_per_window=max(
                0,
                to_int(os.getenv("LIVE_MAX_ESTIMATED_FEE_LAMPORTS_PER_WINDOW"), 500_000),
            ),
            live_estimated_base_fee_lamports=max(
                0,
                to_int(os.getenv("LIVE_ESTIMATED_BASE_FEE_LAMPORTS"), 5000),
            ),
            live_max_consecutive_execution_errors=max(
                1,
                to_int(os.getenv("LIVE_MAX_CONSECUTIVE_EXECUTION_ERRORS"), 3),
            ),
            live_execution_circuit_breaker_seconds=max(
                1.0,
                to_float(os.getenv("LIVE_EXECUTION_CIRCUIT_BREAKER_SECONDS"), 120.0),
            ),
            live_drawdown_window_seconds=max(
                1.0,
                to_float(os.getenv("LIVE_DRAWDOWN_WINDOW_SECONDS"), 86400.0),
            ),
            live_max_drawdown_lamports=max(
                0.0,
                to_float(os.getenv("LIVE_MAX_DRAWDOWN_LAMPORTS"), 0.0),
            ),
            live_max_drawdown_pct=max(
                0.0,
                to_float(os.getenv("LIVE_MAX_DRAWDOWN_PCT"), 0.0),
            ),
            live_final_stop_equity_usd=max(
                0.0,
                to_float(os.getenv("LIVE_FINAL_STOP_EQUITY_USD"), 0.0),
            ),
            live_drawdown_circuit_breaker_seconds=max(
                1.0,
                to_float(os.getenv("LIVE_DRAWDOWN_CIRCUIT_BREAKER_SECONDS"), 900.0),
            ),
        )
