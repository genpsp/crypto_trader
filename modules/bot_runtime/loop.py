from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from collections import deque
from dataclasses import asdict, replace
from typing import Any

from modules.common import guarded_call, log_event
from modules.storage import StorageGateway
from modules.trading import (
    FAIL_REASON_ACCOUNT_LIMIT,
    FAIL_REASON_BELOW_STAGEA_REQUIRED,
    FAIL_REASON_BELOW_STAGEB_REQUIRED,
    FAIL_REASON_OTHER,
    FAIL_REASON_PROBE_LIMIT_LOSS_LAMPORTS,
    FAIL_REASON_PROBE_LIMIT_MAX_BASE_AMOUNT,
    FAIL_REASON_PROBE_LIMIT_NEG_NET,
    FAIL_REASON_RATE_LIMITED,
    FAIL_REASON_NOT_LANDED,
    FAIL_REASON_ROUTE_UNSTABLE_REQUOTE_DROPPED,
    FAIL_REASON_SIMULATION_FAIL,
    FAIL_REASON_SLIPPAGE_EXCEEDED,
    FAIL_REASON_TX_FAIL,
    HeliusRateLimitError,
    PairConfig,
    RuntimeConfig,
    TransactionPendingConfirmationError,
    TraderEngine,
)

from .loop_helpers import (
    bootstrap_dependencies,
    check_and_recover_pending_guard,
    compute_rate_limit_backoff_seconds,
    drawdown_pct_from_peak,
    estimate_execution_fee_lamports,
    prepare_execution_result_metadata,
    to_float,
    try_resume_order_intake,
    wait_with_stop,
)
from .settings import AppSettings


async def run_trading_loop(
    *,
    logger: logging.Logger,
    stop_event: asyncio.Event,
    app_settings: AppSettings,
    storage: StorageGateway,
    pair: PairConfig,
    runtime_defaults: RuntimeConfig,
    trader_engine: TraderEngine,
) -> None:
    PROBE_MAX_NEG_NET_BPS = -0.5
    PROBE_MAX_LOSS_LAMPORTS = 5_000
    PROBE_MAX_BASE_AMOUNT_LAMPORTS = 200_000_000

    loop = asyncio.get_running_loop()
    next_tick = loop.time()

    order_intake_paused = False
    pause_reason = ""
    recent_execution_timestamps: deque[float] = deque()
    recent_execution_fees: deque[tuple[float, int]] = deque()
    recent_execution_fee_total = 0
    recent_pnl_deltas: deque[tuple[float, float]] = deque()
    recent_pnl_total = 0.0
    last_execution_at = 0.0
    consecutive_execution_errors = 0
    execution_rate_limited_count = 0
    execution_circuit_open_until = 0.0
    drawdown_circuit_open_until = 0.0
    peak_wallet_balance_lamports: int | None = None
    final_stop_active = False
    next_pending_recovery_at = loop.time()
    requote_decay_guard_until = 0.0
    requote_decay_required_net_bps_boost = 0.0
    last_initial_breakdown_log_at = 0.0
    probe_window: deque[dict[str, Any]] = deque()
    stage_a_signal_window: deque[tuple[float, bool]] = deque()
    stage_a_long_window: deque[tuple[float, bool]] = deque()
    last_edge_summary_log_at = 0.0
    last_edge_summary_1m_log_at = 0.0
    last_edge_summary_5m_log_at = 0.0
    last_rate_limit_pause_log_at = 0.0
    runtime_config_logged = False
    env_priority_redis_synced = False
    env_priority_last_sync_at = 0.0
    env_priority_sync_interval_seconds = 10.0

    async def open_execution_circuit(*, reason: str, error: str = "") -> None:
        nonlocal execution_circuit_open_until, consecutive_execution_errors

        cooldown_seconds = app_settings.live_execution_circuit_breaker_seconds
        execution_circuit_open_until = loop.time() + cooldown_seconds
        log_event(
            logger,
            level="warning",
            event="execution_circuit_breaker_opened",
            message="Execution circuit breaker opened after repeated execution errors",
            reason=reason,
            error=error,
            cooldown_seconds=cooldown_seconds,
            threshold=app_settings.live_max_consecutive_execution_errors,
        )
        await guarded_call(
            lambda: storage.record_position(
                {
                    "pair": pair.symbol,
                    "status": "execution_circuit_open",
                    "reason": reason,
                    "error": error,
                }
            ),
            logger=logger,
            event="execution_circuit_position_record_failed",
            message="Failed to record execution circuit breaker state",
        )
        await guarded_call(
            lambda: storage.publish_event(
                level="WARNING",
                event="execution_circuit_breaker_opened",
                message="Execution circuit breaker opened",
                details={
                    "pair": pair.symbol,
                    "reason": reason,
                    "error": error,
                    "cooldown_seconds": cooldown_seconds,
                    "threshold": app_settings.live_max_consecutive_execution_errors,
                },
            ),
            logger=logger,
            event="execution_circuit_publish_failed",
            message="Failed to publish execution circuit breaker event",
        )
        consecutive_execution_errors = 0

    soft_reject_error_codes = {"0x1771", "0x1788", "0x1789"}

    def extract_custom_error_code(error: Exception) -> str | None:
        error_text = str(error).lower()
        match = re.search(r"custom program error:\s*(0x[0-9a-f]+)", error_text)
        if not match:
            return None
        return match.group(1)

    def is_soft_execution_reject(error: Exception) -> bool:
        error_text = str(error).lower()
        error_code = extract_custom_error_code(error)
        return (
            error_code in soft_reject_error_codes
            or "insufficient funds" in error_text
            or "slippage tolerance exceeded" in error_text
        )

    def classify_soft_execution_reject(error: Exception) -> str:
        error_text = str(error).lower()
        error_code = extract_custom_error_code(error)
        if "insufficient funds" in error_text:
            return "insufficient_funds"
        if error_code == "0x1771" or "slippage" in error_text:
            return "slippage_or_price_moved"
        if "compute" in error_text:
            return "compute_budget"
        if error_code in {"0x1788", "0x1789"} or "account" in error_text:
            return "account_or_route_state"
        return "unknown"

    def fail_reason_from_soft_reject_category(category: str) -> str:
        if category == "slippage_or_price_moved":
            return FAIL_REASON_SLIPPAGE_EXCEEDED
        if category in {"account_or_route_state", "insufficient_funds"}:
            return FAIL_REASON_ACCOUNT_LIMIT
        if category == "compute_budget":
            return FAIL_REASON_SIMULATION_FAIL
        return FAIL_REASON_SIMULATION_FAIL

    def fail_reason_from_execution_result(status: str, reason: str) -> str:
        normalized_status = str(status or "").strip().lower()
        normalized_reason = str(reason or "").strip().lower()
        if normalized_status == "rate_limited":
            return FAIL_REASON_RATE_LIMITED
        if normalized_status in {"filled", "dry_run", "pending_confirmation"}:
            return ""
        if normalized_status == "not_landed":
            return FAIL_REASON_NOT_LANDED
        if normalized_status in {
            "skipped_requote_unprofitable",
            "skipped_bundle_thin_opportunity",
            "skipped_bundle_unprofitable_with_tip",
            "skipped_min_expected_profit",
            "skipped_bundle_tip_zero",
        }:
            return FAIL_REASON_BELOW_STAGEB_REQUIRED
        if normalized_status == "skipped_probe_limit_neg_net":
            return FAIL_REASON_PROBE_LIMIT_NEG_NET
        if normalized_status == "skipped_probe_limit_loss_lamports":
            return FAIL_REASON_PROBE_LIMIT_LOSS_LAMPORTS
        if normalized_status == "skipped_probe_limit_max_base_amount":
            return FAIL_REASON_PROBE_LIMIT_MAX_BASE_AMOUNT
        if "simulation" in normalized_reason:
            return FAIL_REASON_SIMULATION_FAIL
        if "slippage" in normalized_reason:
            return FAIL_REASON_SLIPPAGE_EXCEEDED
        if "insufficient funds" in normalized_reason:
            return FAIL_REASON_ACCOUNT_LIMIT
        if normalized_status.startswith("skipped_"):
            return FAIL_REASON_OTHER
        if normalized_status == "failed":
            return FAIL_REASON_TX_FAIL
        return FAIL_REASON_OTHER

    def lamports_to_bps(*, lamports: int, notional_lamports: int) -> float:
        if notional_lamports <= 0:
            return 0.0
        return (max(0, lamports) / notional_lamports) * 10_000

    def normalize_quote_params(payload: Any) -> dict[str, str | int | float | bool]:
        if not isinstance(payload, dict):
            return {}
        normalized: dict[str, str | int | float | bool] = {}
        for raw_key, raw_value in payload.items():
            key = str(raw_key).strip()
            if not key:
                continue
            if isinstance(raw_value, bool):
                normalized[key] = raw_value
                continue
            if isinstance(raw_value, int):
                normalized[key] = raw_value
                continue
            if isinstance(raw_value, float):
                normalized[key] = int(raw_value) if raw_value.is_integer() else raw_value
                continue
            value = str(raw_value).strip()
            if not value:
                continue
            lowered = value.lower()
            if lowered in {"true", "false"}:
                normalized[key] = lowered == "true"
                continue
            try:
                normalized[key] = int(value)
                continue
            except ValueError:
                pass
            try:
                parsed_float = float(value)
                normalized[key] = int(parsed_float) if parsed_float.is_integer() else parsed_float
                continue
            except ValueError:
                pass
            normalized[key] = value
        return normalized

    def parse_quote_params_json(raw: str) -> dict[str, str | int | float | bool]:
        text = str(raw or "").strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return normalize_quote_params(parsed)

    env_quote_default_params = parse_quote_params_json(runtime_defaults.quote_default_params_json)
    env_quote_initial_params = parse_quote_params_json(runtime_defaults.quote_initial_params_json)
    env_quote_plan_params = parse_quote_params_json(runtime_defaults.quote_plan_params_json)
    if not env_quote_initial_params:
        env_quote_initial_params = dict(env_quote_default_params)
    if not env_quote_plan_params:
        env_quote_plan_params = dict(env_quote_default_params)

    def resolve_initial_quote_params(
        *,
        redis_config: dict[str, str],
        runtime_config: RuntimeConfig,
    ) -> tuple[dict[str, str | int | float | bool], str]:
        if "quote_initial_params_json" in redis_config:
            parsed = parse_quote_params_json(runtime_config.quote_initial_params_json)
            if parsed:
                return parsed, "redis.quote_initial_params_json"
        if "quote_default_params_json" in redis_config:
            parsed = parse_quote_params_json(runtime_config.quote_default_params_json)
            if parsed:
                return parsed, "redis.quote_default_params_json"
        if env_quote_initial_params:
            return dict(env_quote_initial_params), "env.QUOTE_INITIAL_PARAMS_JSON"
        if env_quote_default_params:
            return dict(env_quote_default_params), "env.QUOTE_DEFAULT_PARAMS_JSON"
        return {}, "watcher.default"

    def resolve_plan_quote_params(
        *,
        redis_config: dict[str, str],
        runtime_config: RuntimeConfig,
    ) -> tuple[dict[str, str | int | float | bool], str]:
        if "quote_plan_params_json" in redis_config:
            parsed = parse_quote_params_json(runtime_config.quote_plan_params_json)
            if parsed:
                return parsed, "redis.quote_plan_params_json"
        if "quote_default_params_json" in redis_config:
            parsed = parse_quote_params_json(runtime_config.quote_default_params_json)
            if parsed:
                return parsed, "redis.quote_default_params_json"
        if env_quote_plan_params:
            return dict(env_quote_plan_params), "env.QUOTE_PLAN_PARAMS_JSON"
        if env_quote_default_params:
            return dict(env_quote_default_params), "env.QUOTE_DEFAULT_PARAMS_JSON"
        return {}, "watcher.default"

    runtime_dump_spec: tuple[tuple[str, str | None, str | None], ...] = (
        ("trade_enabled", "trade_enabled", "trade_enabled"),
        ("TRADE_ENABLED", "trade_enabled", "trade_enabled"),
        ("ATOMIC_SEND_MODE", "atomic_send_mode", "atomic_send_mode"),
        ("ATOMIC_EXPIRY_MS", "atomic_expiry_ms", "atomic_expiry_ms"),
        ("MIN_SPREAD_BPS", "min_spread_bps", "min_spread_bps"),
        ("INITIAL_MIN_SPREAD_BPS", "initial_min_spread_bps", "initial_min_spread_bps"),
        ("ATOMIC_MARGIN_BPS", "atomic_margin_bps", "atomic_margin_bps"),
        ("INITIAL_ATOMIC_MARGIN_BPS", "initial_atomic_margin_bps", "initial_atomic_margin_bps"),
        ("MIN_STAGEA_MARGIN_BPS", "min_stagea_margin_bps", "min_stagea_margin_bps"),
        ("ALLOW_STAGEB_FAIL_PROBE", "allow_stageb_fail_probe", "allow_stageb_fail_probe"),
        ("QUOTE_EXPLORATION_MODE", "quote_exploration_mode", "quote_exploration_mode"),
        ("ENABLE_PROBE_MULTI_AMOUNT", "enable_probe_multi_amount", "enable_probe_multi_amount"),
        ("ENABLE_PROBE_UNCONSTRAINED", "enable_probe_unconstrained", "enable_probe_unconstrained"),
        ("ENABLE_LEGACY_SWAP_FALLBACK", "enable_legacy_swap_fallback", "enable_legacy_swap_fallback"),
        (
            "PRIORITY_FEE_MICRO_LAMPORTS",
            "priority_fee_micro_lamports",
            "priority_fee_micro_lamports",
        ),
        ("PRIORITY_COMPUTE_UNITS", "priority_compute_units", "priority_compute_units"),
        ("QUOTE_DEX_SWEEP_TOPK", "quote_dex_sweep_topk", "quote_dex_sweep_topk"),
        ("QUOTE_DEX_SWEEP_COMBO_LIMIT", "quote_dex_sweep_combo_limit", "quote_dex_sweep_combo_limit"),
        ("JITO_TIP_LAMPORTS_MAX", "jito_tip_lamports_max", "jito_tip_lamports_max"),
        (
            "JITO_TIP_LAMPORTS_RECOMMENDED",
            "jito_tip_lamports_recommended",
            "jito_tip_lamports_recommended",
        ),
        ("JITO_TIP_SHARE", "jito_tip_share", "jito_tip_share"),
        (
            "BASE_AMOUNT_SWEEP_CANDIDATES_RAW",
            "base_amount_sweep_candidates_raw",
            "base_amount_sweep_candidates_raw",
        ),
        ("BASE_AMOUNT_MAX_RAW", "base_amount_max_raw", "base_amount_max_raw"),
        ("PAIR_BASE_AMOUNT", None, None),
        ("QUOTE_MAX_RPS", "quote_max_rps", "quote_max_rps"),
        ("QUOTE_EXPLORATION_MAX_RPS", "quote_exploration_max_rps", "quote_exploration_max_rps"),
        ("QUOTE_EXECUTION_MAX_RPS", "quote_execution_max_rps", "quote_execution_max_rps"),
        ("MAX_REQUOTE_RANGE_BPS", "max_requote_range_bps", "max_requote_range_bps"),
    )

    def build_runtime_dump_values(config: RuntimeConfig) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for output_key, attr_name, _redis_key in runtime_dump_spec:
            if output_key == "PAIR_BASE_AMOUNT":
                values[output_key] = int(pair.base_amount)
                continue
            if attr_name is None:
                values[output_key] = None
                continue
            raw_value = getattr(config, attr_name)
            if isinstance(raw_value, tuple):
                values[output_key] = list(raw_value)
            else:
                values[output_key] = raw_value
        return values

    def build_runtime_dump_sources(
        *,
        priority_mode: str,
        redis_config: dict[str, str],
    ) -> tuple[dict[str, str], dict[str, str]]:
        value_sources: dict[str, str] = {}
        value_transports: dict[str, str] = {}
        for output_key, _attr_name, redis_key in runtime_dump_spec:
            if output_key == "PAIR_BASE_AMOUNT":
                value_sources[output_key] = "env"
                value_transports[output_key] = "env"
                continue
            if priority_mode == "env":
                value_sources[output_key] = "env"
                value_transports[output_key] = "env"
                continue
            if redis_key and redis_key in redis_config:
                value_sources[output_key] = "firestore"
                value_transports[output_key] = "redis"
            else:
                value_sources[output_key] = "env"
                value_transports[output_key] = "env"
        return value_sources, value_transports

    def build_amount_sweep(
        base_amount: int,
        *,
        multipliers: tuple[float, ...],
        candidates_raw: tuple[int, ...],
        max_raw: int,
    ) -> list[int]:
        cap = max(0, int(max_raw))
        amounts: list[int] = []

        if candidates_raw:
            for raw_amount in candidates_raw:
                amount = max(1, int(raw_amount))
                if cap > 0:
                    amount = min(amount, cap)
                if amount not in amounts:
                    amounts.append(amount)
        else:
            for multiplier in multipliers:
                amount = max(1, int(base_amount * float(multiplier)))
                if cap > 0:
                    amount = min(amount, cap)
                if amount not in amounts:
                    amounts.append(amount)
        base = max(1, int(base_amount))
        if cap > 0:
            base = min(base, cap)
        if base not in amounts:
            amounts.insert(0, base)
        return amounts or [base]

    def merge_counter(counter_map: dict[str, int], key: str, delta: int = 1) -> None:
        if delta == 0:
            return
        counter_map[key] = counter_map.get(key, 0) + int(delta)

    async def emit_runtime_counters(counter_map: dict[str, int]) -> None:
        if not counter_map:
            return
        await guarded_call(
            lambda: storage.increment_runtime_counters(counters=counter_map),
            logger=logger,
            event="runtime_counter_update_failed",
            message="Failed to update runtime counters in Redis",
            level="warning",
        )

    def emit_execution_preflight_summary(
        *,
        runtime_config: RuntimeConfig,
        initial_gate_passed: bool | None = None,
        stage_a_pass: bool | None = None,
        stage_b_pass: bool | None = None,
        stage_a_ok: bool | None = None,
        stage_b_ok: bool | None = None,
        is_probe_trade: bool | None = None,
        expected_net_bps_stage_a: float | None = None,
        expected_net_bps_stage_b: float | None = None,
        expected_net_lamports_single: int | None = None,
        expected_net_bps: float | None = None,
        expected_net_lamports: int | None = None,
        guard_block_reason: str = "",
        order_guard_blocked: bool = False,
        final_skip_reason: str = "",
        rate_limit_pause_until_epoch: float | None = None,
    ) -> None:
        pause_until = max(0.0, float(rate_limit_pause_until_epoch or 0.0))
        pause_remaining_seconds = max(0.0, pause_until - time.time())
        resolved_expected_net_bps_stage_a = (
            float(expected_net_bps_stage_a)
            if expected_net_bps_stage_a is not None
            else (float(expected_net_bps) if expected_net_bps is not None else None)
        )
        resolved_expected_net_bps_stage_b = (
            float(expected_net_bps_stage_b)
            if expected_net_bps_stage_b is not None
            else resolved_expected_net_bps_stage_a
        )
        resolved_expected_net_lamports_single = (
            int(expected_net_lamports_single)
            if expected_net_lamports_single is not None
            else (int(expected_net_lamports) if expected_net_lamports is not None else None)
        )
        resolved_stage_a_ok = (
            bool(stage_a_ok)
            if stage_a_ok is not None
            else bool(initial_gate_passed and stage_a_pass)
        )
        resolved_stage_b_ok = (
            bool(stage_b_ok)
            if stage_b_ok is not None
            else bool(stage_b_pass)
        )
        resolved_is_probe_trade = (
            bool(is_probe_trade)
            if is_probe_trade is not None
            else bool(runtime_config.allow_stageb_fail_probe and resolved_stage_a_ok and not resolved_stage_b_ok)
        )
        log_event(
            logger,
            level="info",
            event="execution_preflight_summary",
            message="Execution preflight decision summary",
            trade_enabled=runtime_config.trade_enabled,
            dry_run=app_settings.dry_run,
            initial_gate_passed=initial_gate_passed,
            stageA_pass=stage_a_pass,
            stageB_pass=stage_b_pass,
            stageA_ok=resolved_stage_a_ok,
            stageB_ok=resolved_stage_b_ok,
            allow_stageb_fail_probe=bool(runtime_config.allow_stageb_fail_probe),
            is_probe_trade=resolved_is_probe_trade,
            probe_limits={
                "max_neg_net_bps": PROBE_MAX_NEG_NET_BPS,
                "max_loss_lamports": PROBE_MAX_LOSS_LAMPORTS,
                "max_base_amount_lamports": PROBE_MAX_BASE_AMOUNT_LAMPORTS,
            },
            expected_net_bps_stageA=(
                round(resolved_expected_net_bps_stage_a, 6)
                if resolved_expected_net_bps_stage_a is not None
                else None
            ),
            expected_net_bps_stageB=(
                round(resolved_expected_net_bps_stage_b, 6)
                if resolved_expected_net_bps_stage_b is not None
                else None
            ),
            expected_net_lamports_single=resolved_expected_net_lamports_single,
            expected_net_bps=(
                round(resolved_expected_net_bps_stage_a, 6)
                if resolved_expected_net_bps_stage_a is not None
                else None
            ),
            expected_net_lamports=(
                resolved_expected_net_lamports_single
                if resolved_expected_net_lamports_single is not None
                else None
            ),
            chosen_send_mode=(
                runtime_config.atomic_send_mode
                if runtime_config.execution_mode == "atomic"
                else "legacy"
            ),
            guard_block_reason=guard_block_reason or "",
            order_guard_blocked=bool(order_guard_blocked),
            final_skip_reason=final_skip_reason or "",
            rate_limit_pause_until=(
                round(pause_until, 3)
                if pause_until > 0
                else None
            ),
            rate_limit_pause_remaining_seconds=(
                round(pause_remaining_seconds, 3)
                if pause_remaining_seconds > 0
                else None
            ),
        )

    def classify_execution_fail_reason(error: Exception) -> str:
        error_text = str(error).lower()
        error_code = extract_custom_error_code(error)
        if "429" in error_text or "rate limit" in error_text:
            return FAIL_REASON_RATE_LIMITED
        if "simulation failed" in error_text:
            if error_code == "0x1771" or "slippage" in error_text:
                return FAIL_REASON_SLIPPAGE_EXCEEDED
            if error_code in {"0x1788", "0x1789"} or "account" in error_text:
                return FAIL_REASON_ACCOUNT_LIMIT
            return FAIL_REASON_SIMULATION_FAIL
        if "slippage" in error_text or error_code == "0x1771":
            return FAIL_REASON_SLIPPAGE_EXCEEDED
        if "insufficient funds" in error_text or error_code in {"0x1788", "0x1789"}:
            return FAIL_REASON_ACCOUNT_LIMIT
        return FAIL_REASON_TX_FAIL

    def normalize_fail_reason(raw: str | None) -> str:
        value = str(raw or "").strip().upper()
        if not value:
            return ""
        known = {
            FAIL_REASON_BELOW_STAGEA_REQUIRED,
            FAIL_REASON_BELOW_STAGEB_REQUIRED,
            FAIL_REASON_ROUTE_UNSTABLE_REQUOTE_DROPPED,
            FAIL_REASON_RATE_LIMITED,
            FAIL_REASON_SIMULATION_FAIL,
            FAIL_REASON_TX_FAIL,
            FAIL_REASON_NOT_LANDED,
            FAIL_REASON_PROBE_LIMIT_NEG_NET,
            FAIL_REASON_PROBE_LIMIT_LOSS_LAMPORTS,
            FAIL_REASON_PROBE_LIMIT_MAX_BASE_AMOUNT,
            FAIL_REASON_SLIPPAGE_EXCEEDED,
            FAIL_REASON_ACCOUNT_LIMIT,
            FAIL_REASON_OTHER,
        }
        return value if value in known else FAIL_REASON_OTHER

    def prune_edge_windows(now_time: float) -> None:
        summary_window = max(app_settings.live_edge_summary_window_seconds, 300.0)
        while probe_window and (now_time - float(probe_window[0].get("timestamp", 0.0))) > summary_window:
            probe_window.popleft()

        no_edge_window = app_settings.live_edge_no_edge_window_seconds
        while stage_a_signal_window and (now_time - stage_a_signal_window[0][0]) > no_edge_window:
            stage_a_signal_window.popleft()

        long_window = app_settings.live_edge_consider_stop_window_seconds
        while stage_a_long_window and (now_time - stage_a_long_window[0][0]) > long_window:
            stage_a_long_window.popleft()

    def percentile(values: list[float], ratio: float) -> float | None:
        if not values:
            return None
        sorted_values = sorted(values)
        normalized_ratio = min(max(ratio, 0.0), 1.0)
        index = min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * normalized_ratio)))
        return float(sorted_values[index])

    def summarize_probe_window(items: list[dict[str, Any]]) -> dict[str, Any]:
        spread_probe_count = len(items)
        median_requote_count = sum(1 for item in items if bool(item.get("median_requote_applied")))
        stage_a_pass_count = sum(1 for item in items if bool(item.get("stage_a_pass")))
        stage_b_pass_count = sum(1 for item in items if bool(item.get("stage_b_pass")))
        avg_best_spread_bps = (
            sum(float(item.get("best_spread_bps", 0.0)) for item in items) / spread_probe_count
            if spread_probe_count > 0
            else 0.0
        )
        avg_stage_a_required_bps = (
            sum(float(item.get("stage_a_required_bps", 0.0)) for item in items) / spread_probe_count
            if spread_probe_count > 0
            else 0.0
        )
        stage_a_pass_rate = (stage_a_pass_count / spread_probe_count) if spread_probe_count > 0 else 0.0

        spread_values = [float(item.get("best_spread_bps", 0.0)) for item in items]
        requote_range_values = [
            float(item.get("median_requote_range_bps"))
            for item in items
            if item.get("median_requote_range_bps") is not None
        ]
        route_candidate_values = [float(item.get("route_candidate_count", 0.0)) for item in items]
        route_sampled_values = [float(item.get("route_sampled_count", 0.0)) for item in items]

        fail_reason_counts: dict[str, int] = {}
        for item in items:
            reason = str(item.get("fail_reason") or "")
            if not reason:
                continue
            fail_reason_counts[reason] = fail_reason_counts.get(reason, 0) + 1

        return {
            "spread_probe_count": spread_probe_count,
            "median_requote_count": median_requote_count,
            "stageA_pass_count": stage_a_pass_count,
            "stageB_pass_count": stage_b_pass_count,
            "stageA_pass_rate": stage_a_pass_rate,
            "avg_best_spread_bps": avg_best_spread_bps,
            "avg_stageA_required_spread_bps": avg_stage_a_required_bps,
            "best_spread_bps_p50": percentile(spread_values, 0.5),
            "best_spread_bps_p90": percentile(spread_values, 0.9),
            "requote_range_bps_p50": percentile(requote_range_values, 0.5),
            "requote_range_bps_p90": percentile(requote_range_values, 0.9),
            "route_candidate_count_p50": percentile(route_candidate_values, 0.5),
            "route_candidate_count_p90": percentile(route_candidate_values, 0.9),
            "route_sampled_count_p50": percentile(route_sampled_values, 0.5),
            "route_sampled_count_p90": percentile(route_sampled_values, 0.9),
            "fail_reason_counts": dict(sorted(fail_reason_counts.items(), key=lambda item: item[0])),
        }

    def probe_items_within_window(now_time: float, window_sec: float) -> list[dict[str, Any]]:
        if window_sec <= 0:
            return list(probe_window)
        return [
            item
            for item in probe_window
            if (now_time - float(item.get("timestamp", 0.0))) <= window_sec
        ]

    def edge_signal(now_time: float) -> tuple[str, int, int, int, int]:
        prune_edge_windows(now_time)
        no_edge_total = len(stage_a_signal_window)
        no_edge_pass = sum(1 for _ts, passed in stage_a_signal_window if passed)
        long_total = len(stage_a_long_window)
        long_pass = sum(1 for _ts, passed in stage_a_long_window if passed)

        if no_edge_total > 0 and no_edge_pass == 0:
            return "MARKET_NO_EDGE", no_edge_total, no_edge_pass, long_total, long_pass

        if long_total > 0:
            long_pass_rate_pct = (long_pass / long_total) * 100.0
            if long_pass_rate_pct < app_settings.live_edge_consider_stop_pass_rate_pct:
                return "CONSIDER_STOPPING", no_edge_total, no_edge_pass, long_total, long_pass

        if no_edge_pass > 0 or long_pass > 0:
            return "EDGE_PRESENT", no_edge_total, no_edge_pass, long_total, long_pass
        return "INSUFFICIENT_DATA", no_edge_total, no_edge_pass, long_total, long_pass

    async def emit_edge_summary(now_time: float, *, force: bool = False) -> None:
        nonlocal last_edge_summary_log_at, last_edge_summary_1m_log_at, last_edge_summary_5m_log_at
        if (
            not force
            and app_settings.live_edge_summary_interval_seconds > 0
            and (now_time - last_edge_summary_log_at) < app_settings.live_edge_summary_interval_seconds
        ):
            return

        prune_edge_windows(now_time)
        summary_items = probe_items_within_window(now_time, app_settings.live_edge_summary_window_seconds)
        summary = summarize_probe_window(summary_items)
        signal, no_edge_total, no_edge_pass, long_total, long_pass = edge_signal(now_time)
        long_pass_rate_pct = (long_pass / long_total * 100.0) if long_total > 0 else 0.0

        last_edge_summary_log_at = now_time
        log_event(
            logger,
            level="info",
            event="window_edge_summary",
            message="Rolling edge summary for initial-stage opportunity quality",
            window_sec=app_settings.live_edge_summary_window_seconds,
            spread_probe_count=summary["spread_probe_count"],
            median_requote_count=summary["median_requote_count"],
            stageA_pass_count=summary["stageA_pass_count"],
            stageB_pass_count=summary["stageB_pass_count"],
            stageA_pass_rate=round(float(summary["stageA_pass_rate"]), 6),
            avg_best_spread_bps=round(float(summary["avg_best_spread_bps"]), 6),
            avg_stageA_required_spread_bps=round(float(summary["avg_stageA_required_spread_bps"]), 6),
            best_spread_bps_p50=(
                round(float(summary["best_spread_bps_p50"]), 6)
                if summary["best_spread_bps_p50"] is not None
                else None
            ),
            best_spread_bps_p90=(
                round(float(summary["best_spread_bps_p90"]), 6)
                if summary["best_spread_bps_p90"] is not None
                else None
            ),
            requote_range_bps_p50=(
                round(float(summary["requote_range_bps_p50"]), 6)
                if summary["requote_range_bps_p50"] is not None
                else None
            ),
            requote_range_bps_p90=(
                round(float(summary["requote_range_bps_p90"]), 6)
                if summary["requote_range_bps_p90"] is not None
                else None
            ),
            route_candidate_count_p50=(
                round(float(summary["route_candidate_count_p50"]), 6)
                if summary["route_candidate_count_p50"] is not None
                else None
            ),
            route_candidate_count_p90=(
                round(float(summary["route_candidate_count_p90"]), 6)
                if summary["route_candidate_count_p90"] is not None
                else None
            ),
            route_sampled_count_p50=(
                round(float(summary["route_sampled_count_p50"]), 6)
                if summary["route_sampled_count_p50"] is not None
                else None
            ),
            route_sampled_count_p90=(
                round(float(summary["route_sampled_count_p90"]), 6)
                if summary["route_sampled_count_p90"] is not None
                else None
            ),
            fail_reason_counts=summary["fail_reason_counts"],
            signal=signal,
            signal_no_edge_window_sec=app_settings.live_edge_no_edge_window_seconds,
            signal_no_edge_probe_count=no_edge_total,
            signal_no_edge_pass_count=no_edge_pass,
            signal_consider_stop_window_sec=app_settings.live_edge_consider_stop_window_seconds,
            signal_consider_stop_probe_count=long_total,
            signal_consider_stop_pass_rate_pct=round(long_pass_rate_pct, 6),
            signal_consider_stop_threshold_pct=round(
                app_settings.live_edge_consider_stop_pass_rate_pct,
                6,
            ),
        )
        await guarded_call(
            lambda: storage.set_runtime_summary(
                values={
                    "updated_at": now_time,
                    "window_sec": app_settings.live_edge_summary_window_seconds,
                    "spread_probe_count": summary["spread_probe_count"],
                    "median_requote_count": summary["median_requote_count"],
                    "stageA_pass_count": summary["stageA_pass_count"],
                    "stageB_pass_count": summary["stageB_pass_count"],
                    "stageA_pass_rate": summary["stageA_pass_rate"],
                    "avg_best_spread_bps": round(float(summary["avg_best_spread_bps"]), 6),
                    "avg_stageA_required_spread_bps": round(float(summary["avg_stageA_required_spread_bps"]), 6),
                    "best_spread_bps_p50": summary["best_spread_bps_p50"],
                    "best_spread_bps_p90": summary["best_spread_bps_p90"],
                    "requote_range_bps_p50": summary["requote_range_bps_p50"],
                    "requote_range_bps_p90": summary["requote_range_bps_p90"],
                    "route_candidate_count_p50": summary["route_candidate_count_p50"],
                    "route_candidate_count_p90": summary["route_candidate_count_p90"],
                    "route_sampled_count_p50": summary["route_sampled_count_p50"],
                    "route_sampled_count_p90": summary["route_sampled_count_p90"],
                    "fail_reason_counts": summary["fail_reason_counts"],
                    "signal": signal,
                    "signal_no_edge_probe_count": no_edge_total,
                    "signal_no_edge_pass_count": no_edge_pass,
                    "signal_consider_stop_probe_count": long_total,
                    "signal_consider_stop_pass_rate_pct": round(long_pass_rate_pct, 6),
                    "signal_consider_stop_threshold_pct": round(
                        app_settings.live_edge_consider_stop_pass_rate_pct,
                        6,
                    ),
                }
            ),
            logger=logger,
            event="runtime_summary_update_failed",
            message="Failed to update rolling edge summary in Redis",
            level="warning",
        )

        if force or (now_time - last_edge_summary_1m_log_at) >= 60.0:
            last_edge_summary_1m_log_at = now_time
            summary_1m = summarize_probe_window(probe_items_within_window(now_time, 60.0))
            log_event(
                logger,
                level="info",
                event="window_edge_summary_1m",
                message="1-minute edge summary",
                window_sec=60.0,
                **summary_1m,
            )

        if force or (now_time - last_edge_summary_5m_log_at) >= 300.0:
            last_edge_summary_5m_log_at = now_time
            summary_5m = summarize_probe_window(probe_items_within_window(now_time, 300.0))
            log_event(
                logger,
                level="info",
                event="window_edge_summary_5m",
                message="5-minute edge summary",
                window_sec=300.0,
                **summary_5m,
            )

    def refresh_recent_pnl(now_time: float) -> float:
        nonlocal recent_pnl_total
        window_seconds = app_settings.live_drawdown_window_seconds
        while recent_pnl_deltas and (now_time - recent_pnl_deltas[0][0]) > window_seconds:
            _, expired_delta = recent_pnl_deltas.popleft()
            recent_pnl_total -= expired_delta
        return max(0.0, -recent_pnl_total)

    def record_realized_pnl(*, now_time: float, pnl_delta_lamports: float) -> None:
        nonlocal recent_pnl_total
        recent_pnl_deltas.append((now_time, pnl_delta_lamports))
        recent_pnl_total += pnl_delta_lamports

    async def open_drawdown_circuit(
        *,
        reason: str,
        current_drawdown_lamports: float,
        current_drawdown_pct: float | None = None,
    ) -> None:
        nonlocal drawdown_circuit_open_until

        cooldown_seconds = app_settings.live_drawdown_circuit_breaker_seconds
        drawdown_circuit_open_until = loop.time() + cooldown_seconds
        max_drawdown_pct = app_settings.live_max_drawdown_pct
        log_event(
            logger,
            level="warning",
            event="drawdown_circuit_breaker_opened",
            message="Drawdown circuit breaker opened",
            reason=reason,
            cooldown_seconds=cooldown_seconds,
            current_drawdown_lamports=round(current_drawdown_lamports, 4),
            max_drawdown_lamports=round(app_settings.live_max_drawdown_lamports, 4),
            current_drawdown_pct=round(current_drawdown_pct, 4) if current_drawdown_pct is not None else None,
            max_drawdown_pct=round(max_drawdown_pct, 4) if max_drawdown_pct > 0 else None,
        )
        await guarded_call(
            lambda: storage.record_position(
                {
                    "pair": pair.symbol,
                    "status": "drawdown_circuit_open",
                    "reason": reason,
                    "current_drawdown_lamports": round(current_drawdown_lamports, 4),
                    "current_drawdown_pct": (
                        round(current_drawdown_pct, 4) if current_drawdown_pct is not None else None
                    ),
                }
            ),
            logger=logger,
            event="drawdown_circuit_position_record_failed",
            message="Failed to record drawdown circuit breaker state",
        )
        await guarded_call(
            lambda: storage.publish_event(
                level="WARNING",
                event="drawdown_circuit_breaker_opened",
                message="Drawdown circuit breaker opened",
                details={
                    "pair": pair.symbol,
                    "reason": reason,
                    "current_drawdown_lamports": round(current_drawdown_lamports, 4),
                    "max_drawdown_lamports": round(app_settings.live_max_drawdown_lamports, 4),
                    "current_drawdown_pct": (
                        round(current_drawdown_pct, 4) if current_drawdown_pct is not None else None
                    ),
                    "max_drawdown_pct": round(max_drawdown_pct, 4) if max_drawdown_pct > 0 else None,
                    "cooldown_seconds": cooldown_seconds,
                },
            ),
            logger=logger,
            event="drawdown_circuit_publish_failed",
            message="Failed to publish drawdown circuit breaker event",
        )

    async def enforce_equity_guards(
        *,
        observation_forward_price: float,
        now_time: float,
    ) -> bool:
        nonlocal peak_wallet_balance_lamports, final_stop_active

        if app_settings.dry_run:
            return False

        if final_stop_active:
            return True

        needs_balance_check = (
            app_settings.live_max_drawdown_pct > 0
            or app_settings.live_final_stop_equity_usd > 0
        )
        if not needs_balance_check:
            return False

        balance_reader = getattr(trader_engine.executor, "get_wallet_balance_lamports", None)
        if not callable(balance_reader):
            return False

        wallet_balance_raw = await guarded_call(
            balance_reader,
            logger=logger,
            event="wallet_balance_fetch_failed",
            message="Failed to fetch wallet balance for equity guard checks",
            level="warning",
            default=None,
        )
        if wallet_balance_raw is None:
            return False

        wallet_balance_lamports = int(to_float(wallet_balance_raw, -1))
        if wallet_balance_lamports < 0:
            return False

        if peak_wallet_balance_lamports is None or wallet_balance_lamports > peak_wallet_balance_lamports:
            peak_wallet_balance_lamports = wallet_balance_lamports

        peak_balance = max(1, peak_wallet_balance_lamports or wallet_balance_lamports)
        balance_drawdown_lamports = max(0, peak_balance - wallet_balance_lamports)
        balance_drawdown_pct = drawdown_pct_from_peak(
            current_lamports=wallet_balance_lamports,
            peak_lamports=peak_balance,
        )

        if (
            app_settings.live_max_drawdown_pct > 0
            and balance_drawdown_pct >= app_settings.live_max_drawdown_pct
        ):
            if drawdown_circuit_open_until <= now_time:
                await open_drawdown_circuit(
                    reason="drawdown_pct_limit_exceeded",
                    current_drawdown_lamports=float(balance_drawdown_lamports),
                    current_drawdown_pct=balance_drawdown_pct,
                )
            return True

        if app_settings.live_final_stop_equity_usd <= 0:
            return False

        if observation_forward_price <= 0:
            log_event(
                logger,
                level="warning",
                event="final_equity_stop_price_unavailable",
                message="Final equity USD stop could not be evaluated because forward price was unavailable",
            )
            return False

        wallet_balance_units = wallet_balance_lamports / (10**pair.base_decimals)
        current_equity_usd = wallet_balance_units * observation_forward_price
        if current_equity_usd > app_settings.live_final_stop_equity_usd:
            return False

        final_stop_active = True
        log_event(
            logger,
            level="error",
            event="final_equity_stop_triggered",
            message="Final equity stop triggered; execution is now disabled until manual intervention",
            current_equity_usd=round(current_equity_usd, 6),
            final_stop_equity_usd=round(app_settings.live_final_stop_equity_usd, 6),
            wallet_balance_lamports=wallet_balance_lamports,
            peak_wallet_balance_lamports=peak_balance,
            current_drawdown_pct=round(balance_drawdown_pct, 6),
        )
        await guarded_call(
            lambda: storage.record_position(
                {
                    "pair": pair.symbol,
                    "status": "final_stop_active",
                    "reason": "equity_below_usd_threshold",
                    "current_equity_usd": round(current_equity_usd, 6),
                    "final_stop_equity_usd": round(app_settings.live_final_stop_equity_usd, 6),
                    "wallet_balance_lamports": wallet_balance_lamports,
                }
            ),
            logger=logger,
            event="final_equity_stop_position_record_failed",
            message="Failed to record final equity stop state",
        )
        await guarded_call(
            lambda: storage.publish_event(
                level="ERROR",
                event="final_equity_stop_triggered",
                message="Final equity stop triggered and execution has been disabled",
                details={
                    "pair": pair.symbol,
                    "current_equity_usd": round(current_equity_usd, 6),
                    "final_stop_equity_usd": round(app_settings.live_final_stop_equity_usd, 6),
                    "wallet_balance_lamports": wallet_balance_lamports,
                    "peak_wallet_balance_lamports": peak_balance,
                    "current_drawdown_pct": round(balance_drawdown_pct, 6),
                },
            ),
            logger=logger,
            event="final_equity_stop_publish_failed",
            message="Failed to publish final equity stop event",
        )
        return True

    while not stop_event.is_set():
        transient_backoff_seconds = 0.0
        runtime_config = runtime_defaults

        try:
            if order_intake_paused:
                recovered = await try_resume_order_intake(
                    logger=logger,
                    storage=storage,
                    trader_engine=trader_engine,
                    pause_reason=pause_reason,
                )
                if not recovered:
                    await guarded_call(
                        storage.update_heartbeat,
                        logger=logger,
                        event="order_intake_paused_heartbeat_failed",
                        message="Failed to update heartbeat while intake is paused",
                    )
                    continue

                order_intake_paused = False
                pause_reason = ""
                await guarded_call(
                    lambda: storage.record_position(
                        {
                            "pair": pair.symbol,
                            "status": "order_intake_resumed",
                            "reason": "dependencies_recovered",
                        }
                    ),
                    logger=logger,
                    event="order_intake_resumed_position_record_failed",
                    message="Failed to record resumed position state",
                )
                await guarded_call(
                    lambda: storage.publish_event(
                        level="INFO",
                        event="order_intake_resumed",
                        message="Order intake resumed after successful recovery",
                        details={"pair": pair.symbol},
                        event_id=f"order_intake_resumed:{storage.run_id}",
                    ),
                    logger=logger,
                    event="order_intake_resumed_publish_failed",
                    message="Failed to publish order_intake_resumed event",
                )

            redis_config = await storage.get_runtime_config()
            firestore_runtime_config = RuntimeConfig.from_redis(redis_config, runtime_defaults)
            if app_settings.env_config_priority == "env":
                runtime_config = runtime_defaults
                should_reconcile_env_priority = (
                    firestore_runtime_config != runtime_defaults
                    and (loop.time() - env_priority_last_sync_at) >= env_priority_sync_interval_seconds
                )
                if (not env_priority_redis_synced) or should_reconcile_env_priority:
                    sync_source = "env_priority" if not env_priority_redis_synced else "env_priority_reconcile"
                    await guarded_call(
                        lambda: storage.sync_config_to_redis(
                            asdict(runtime_defaults),
                            source=sync_source,
                        ),
                        logger=logger,
                        event="env_priority_redis_sync_failed",
                        message="Failed to sync env-priority runtime config to Redis",
                    )
                    env_priority_redis_synced = True
                    env_priority_last_sync_at = loop.time()
                    redis_config = await storage.get_runtime_config()
                    firestore_runtime_config = RuntimeConfig.from_redis(redis_config, runtime_defaults)
            else:
                runtime_config = firestore_runtime_config

            priority_fee_plan = await trader_engine.resolve_priority_fee(runtime_config=runtime_config)

            if app_settings.env_config_priority == "env":
                if env_quote_initial_params:
                    initial_quote_params = dict(env_quote_initial_params)
                    initial_quote_params_source = "env.QUOTE_INITIAL_PARAMS_JSON"
                elif env_quote_default_params:
                    initial_quote_params = dict(env_quote_default_params)
                    initial_quote_params_source = "env.QUOTE_DEFAULT_PARAMS_JSON"
                else:
                    initial_quote_params = {}
                    initial_quote_params_source = "watcher.default"

                if env_quote_plan_params:
                    plan_quote_params = dict(env_quote_plan_params)
                    plan_quote_params_source = "env.QUOTE_PLAN_PARAMS_JSON"
                elif env_quote_default_params:
                    plan_quote_params = dict(env_quote_default_params)
                    plan_quote_params_source = "env.QUOTE_DEFAULT_PARAMS_JSON"
                else:
                    plan_quote_params = {}
                    plan_quote_params_source = "watcher.default"
            else:
                initial_quote_params, initial_quote_params_source = resolve_initial_quote_params(
                    redis_config=redis_config,
                    runtime_config=runtime_config,
                )
                plan_quote_params, plan_quote_params_source = resolve_plan_quote_params(
                    redis_config=redis_config,
                    runtime_config=runtime_config,
                )

            if not runtime_config_logged:
                effective_values = build_runtime_dump_values(runtime_config)
                effective_sources, effective_transports = build_runtime_dump_sources(
                    priority_mode=app_settings.env_config_priority,
                    redis_config=redis_config,
                )
                log_fields: dict[str, Any] = {
                    "config_priority": app_settings.env_config_priority,
                    "runtime_transport": (
                        "env"
                        if app_settings.env_config_priority == "env"
                        else "redis"
                    ),
                    "execution_mode": runtime_config.execution_mode,
                    "initial_quote_params_source": initial_quote_params_source,
                    "plan_quote_params_source": plan_quote_params_source,
                    "runtime_config_items": len(redis_config),
                }
                for key, value in effective_values.items():
                    log_fields[key] = value
                    log_fields[f"{key}_source"] = effective_sources.get(key, "env")
                    log_fields[f"{key}_transport"] = effective_transports.get(key, "env")

                if app_settings.env_config_priority == "env":
                    firestore_candidate_values = build_runtime_dump_values(firestore_runtime_config)
                    for key, value in firestore_candidate_values.items():
                        log_fields[f"{key}_firestore_candidate"] = value

                log_event(
                    logger,
                    level="info",
                    event="effective_config_dump_runtime",
                    message="Effective runtime configuration at first trading loop",
                    **log_fields,
                )
                runtime_config_logged = True

            set_plan_quote_params = getattr(trader_engine.executor, "set_plan_quote_params", None)
            if callable(set_plan_quote_params):
                set_plan_quote_params(plan_quote_params)

            rate_limit_pause_until_epoch = max(
                0.0,
                float(
                    await guarded_call(
                        storage.get_rate_limit_pause_until,
                        logger=logger,
                        event="rate_limit_pause_read_failed",
                        message="Failed to read shared rate-limit pause state from Redis",
                        level="warning",
                        default=0.0,
                    )
                ),
            )
            watcher_pause_until_epoch = 0.0
            watcher_pause_getter = getattr(trader_engine.watcher, "rate_limit_pause_until_epoch", None)
            if callable(watcher_pause_getter):
                try:
                    watcher_pause_until_epoch = max(0.0, float(watcher_pause_getter()))
                except Exception:
                    watcher_pause_until_epoch = 0.0
            rate_limit_pause_until_epoch = max(rate_limit_pause_until_epoch, watcher_pause_until_epoch)
            watcher_pause_setter = getattr(trader_engine.watcher, "set_external_rate_limit_pause_until", None)
            if callable(watcher_pause_setter) and rate_limit_pause_until_epoch > 0:
                watcher_pause_setter(
                    pause_until_epoch=rate_limit_pause_until_epoch,
                    source="redis",
                )

            pause_remaining_seconds = max(0.0, rate_limit_pause_until_epoch - time.time())
            if pause_remaining_seconds > 0:
                now_time = loop.time()
                if (now_time - last_rate_limit_pause_log_at) >= 5.0:
                    last_rate_limit_pause_log_at = now_time
                    log_event(
                        logger,
                        level="warning",
                        event="quote_rate_limit_pause_active",
                        message="Quote exploration is paused because provider rate limit cooldown is active",
                        pause_until_epoch=round(rate_limit_pause_until_epoch, 3),
                        remaining_seconds=round(pause_remaining_seconds, 3),
                    )
                emit_execution_preflight_summary(
                    runtime_config=runtime_config,
                    initial_gate_passed=False,
                    guard_block_reason="rate_limited_pause_active",
                    final_skip_reason="rate_limited_pause_active",
                    rate_limit_pause_until_epoch=rate_limit_pause_until_epoch,
                )
                await guarded_call(
                    storage.update_heartbeat,
                    logger=logger,
                    event="rate_limit_pause_heartbeat_failed",
                    message="Failed to update heartbeat during shared rate-limit pause",
                )
                continue

            now_time = loop.time()
            if not app_settings.dry_run and now_time >= next_pending_recovery_at:
                recover_pending = getattr(trader_engine.executor, "recover_pending", None)
                if callable(recover_pending):
                    await guarded_call(
                        recover_pending,
                        logger=logger,
                        event="pending_recovery_failed",
                        message="Failed to recover pending live executions",
                        level="warning",
                    )
                next_pending_recovery_at = now_time + app_settings.live_pending_recovery_interval_seconds

            amount_candidates = build_amount_sweep(
                pair.base_amount,
                multipliers=runtime_config.base_amount_sweep_multipliers,
                candidates_raw=runtime_config.base_amount_sweep_candidates_raw,
                max_raw=runtime_config.base_amount_max_raw,
            )
            dex_allowlist_forward = runtime_config.quote_dex_allowlist_forward
            dex_allowlist_reverse = runtime_config.quote_dex_allowlist_reverse or dex_allowlist_forward
            selected_pair: PairConfig | None = None
            selected_observation = None
            selected_decision = None
            amount_sweep_results: list[dict[str, Any]] = []
            for amount_raw in amount_candidates:
                pair_candidate = pair if amount_raw == pair.base_amount else replace(pair, base_amount=amount_raw)
                priority_fee_lamports_candidate = int(
                    (priority_fee_plan.recommended_micro_lamports * runtime_config.priority_compute_units) / 1_000_000
                )
                priority_fee_bps_candidate = lamports_to_bps(
                    lamports=priority_fee_lamports_candidate,
                    notional_lamports=pair_candidate.base_amount,
                )
                stage_a_required_bps_candidate = (
                    runtime_config.initial_min_spread_bps
                    + runtime_config.dex_fee_bps
                    + priority_fee_bps_candidate
                    + runtime_config.initial_atomic_margin_bps
                )
                stage_a_required_components_candidate = {
                    "min_spread_bps": float(runtime_config.initial_min_spread_bps),
                    "dex_fee_bps": float(runtime_config.dex_fee_bps),
                    "slippage_buffer_bps": float(runtime_config.initial_atomic_margin_bps),
                    "priority_fee_bps": float(priority_fee_bps_candidate),
                    "ata_cost_bps": 0.0,
                    "extra_accounts_cost_bps": 0.0,
                }
                stage_b_required_bps_without_tip_candidate = (
                    runtime_config.min_spread_bps
                    + runtime_config.dex_fee_bps
                    + priority_fee_bps_candidate
                    + runtime_config.atomic_margin_bps
                )
                stage_b_required_components_candidate = {
                    "min_spread_bps": float(runtime_config.min_spread_bps),
                    "dex_fee_bps": float(runtime_config.dex_fee_bps),
                    "slippage_buffer_bps": float(runtime_config.atomic_margin_bps),
                    "priority_fee_bps": float(priority_fee_bps_candidate),
                    "ata_cost_bps": 0.0,
                    "extra_accounts_cost_bps": 0.0,
                    "tip_fee_bps": 0.0,
                }
                try:
                    observation_candidate = await trader_engine.watcher.fetch_spread(
                        pair_candidate,
                        quote_params=initial_quote_params,
                        quote_params_source=initial_quote_params_source,
                        dex_allowlist_forward=dex_allowlist_forward,
                        dex_allowlist_reverse=dex_allowlist_reverse,
                        dex_excludelist=runtime_config.quote_dex_excludelist,
                        exploration_mode=runtime_config.quote_exploration_mode,
                        sweep_top_k=runtime_config.quote_dex_sweep_topk,
                        sweep_top_k_max=runtime_config.quote_dex_sweep_topk_max,
                        sweep_combo_limit=runtime_config.quote_dex_sweep_combo_limit,
                        near_miss_expand_bps=runtime_config.quote_near_miss_expand_bps,
                        median_requote_max_range_bps=runtime_config.max_requote_range_bps,
                        min_improvement_bps=runtime_config.min_improvement_bps,
                        quote_max_rps=runtime_config.quote_max_rps,
                        quote_exploration_max_rps=runtime_config.quote_exploration_max_rps,
                        quote_execution_max_rps=runtime_config.quote_execution_max_rps,
                        quote_cache_ttl_ms=runtime_config.quote_cache_ttl_ms,
                        no_routes_cache_ttl_seconds=runtime_config.quote_no_routes_cache_ttl_seconds,
                        probe_max_routes=runtime_config.quote_probe_max_routes,
                        probe_base_amounts_raw=runtime_config.quote_probe_base_amounts_raw,
                        dynamic_allowlist_topk=runtime_config.quote_dynamic_allowlist_topk,
                        dynamic_allowlist_good_candidate_alpha=(
                            runtime_config.quote_dynamic_allowlist_good_candidate_alpha
                        ),
                        dynamic_allowlist_ttl_seconds=runtime_config.quote_dynamic_allowlist_ttl_seconds,
                        dynamic_allowlist_refresh_seconds=runtime_config.quote_dynamic_allowlist_refresh_seconds,
                        negative_fallback_streak_threshold=runtime_config.quote_negative_fallback_streak_threshold,
                        enable_probe_unconstrained=runtime_config.enable_probe_unconstrained,
                        enable_probe_multi_amount=runtime_config.enable_probe_multi_amount,
                        enable_stagea_relaxed_gate=runtime_config.enable_stagea_relaxed_gate,
                        enable_route_instability_cooldown=runtime_config.enable_route_instability_cooldown,
                        route_instability_cooldown_requote_seconds=(
                            runtime_config.route_instability_cooldown_requote_seconds
                        ),
                        route_instability_cooldown_decay_requote_bps=(
                            runtime_config.route_instability_cooldown_decay_requote_bps
                        ),
                        stage_a_required_bps=stage_a_required_bps_candidate,
                        stage_a_min_margin_bps=runtime_config.min_stagea_margin_bps,
                        stage_a_required_components=stage_a_required_components_candidate,
                        stage_b_required_bps_without_tip=stage_b_required_bps_without_tip_candidate,
                        stage_b_tip_share=runtime_config.jito_tip_share,
                        stage_b_tip_lamports_max=runtime_config.jito_tip_lamports_max,
                        stage_b_required_components=stage_b_required_components_candidate,
                    )
                except HeliusRateLimitError as error:
                    retry_after = max(0.0, float(error.retry_after_seconds or 0.0))
                    pause_until_epoch_from_watcher = 0.0
                    watcher_pause_getter = getattr(trader_engine.watcher, "rate_limit_pause_until_epoch", None)
                    if callable(watcher_pause_getter):
                        with contextlib.suppress(Exception):
                            pause_until_epoch_from_watcher = max(0.0, float(watcher_pause_getter()))
                    pause_until_epoch = max(
                        pause_until_epoch_from_watcher,
                        time.time() + max(60.0, retry_after),
                    )
                    await guarded_call(
                        lambda: storage.set_rate_limit_pause_until(
                            pause_until_epoch=pause_until_epoch,
                        ),
                        logger=logger,
                        event="rate_limit_pause_write_failed",
                        message="Failed to persist quote rate-limit pause window to Redis",
                        level="warning",
                    )
                    if selected_decision is None:
                        raise
                    log_event(
                        logger,
                        level="warning",
                        event="amount_sweep_truncated_rate_limited",
                        message=(
                            "Rate limit interrupted amount sweep; reusing the best collected candidate "
                            "for this cycle."
                        ),
                        attempted_amount_raw=int(pair_candidate.base_amount),
                        selected_amount_raw=int(selected_pair.base_amount) if selected_pair else None,
                        collected_candidate_count=len(amount_sweep_results),
                        retry_after_seconds=round(retry_after, 3) if retry_after > 0 else None,
                        pause_until_epoch=round(pause_until_epoch, 3),
                    )
                    await emit_runtime_counters({"rate_limited_count": 1})
                    break
                decision_candidate = trader_engine.evaluate(
                    observation=observation_candidate,
                    runtime_config=runtime_config,
                    pair=pair_candidate,
                    priority_fee_plan=priority_fee_plan,
                )

                amount_sweep_results.append(
                    {
                        "amount_raw": int(pair_candidate.base_amount),
                        "observed_spread_bps": round(decision_candidate.spread_bps, 6),
                        "expected_net_bps": round(decision_candidate.expected_net_bps_single, 6),
                        "expected_net_lamports": int(decision_candidate.expected_net_lamports_single),
                    }
                )
                if selected_decision is None:
                    selected_pair = pair_candidate
                    selected_observation = observation_candidate
                    selected_decision = decision_candidate
                    continue

                if (
                    decision_candidate.spread_bps > selected_decision.spread_bps
                    or (
                        decision_candidate.spread_bps == selected_decision.spread_bps
                        and decision_candidate.expected_net_lamports_single
                        > selected_decision.expected_net_lamports_single
                    )
                    or (
                        decision_candidate.spread_bps == selected_decision.spread_bps
                        and decision_candidate.expected_net_lamports_single
                        == selected_decision.expected_net_lamports_single
                        and decision_candidate.expected_net_bps_single > selected_decision.expected_net_bps_single
                    )
                ):
                    selected_pair = pair_candidate
                    selected_observation = observation_candidate
                    selected_decision = decision_candidate

            if selected_pair is None or selected_observation is None or selected_decision is None:
                raise RuntimeError("Amount sweep failed to produce any observation candidates.")

            pair_for_cycle = selected_pair
            observation = selected_observation
            decision = selected_decision

            await storage.record_price(
                pair=pair.symbol,
                price=observation.forward_price,
                raw={
                    "forward_out_amount": observation.forward_out_amount,
                    "reverse_out_amount": observation.reverse_out_amount,
                    "spread_bps": observation.spread_bps,
                    "timestamp": observation.timestamp,
                    "base_amount_raw": pair_for_cycle.base_amount,
                    "quote_params_source": initial_quote_params_source,
                },
            )
            priority_fee_lamports = int(
                (priority_fee_plan.recommended_micro_lamports * runtime_config.priority_compute_units) / 1_000_000
            )
            priority_fee_bps = lamports_to_bps(
                lamports=priority_fee_lamports,
                notional_lamports=pair_for_cycle.base_amount,
            )
            tip_fee_bps = lamports_to_bps(
                lamports=decision.tip_lamports,
                notional_lamports=pair_for_cycle.base_amount,
            )
            tip_lamports_effective = int(decision.tip_lamports)
            tip_fee_bps_effective = float(tip_fee_bps)
            stage_a_required_bps = float(decision.required_spread_bps_single)
            expected_net_bps_stage_a = float(decision.expected_net_bps_single)
            expected_profit_lamports_stage_a = max(0, int(decision.expected_net_lamports_single))
            stage_b_tip_share = max(0.0, min(1.0, float(runtime_config.jito_tip_share)))
            stage_b_tip_lamports_effective = int(expected_profit_lamports_stage_a * stage_b_tip_share)
            if expected_profit_lamports_stage_a > 0 and stage_b_tip_share > 0 and stage_b_tip_lamports_effective <= 0:
                stage_b_tip_lamports_effective = 1
            stage_b_tip_max = max(0, int(runtime_config.jito_tip_lamports_max))
            if stage_b_tip_max > 0:
                stage_b_tip_lamports_effective = min(stage_b_tip_lamports_effective, stage_b_tip_max)
            stage_b_tip_fee_bps_effective = lamports_to_bps(
                lamports=stage_b_tip_lamports_effective,
                notional_lamports=pair_for_cycle.base_amount,
            )
            stage_b_required_bps = (
                runtime_config.min_spread_bps
                + runtime_config.dex_fee_bps
                + priority_fee_bps
                + runtime_config.atomic_margin_bps
                + stage_b_tip_fee_bps_effective
            )
            expected_net_bps_stage_b = float(decision.spread_bps - stage_b_required_bps)
            expected_net_lamports_stage_b = int((pair_for_cycle.base_amount * expected_net_bps_stage_b) / 10_000)
            base_amount_raw = int(pair_for_cycle.base_amount)
            forward_out_raw = int(observation.forward_out_amount)
            reverse_in_raw = int(
                to_float(observation.reverse_quote.get("inAmount"), float(forward_out_raw))
            )
            reverse_out_raw = int(observation.reverse_out_amount)
            computed_spread_bps_check = (
                ((reverse_out_raw - base_amount_raw) / base_amount_raw) * 10_000
                if base_amount_raw > 0
                else 0.0
            )

            stage_a_pass = (
                observation.stage_a_pass
                if observation.stage_a_pass is not None
                else (decision.expected_net_bps_single >= 0)
            )
            stage_a_margin_bps = (
                observation.stage_a_margin_bps
                if observation.stage_a_margin_bps is not None
                else decision.expected_net_bps_single
            )
            stage_b_pass = (
                observation.stage_b_pass
                if observation.stage_b_pass is not None
                else (expected_net_bps_stage_b >= 0)
            )
            stage_b_margin_bps = (
                observation.stage_b_margin_bps
                if observation.stage_b_margin_bps is not None
                else expected_net_bps_stage_b
            )

            stage_a_required_components = {
                "min_spread_bps": float(runtime_config.initial_min_spread_bps),
                "dex_fee_bps": float(runtime_config.dex_fee_bps),
                "slippage_buffer_bps": float(runtime_config.initial_atomic_margin_bps),
                "priority_fee_bps": float(priority_fee_bps),
                "ata_cost_bps": 0.0,
                "extra_accounts_cost_bps": 0.0,
                "min_stagea_margin_bps": float(runtime_config.min_stagea_margin_bps),
            }
            stage_b_required_components = {
                "min_spread_bps": float(runtime_config.min_spread_bps),
                "dex_fee_bps": float(runtime_config.dex_fee_bps),
                "slippage_buffer_bps": float(runtime_config.atomic_margin_bps),
                "priority_fee_bps": float(priority_fee_bps),
                "ata_cost_bps": 0.0,
                "extra_accounts_cost_bps": 0.0,
                "tip_fee_bps": float(stage_b_tip_fee_bps_effective),
            }
            stage_a_gate_mode = (
                "relaxed_non_negative" if runtime_config.enable_stagea_relaxed_gate else "strict_margin"
            )
            stage_a_pass_threshold_bps = (
                stage_a_required_bps
                if runtime_config.enable_stagea_relaxed_gate
                else stage_a_required_bps + float(runtime_config.min_stagea_margin_bps)
            )

            initial_requote_gate_bps = app_settings.live_initial_requote_gate_bps
            required_initial_profit_lamports = (
                int(runtime_config.initial_profit_gate_relaxed_lamports)
                if runtime_config.enable_initial_profit_gate_relaxed
                else int(runtime_config.min_expected_profit_lamports)
            )
            initial_gate_passed = (
                decision.expected_net_bps_single >= initial_requote_gate_bps
                or decision.expected_net_lamports_single >= required_initial_profit_lamports
            )
            stage_a_ok = bool(initial_gate_passed and stage_a_pass)
            stage_b_ok = bool(stage_b_pass)
            allow_stageb_fail_probe = bool(runtime_config.allow_stageb_fail_probe)
            is_probe_trade = bool(allow_stageb_fail_probe and stage_a_ok and not stage_b_ok)
            probe_limit_fail_reason = ""
            if is_probe_trade:
                if expected_net_bps_stage_b < PROBE_MAX_NEG_NET_BPS:
                    probe_limit_fail_reason = FAIL_REASON_PROBE_LIMIT_NEG_NET
                elif expected_net_lamports_stage_b < -PROBE_MAX_LOSS_LAMPORTS:
                    probe_limit_fail_reason = FAIL_REASON_PROBE_LIMIT_LOSS_LAMPORTS
                elif base_amount_raw > PROBE_MAX_BASE_AMOUNT_LAMPORTS:
                    probe_limit_fail_reason = FAIL_REASON_PROBE_LIMIT_MAX_BASE_AMOUNT
                if probe_limit_fail_reason:
                    is_probe_trade = False
            initial_fail_reason = normalize_fail_reason(observation.fail_reason)
            if initial_fail_reason == FAIL_REASON_ROUTE_UNSTABLE_REQUOTE_DROPPED:
                initial_gate_passed = False
            elif not stage_a_ok:
                initial_fail_reason = FAIL_REASON_BELOW_STAGEA_REQUIRED
                initial_gate_passed = False
            elif not stage_b_ok and not allow_stageb_fail_probe:
                initial_fail_reason = FAIL_REASON_BELOW_STAGEB_REQUIRED
            elif probe_limit_fail_reason:
                initial_fail_reason = probe_limit_fail_reason
            if not initial_gate_passed and not initial_fail_reason:
                initial_fail_reason = FAIL_REASON_BELOW_STAGEA_REQUIRED

            execute_allowed = bool(stage_b_ok or is_probe_trade)
            should_attempt_execution = (
                runtime_config.trade_enabled
                and not decision.blocked_by_fee_cap
                and execute_allowed
            )

            initial_breakdown_interval = app_settings.live_initial_breakdown_log_interval_seconds
            should_emit_initial_breakdown = (
                initial_breakdown_interval <= 0
                or (now_time - last_initial_breakdown_log_at) >= initial_breakdown_interval
            )
            if should_emit_initial_breakdown:
                last_initial_breakdown_log_at = now_time
                log_event(
                    logger,
                    level="info",
                    event="initial_decision_breakdown",
                    message="Initial decision breakdown",
                    pair=pair.symbol,
                    observed_spread_bps=round(decision.spread_bps, 6),
                    required_spread_bps=round(decision.required_spread_bps_single, 6),
                    required_spread_bps_single=round(decision.required_spread_bps_single, 6),
                    required_spread_bps_bundle=round(decision.required_spread_bps_bundle, 6),
                    initial_expected_net_bps=round(decision.expected_net_bps_single, 6),
                    expected_net_bps_stageA=round(expected_net_bps_stage_a, 6),
                    expected_net_bps_stageB=round(expected_net_bps_stage_b, 6),
                    expected_net_lamports_stageB=int(expected_net_lamports_stage_b),
                    initial_expected_net_bps_single=round(decision.expected_net_bps_single, 6),
                    initial_expected_net_bps_bundle=round(decision.expected_net_bps_bundle, 6),
                    initial_expected_net_lamports_single=int(decision.expected_net_lamports_single),
                    initial_expected_net_lamports_bundle=int(decision.expected_net_lamports_bundle),
                    min_spread_bps=round(runtime_config.min_spread_bps, 6),
                    min_spread_bps_initial=round(runtime_config.initial_min_spread_bps, 6),
                    min_spread_bps_execution=round(runtime_config.min_spread_bps, 6),
                    atomic_margin_bps=round(decision.atomic_margin_bps, 6),
                    atomic_margin_initial_bps=round(runtime_config.initial_atomic_margin_bps, 6),
                    atomic_margin_execution_bps=round(runtime_config.atomic_margin_bps, 6),
                    atomic_margin_single_bps=round(decision.atomic_margin_single_bps, 6),
                    atomic_margin_bundle_bps=round(decision.atomic_margin_bundle_bps, 6),
                    total_fee_bps_single=round(decision.total_fee_bps_single, 6),
                    total_fee_bps_bundle=round(decision.total_fee_bps_bundle, 6),
                    tip_fee_bps=round(tip_fee_bps, 6),
                    tip_lamports_effective=tip_lamports_effective,
                    tip_fee_bps_effective=round(tip_fee_bps_effective, 6),
                    priority_fee_bps=round(priority_fee_bps, 6),
                    priority_fee_micro_lamports=priority_fee_plan.recommended_micro_lamports,
                    stageA_required_spread_bps=round(stage_a_required_bps, 6),
                    stageA_pass_threshold_bps=round(stage_a_pass_threshold_bps, 6),
                    stageA_gate_mode=stage_a_gate_mode,
                    stageA_required_components=stage_a_required_components,
                    stageA_pass=stage_a_pass,
                    stageA_ok=stage_a_ok,
                    stageA_margin_bps=(
                        round(float(stage_a_margin_bps), 6)
                        if stage_a_margin_bps is not None
                        else None
                    ),
                    stageB_required_spread_bps=round(stage_b_required_bps, 6),
                    stageB_required_components=stage_b_required_components,
                    stageB_pass=stage_b_pass,
                    stageB_ok=stage_b_ok,
                    stageB_margin_bps=(
                        round(float(stage_b_margin_bps), 6)
                        if stage_b_margin_bps is not None
                        else None
                    ),
                    stageB_tip_share=round(stage_b_tip_share, 6),
                    stageB_tip_lamports_effective=stage_b_tip_lamports_effective,
                    stageB_tip_fee_bps_effective=round(stage_b_tip_fee_bps_effective, 6),
                    allow_stageb_fail_probe=allow_stageb_fail_probe,
                    is_probe_trade=is_probe_trade,
                    probe_limit_fail_reason=probe_limit_fail_reason or "",
                    probe_limits={
                        "max_neg_net_bps": PROBE_MAX_NEG_NET_BPS,
                        "max_loss_lamports": PROBE_MAX_LOSS_LAMPORTS,
                        "max_base_amount_lamports": PROBE_MAX_BASE_AMOUNT_LAMPORTS,
                    },
                    initial_fail_reason=initial_fail_reason,
                    exploration_mode=observation.exploration_mode,
                    min_expected_profit_lamports=int(runtime_config.min_expected_profit_lamports),
                    required_initial_profit_lamports=required_initial_profit_lamports,
                    initial_profit_gate_relaxed_enabled=runtime_config.enable_initial_profit_gate_relaxed,
                    initial_profit_gate_relaxed_lamports=int(runtime_config.initial_profit_gate_relaxed_lamports),
                    base_amount_raw=base_amount_raw,
                    forward_out_raw=forward_out_raw,
                    reverse_in_raw=reverse_in_raw,
                    reverse_out_raw=reverse_out_raw,
                    computed_spread_bps_check=round(computed_spread_bps_check, 6),
                    route_candidate_count=observation.route_candidate_count,
                    route_sampled_count=observation.route_sampled_count,
                    median_requote_applied=observation.median_requote_applied,
                    median_requote_sample_count=observation.median_requote_sample_count,
                    median_requote_range_bps=(
                        round(float(observation.median_requote_range_bps), 6)
                        if observation.median_requote_range_bps is not None
                        else None
                    ),
                    requote_spread_samples_bps=[
                        round(float(value), 6) for value in observation.requote_spread_samples_bps
                    ],
                    requote_samples_route_hashes_forward=list(observation.requote_samples_route_hashes_forward),
                    requote_samples_route_hashes_reverse=list(observation.requote_samples_route_hashes_reverse),
                    requote_median_spread_bps=(
                        round(float(observation.requote_median_spread_bps), 6)
                        if observation.requote_median_spread_bps is not None
                        else None
                    ),
                    unstable_drop_threshold_bps=(
                        round(float(observation.unstable_drop_threshold_bps), 6)
                        if observation.unstable_drop_threshold_bps is not None
                        else None
                    ),
                    best_spread_pre_requote=(
                        round(float(observation.best_spread_pre_requote), 6)
                        if observation.best_spread_pre_requote is not None
                        else None
                    ),
                    best_spread_post_requote=(
                        round(float(observation.best_spread_post_requote), 6)
                        if observation.best_spread_post_requote is not None
                        else None
                    ),
                    quote_params_source=initial_quote_params_source,
                    quote_plan_params_source=plan_quote_params_source,
                    quote_params=observation.quote_params,
                    forward_route_fingerprint="|".join(observation.forward_route_dexes),
                    reverse_route_fingerprint="|".join(observation.reverse_route_dexes),
                    forward_route_hash=observation.forward_route_hash,
                    reverse_route_hash=observation.reverse_route_hash,
                    route_cooldown_skipped_count=observation.route_cooldown_skipped_count,
                    selected_base_amount_raw=int(pair_for_cycle.base_amount),
                )
                if runtime_config.enable_priority_fee_breakdown_logging:
                    log_event(
                        logger,
                        level="info",
                        event="priority_fee_breakdown",
                        message="Priority fee breakdown for initial-stage spread gating",
                        pair=pair.symbol,
                        base_amount_raw=int(pair_for_cycle.base_amount),
                        priority_fee_micro_lamports_selected=int(priority_fee_plan.selected_micro_lamports),
                        priority_fee_micro_lamports_recommended=int(
                            priority_fee_plan.recommended_micro_lamports
                        ),
                        priority_compute_units=int(runtime_config.priority_compute_units),
                        priority_fee_lamports_estimate=int(priority_fee_lamports),
                        priority_fee_bps=round(float(priority_fee_bps), 6),
                        priority_fee_plan_source=priority_fee_plan.source,
                        priority_fee_plan_sample_size=int(priority_fee_plan.sample_size),
                    )
                if len(amount_sweep_results) > 1:
                    log_event(
                        logger,
                        level="info",
                        event="initial_amount_sweep",
                        message="Initial amount sweep results",
                        pair=pair.symbol,
                        selected_base_amount_raw=int(pair_for_cycle.base_amount),
                        results=amount_sweep_results,
                        min_expected_profit_lamports=int(runtime_config.min_expected_profit_lamports),
                        required_initial_profit_lamports=required_initial_profit_lamports,
                    )
                log_event(
                    logger,
                    level="info",
                    event="quote_params_effective",
                    message="Effective quote params resolved for initial and plan phases",
                    initial_quote_params_source=initial_quote_params_source,
                    plan_quote_params_source=plan_quote_params_source,
                    initial_quote_params=initial_quote_params,
                    plan_quote_params=plan_quote_params,
                )

            now_time = loop.time()
            probe_window.append(
                {
                    "timestamp": now_time,
                    "best_spread_bps": float(decision.spread_bps),
                    "stage_a_required_bps": float(stage_a_required_bps),
                    "stage_a_pass": bool(stage_a_pass),
                    "stage_b_pass": bool(stage_b_pass),
                    "median_requote_applied": bool(observation.median_requote_applied),
                    "median_requote_range_bps": (
                        float(observation.median_requote_range_bps)
                        if observation.median_requote_range_bps is not None
                        else None
                    ),
                    "route_candidate_count": int(observation.route_candidate_count),
                    "route_sampled_count": int(observation.route_sampled_count),
                    "fail_reason": initial_fail_reason,
                }
            )
            stage_a_signal_window.append((now_time, bool(stage_a_pass)))
            stage_a_long_window.append((now_time, bool(stage_a_pass)))
            prune_edge_windows(now_time)

            probe_counters: dict[str, int] = {}
            merge_counter(probe_counters, "spread_probe_count", 1)
            if observation.median_requote_applied:
                merge_counter(probe_counters, "median_requote_count", 1)
            if stage_a_pass:
                merge_counter(probe_counters, "stageA_pass_count", 1)
            if stage_b_pass:
                merge_counter(probe_counters, "stageB_pass_count", 1)
            if initial_fail_reason:
                merge_counter(probe_counters, f"fail_reason:{initial_fail_reason}", 1)
            await emit_runtime_counters(probe_counters)
            await emit_edge_summary(now_time)

            await storage.record_spread(
                pair=pair.symbol,
                spread_bps=decision.spread_bps,
                required_spread_bps=decision.required_spread_bps,
                total_fee_bps=decision.total_fee_bps,
                profitable=decision.profitable,
                extra={
                    "priority_fee_micro_lamports": decision.priority_fee_micro_lamports,
                    "priority_fee_source": priority_fee_plan.source,
                    "priority_fee_recommended_micro_lamports": priority_fee_plan.recommended_micro_lamports,
                    "priority_fee_sample_size": priority_fee_plan.sample_size,
                    "blocked_by_fee_cap": decision.blocked_by_fee_cap,
                    "tip_lamports": decision.tip_lamports,
                    "tip_fee_bps": decision.tip_fee_bps,
                    "tip_lamports_effective": tip_lamports_effective,
                    "tip_fee_bps_effective": tip_fee_bps_effective,
                    "atomic_margin_bps": decision.atomic_margin_bps,
                    "initial_min_spread_bps": runtime_config.initial_min_spread_bps,
                    "initial_atomic_margin_bps": runtime_config.initial_atomic_margin_bps,
                    "atomic_margin_single_bps": decision.atomic_margin_single_bps,
                    "atomic_margin_bundle_bps": decision.atomic_margin_bundle_bps,
                    "expected_net_bps": decision.expected_net_bps,
                    "expected_net_bps_stageA": expected_net_bps_stage_a,
                    "expected_net_bps_stageB": expected_net_bps_stage_b,
                    "expected_net_lamports_stageB": expected_net_lamports_stage_b,
                    "expected_net_bps_single": decision.expected_net_bps_single,
                    "expected_net_bps_bundle": decision.expected_net_bps_bundle,
                    "expected_net_lamports_single": decision.expected_net_lamports_single,
                    "expected_net_lamports_bundle": decision.expected_net_lamports_bundle,
                    "required_spread_bps_single": decision.required_spread_bps_single,
                    "required_spread_bps_bundle": decision.required_spread_bps_bundle,
                    "required_spread_bps_stageA": stage_a_required_bps,
                    "required_spread_bps_stageB": stage_b_required_bps,
                    "required_spread_bps_stageA_threshold": stage_a_pass_threshold_bps,
                    "stageA_gate_mode": stage_a_gate_mode,
                    "required_initial_profit_lamports": required_initial_profit_lamports,
                    "required_spread_components_stageA": stage_a_required_components,
                    "required_spread_components_stageB": stage_b_required_components,
                    "min_stagea_margin_bps": runtime_config.min_stagea_margin_bps,
                    "stageA_pass": stage_a_pass,
                    "stageA_ok": stage_a_ok,
                    "stageA_margin_bps": stage_a_margin_bps,
                    "stageB_pass": stage_b_pass,
                    "stageB_ok": stage_b_ok,
                    "stageB_margin_bps": stage_b_margin_bps,
                    "allow_stageb_fail_probe": allow_stageb_fail_probe,
                    "is_probe_trade": is_probe_trade,
                    "probe_limit_fail_reason": probe_limit_fail_reason or "",
                    "probe_limits": {
                        "max_neg_net_bps": PROBE_MAX_NEG_NET_BPS,
                        "max_loss_lamports": PROBE_MAX_LOSS_LAMPORTS,
                        "max_base_amount_lamports": PROBE_MAX_BASE_AMOUNT_LAMPORTS,
                    },
                    "fail_reason": initial_fail_reason,
                    "stageB_tip_share": stage_b_tip_share,
                    "stageB_tip_lamports_effective": stage_b_tip_lamports_effective,
                    "stageB_tip_fee_bps_effective": stage_b_tip_fee_bps_effective,
                    "total_fee_bps_single": decision.total_fee_bps_single,
                    "total_fee_bps_bundle": decision.total_fee_bps_bundle,
                    "execution_mode": runtime_config.execution_mode,
                    "atomic_send_mode": runtime_config.atomic_send_mode,
                    "exploration_mode": observation.exploration_mode,
                    "base_amount_raw": pair_for_cycle.base_amount,
                    "base_amount_sweep_candidates_raw": list(runtime_config.base_amount_sweep_candidates_raw),
                    "amount_sweep_results": amount_sweep_results,
                    "route_candidate_count": observation.route_candidate_count,
                    "route_sampled_count": observation.route_sampled_count,
                    "median_requote_applied": observation.median_requote_applied,
                    "median_requote_sample_count": observation.median_requote_sample_count,
                    "median_requote_range_bps": observation.median_requote_range_bps,
                    "requote_spread_samples_bps": list(observation.requote_spread_samples_bps),
                    "requote_samples_route_hashes_forward": list(observation.requote_samples_route_hashes_forward),
                    "requote_samples_route_hashes_reverse": list(observation.requote_samples_route_hashes_reverse),
                    "requote_median_spread_bps": observation.requote_median_spread_bps,
                    "best_spread_pre_requote": observation.best_spread_pre_requote,
                    "best_spread_post_requote": observation.best_spread_post_requote,
                    "unstable_drop_threshold_bps": observation.unstable_drop_threshold_bps,
                    "forward_route_hash": observation.forward_route_hash,
                    "reverse_route_hash": observation.reverse_route_hash,
                    "route_cooldown_skipped_count": observation.route_cooldown_skipped_count,
                    "quote_params_source": initial_quote_params_source,
                    "quote_plan_params_source": plan_quote_params_source,
                },
            )
            await storage.update_heartbeat()

            now_time = loop.time()
            equity_guard_blocked = await enforce_equity_guards(
                observation_forward_price=observation.forward_price,
                now_time=now_time,
            )
            if equity_guard_blocked:
                emit_execution_preflight_summary(
                    runtime_config=runtime_config,
                    initial_gate_passed=initial_gate_passed,
                    stage_a_pass=stage_a_pass,
                    stage_b_pass=stage_b_pass,
                    stage_a_ok=stage_a_ok,
                    stage_b_ok=stage_b_ok,
                    is_probe_trade=is_probe_trade,
                    expected_net_bps_stage_a=expected_net_bps_stage_a,
                    expected_net_bps_stage_b=expected_net_bps_stage_b,
                    expected_net_lamports_single=decision.expected_net_lamports_single,
                    expected_net_bps=decision.expected_net_bps_single,
                    expected_net_lamports=decision.expected_net_lamports_single,
                    guard_block_reason="equity_guard",
                    final_skip_reason="equity_guard",
                    rate_limit_pause_until_epoch=rate_limit_pause_until_epoch,
                )
                continue

            if should_attempt_execution:
                if not app_settings.dry_run and app_settings.live_max_drawdown_lamports > 0:
                    current_drawdown_lamports = refresh_recent_pnl(now_time)
                    if current_drawdown_lamports >= app_settings.live_max_drawdown_lamports:
                        if drawdown_circuit_open_until <= now_time:
                            await open_drawdown_circuit(
                                reason="drawdown_limit_exceeded",
                                current_drawdown_lamports=current_drawdown_lamports,
                            )
                        emit_execution_preflight_summary(
                            runtime_config=runtime_config,
                            initial_gate_passed=initial_gate_passed,
                            stage_a_pass=stage_a_pass,
                            stage_b_pass=stage_b_pass,
                            stage_a_ok=stage_a_ok,
                            stage_b_ok=stage_b_ok,
                            is_probe_trade=is_probe_trade,
                            expected_net_bps_stage_a=expected_net_bps_stage_a,
                            expected_net_bps_stage_b=expected_net_bps_stage_b,
                            expected_net_lamports_single=decision.expected_net_lamports_single,
                            expected_net_bps=decision.expected_net_bps_single,
                            expected_net_lamports=decision.expected_net_lamports_single,
                            guard_block_reason="drawdown_guard",
                            final_skip_reason="drawdown_guard",
                            rate_limit_pause_until_epoch=rate_limit_pause_until_epoch,
                        )
                        continue

                if (
                    not app_settings.dry_run
                    and requote_decay_guard_until > now_time
                ):
                    remaining_seconds = max(0.0, requote_decay_guard_until - now_time)
                    log_event(
                        logger,
                        level="warning",
                        event="execution_requote_decay_guard_active",
                        message="Execution skipped because requote-decay guard is active",
                        remaining_seconds=round(remaining_seconds, 3),
                        required_net_bps_boost=round(requote_decay_required_net_bps_boost, 6),
                    )
                    emit_execution_preflight_summary(
                        runtime_config=runtime_config,
                        initial_gate_passed=initial_gate_passed,
                        stage_a_pass=stage_a_pass,
                        stage_b_pass=stage_b_pass,
                        stage_a_ok=stage_a_ok,
                        stage_b_ok=stage_b_ok,
                        is_probe_trade=is_probe_trade,
                        expected_net_bps_stage_a=expected_net_bps_stage_a,
                        expected_net_bps_stage_b=expected_net_bps_stage_b,
                        expected_net_lamports_single=decision.expected_net_lamports_single,
                        expected_net_bps=decision.expected_net_bps_single,
                        expected_net_lamports=decision.expected_net_lamports_single,
                        guard_block_reason="requote_decay_guard",
                        final_skip_reason="requote_decay_guard",
                        rate_limit_pause_until_epoch=rate_limit_pause_until_epoch,
                    )
                    continue

                if (not app_settings.dry_run) and drawdown_circuit_open_until > now_time:
                    remaining_seconds = max(0.0, drawdown_circuit_open_until - now_time)
                    log_event(
                        logger,
                        level="warning",
                        event="drawdown_circuit_open",
                        message="Execution skipped because drawdown circuit breaker is open",
                        remaining_seconds=round(remaining_seconds, 3),
                    )
                    emit_execution_preflight_summary(
                        runtime_config=runtime_config,
                        initial_gate_passed=initial_gate_passed,
                        stage_a_pass=stage_a_pass,
                        stage_b_pass=stage_b_pass,
                        stage_a_ok=stage_a_ok,
                        stage_b_ok=stage_b_ok,
                        is_probe_trade=is_probe_trade,
                        expected_net_bps_stage_a=expected_net_bps_stage_a,
                        expected_net_bps_stage_b=expected_net_bps_stage_b,
                        expected_net_lamports_single=decision.expected_net_lamports_single,
                        expected_net_bps=decision.expected_net_bps_single,
                        expected_net_lamports=decision.expected_net_lamports_single,
                        guard_block_reason="drawdown_circuit_open",
                        final_skip_reason="drawdown_circuit_open",
                        rate_limit_pause_until_epoch=rate_limit_pause_until_epoch,
                    )
                    continue

                if (not app_settings.dry_run) and execution_circuit_open_until > now_time:
                    remaining_seconds = max(0.0, execution_circuit_open_until - now_time)
                    log_event(
                        logger,
                        level="warning",
                        event="execution_circuit_open",
                        message="Execution skipped because circuit breaker is open",
                        remaining_seconds=round(remaining_seconds, 3),
                    )
                    emit_execution_preflight_summary(
                        runtime_config=runtime_config,
                        initial_gate_passed=initial_gate_passed,
                        stage_a_pass=stage_a_pass,
                        stage_b_pass=stage_b_pass,
                        stage_a_ok=stage_a_ok,
                        stage_b_ok=stage_b_ok,
                        is_probe_trade=is_probe_trade,
                        expected_net_bps_stage_a=expected_net_bps_stage_a,
                        expected_net_bps_stage_b=expected_net_bps_stage_b,
                        expected_net_lamports_single=decision.expected_net_lamports_single,
                        expected_net_bps=decision.expected_net_bps_single,
                        expected_net_lamports=decision.expected_net_lamports_single,
                        guard_block_reason="execution_circuit_open",
                        final_skip_reason="execution_circuit_open",
                        rate_limit_pause_until_epoch=rate_limit_pause_until_epoch,
                    )
                    continue

                cooldown_seconds = app_settings.live_execution_cooldown_seconds
                if cooldown_seconds > 0 and (now_time - last_execution_at) < cooldown_seconds:
                    log_event(
                        logger,
                        level="warning",
                        event="execution_cooldown_active",
                        message="Execution skipped due to cooldown guard",
                        cooldown_seconds=cooldown_seconds,
                        remaining_seconds=round(cooldown_seconds - (now_time - last_execution_at), 3),
                    )
                    emit_execution_preflight_summary(
                        runtime_config=runtime_config,
                        initial_gate_passed=initial_gate_passed,
                        stage_a_pass=stage_a_pass,
                        stage_b_pass=stage_b_pass,
                        stage_a_ok=stage_a_ok,
                        stage_b_ok=stage_b_ok,
                        is_probe_trade=is_probe_trade,
                        expected_net_bps_stage_a=expected_net_bps_stage_a,
                        expected_net_bps_stage_b=expected_net_bps_stage_b,
                        expected_net_lamports_single=decision.expected_net_lamports_single,
                        expected_net_bps=decision.expected_net_bps_single,
                        expected_net_lamports=decision.expected_net_lamports_single,
                        guard_block_reason="execution_cooldown",
                        final_skip_reason="execution_cooldown",
                        rate_limit_pause_until_epoch=rate_limit_pause_until_epoch,
                    )
                    continue

                window_seconds = app_settings.live_execution_window_seconds
                while recent_execution_timestamps and (now_time - recent_execution_timestamps[0]) > window_seconds:
                    recent_execution_timestamps.popleft()

                if len(recent_execution_timestamps) >= app_settings.live_max_executions_per_window:
                    log_event(
                        logger,
                        level="warning",
                        event="execution_rate_limited",
                        message="Execution skipped due to rate-limit guard",
                        window_seconds=window_seconds,
                        max_executions=app_settings.live_max_executions_per_window,
                        current_count=len(recent_execution_timestamps),
                    )
                    emit_execution_preflight_summary(
                        runtime_config=runtime_config,
                        initial_gate_passed=initial_gate_passed,
                        stage_a_pass=stage_a_pass,
                        stage_b_pass=stage_b_pass,
                        stage_a_ok=stage_a_ok,
                        stage_b_ok=stage_b_ok,
                        is_probe_trade=is_probe_trade,
                        expected_net_bps_stage_a=expected_net_bps_stage_a,
                        expected_net_bps_stage_b=expected_net_bps_stage_b,
                        expected_net_lamports_single=decision.expected_net_lamports_single,
                        expected_net_bps=decision.expected_net_bps_single,
                        expected_net_lamports=decision.expected_net_lamports_single,
                        guard_block_reason="execution_window_rate_limit",
                        final_skip_reason="execution_window_rate_limit",
                        rate_limit_pause_until_epoch=rate_limit_pause_until_epoch,
                    )
                    continue

                runtime_config_for_execution = runtime_config
                if (
                    runtime_config.execution_mode == "atomic"
                    and requote_decay_required_net_bps_boost > 0
                ):
                    runtime_config_for_execution = replace(
                        runtime_config,
                        atomic_bundle_min_expected_net_bps=(
                            runtime_config.atomic_bundle_min_expected_net_bps
                            + requote_decay_required_net_bps_boost
                        ),
                    )

                estimated_fee_lamports = 0
                if not app_settings.dry_run:
                    pending_guard_active = await check_and_recover_pending_guard(
                        logger=logger,
                        storage=storage,
                        trader_engine=trader_engine,
                        runtime_config=runtime_config_for_execution,
                        app_settings=app_settings,
                    )
                    if pending_guard_active:
                        emit_execution_preflight_summary(
                            runtime_config=runtime_config,
                            initial_gate_passed=initial_gate_passed,
                            stage_a_pass=stage_a_pass,
                            stage_b_pass=stage_b_pass,
                            stage_a_ok=stage_a_ok,
                            stage_b_ok=stage_b_ok,
                            is_probe_trade=is_probe_trade,
                            expected_net_bps_stage_a=expected_net_bps_stage_a,
                            expected_net_bps_stage_b=expected_net_bps_stage_b,
                            expected_net_lamports_single=decision.expected_net_lamports_single,
                            expected_net_bps=decision.expected_net_bps_single,
                            expected_net_lamports=decision.expected_net_lamports_single,
                            guard_block_reason="pending_guard",
                            order_guard_blocked=True,
                            final_skip_reason="pending_guard",
                            rate_limit_pause_until_epoch=rate_limit_pause_until_epoch,
                        )
                        continue

                    tip_lamports = 0
                    if (
                        runtime_config_for_execution.execution_mode == "atomic"
                        and runtime_config_for_execution.atomic_send_mode != "single_tx"
                    ):
                        expected_profit_lamports = max(0, int(decision.expected_net_lamports_single))
                        tip_share = max(0.0, min(1.0, float(runtime_config_for_execution.jito_tip_share)))
                        max_tip = max(0, runtime_config_for_execution.jito_tip_lamports_max)
                        tip_lamports = int(expected_profit_lamports * tip_share)
                        if expected_profit_lamports > 0 and tip_share > 0 and tip_lamports <= 0:
                            tip_lamports = 1
                        if max_tip > 0:
                            tip_lamports = min(tip_lamports, max_tip)

                    estimated_fee_lamports = estimate_execution_fee_lamports(
                        priority_fee_micro_lamports=priority_fee_plan.selected_micro_lamports,
                        priority_compute_units=runtime_config_for_execution.priority_compute_units,
                        base_fee_lamports=app_settings.live_estimated_base_fee_lamports,
                        tip_lamports=tip_lamports,
                    )

                    fee_window_seconds = app_settings.live_fee_budget_window_seconds
                    while recent_execution_fees and (now_time - recent_execution_fees[0][0]) > fee_window_seconds:
                        _, expired_fee = recent_execution_fees.popleft()
                        recent_execution_fee_total = max(0, recent_execution_fee_total - expired_fee)

                    max_fee_per_window = app_settings.live_max_estimated_fee_lamports_per_window
                    if max_fee_per_window > 0 and (
                        recent_execution_fee_total + estimated_fee_lamports
                    ) > max_fee_per_window:
                        log_event(
                            logger,
                            level="warning",
                            event="execution_fee_budget_guard",
                            message="Execution skipped due to fee budget guard",
                            fee_window_seconds=fee_window_seconds,
                            estimated_fee_lamports=estimated_fee_lamports,
                            current_window_fee_lamports=recent_execution_fee_total,
                            max_window_fee_lamports=max_fee_per_window,
                            tip_lamports=tip_lamports,
                            tip_share=runtime_config_for_execution.jito_tip_share,
                        )
                        emit_execution_preflight_summary(
                            runtime_config=runtime_config,
                            initial_gate_passed=initial_gate_passed,
                            stage_a_pass=stage_a_pass,
                            stage_b_pass=stage_b_pass,
                            stage_a_ok=stage_a_ok,
                            stage_b_ok=stage_b_ok,
                            is_probe_trade=is_probe_trade,
                            expected_net_bps_stage_a=expected_net_bps_stage_a,
                            expected_net_bps_stage_b=expected_net_bps_stage_b,
                            expected_net_lamports_single=decision.expected_net_lamports_single,
                            expected_net_bps=decision.expected_net_bps_single,
                            expected_net_lamports=decision.expected_net_lamports_single,
                            guard_block_reason="fee_budget_guard",
                            final_skip_reason="fee_budget_guard",
                            rate_limit_pause_until_epoch=rate_limit_pause_until_epoch,
                        )
                        continue

                idempotency_key = trader_engine.build_idempotency_key(pair=pair_for_cycle, observation=observation)
                emit_execution_preflight_summary(
                    runtime_config=runtime_config_for_execution,
                    initial_gate_passed=initial_gate_passed,
                    stage_a_pass=stage_a_pass,
                    stage_b_pass=stage_b_pass,
                    stage_a_ok=stage_a_ok,
                    stage_b_ok=stage_b_ok,
                    is_probe_trade=is_probe_trade,
                    expected_net_bps_stage_a=expected_net_bps_stage_a,
                    expected_net_bps_stage_b=expected_net_bps_stage_b,
                    expected_net_lamports_single=decision.expected_net_lamports_single,
                    expected_net_bps=decision.expected_net_bps_single,
                    expected_net_lamports=decision.expected_net_lamports_single,
                    final_skip_reason="execution_attempt",
                    rate_limit_pause_until_epoch=rate_limit_pause_until_epoch,
                )
                await emit_runtime_counters({"stageA_execute_attempt_count": 1})
                observation_for_execution = (
                    replace(observation, is_probe_trade=is_probe_trade)
                    if observation.is_probe_trade != is_probe_trade
                    else observation
                )
                try:
                    result = await trader_engine.execute(
                        pair=pair_for_cycle,
                        observation=observation_for_execution,
                        runtime_config=runtime_config_for_execution,
                        priority_fee_plan=priority_fee_plan,
                        idempotency_key=idempotency_key,
                    )
                except TransactionPendingConfirmationError as error:
                    pending_retry_after_seconds = max(
                        app_settings.live_confirm_poll_interval_seconds,
                        min(app_settings.live_pending_guard_ttl_seconds, app_settings.error_backoff_seconds),
                    )
                    transient_backoff_seconds = max(
                        transient_backoff_seconds,
                        pending_retry_after_seconds,
                    )
                    log_event(
                        logger,
                        level="warning",
                        event="execution_pending_confirmation",
                        message="Execution is pending confirmation; keeping intake active without incrementing error circuit",
                        error=str(error),
                        pending_retry_after_seconds=round(pending_retry_after_seconds, 3),
                        consecutive_errors=consecutive_execution_errors,
                    )
                    await guarded_call(
                        storage.update_heartbeat,
                        logger=logger,
                        event="execution_pending_confirmation_heartbeat_failed",
                        message="Failed to update heartbeat during pending confirmation backoff",
                    )
                    continue
                except Exception as error:
                    if is_soft_execution_reject(error):
                        error_code = extract_custom_error_code(error)
                        reject_category = classify_soft_execution_reject(error)
                        fail_reason = fail_reason_from_soft_reject_category(reject_category)
                        log_event(
                            logger,
                            level="warning",
                            event="execution_soft_reject",
                            message=(
                                "Execution simulation was rejected by swap program; "
                                "skipping without incrementing circuit breaker errors"
                            ),
                            error=str(error),
                            custom_error_code=error_code,
                            reject_category=reject_category,
                            fail_reason=fail_reason,
                            consecutive_errors=consecutive_execution_errors,
                        )
                        await emit_runtime_counters(
                            {
                                "stageA_execute_fail_count": 1,
                                f"stageA_execute_fail_reason:{fail_reason}": 1,
                            }
                        )
                        transient_backoff_seconds = max(
                            transient_backoff_seconds,
                            app_settings.live_confirm_poll_interval_seconds,
                        )
                        await guarded_call(
                            storage.update_heartbeat,
                            logger=logger,
                            event="execution_soft_reject_heartbeat_failed",
                            message="Failed to update heartbeat after soft execution reject",
                        )
                        continue

                    consecutive_execution_errors += 1
                    fail_reason = classify_execution_fail_reason(error)
                    log_event(
                        logger,
                        level="warning",
                        event="execution_attempt_failed",
                        message="Execution attempt failed",
                        error=str(error),
                        fail_reason=fail_reason,
                        consecutive_errors=consecutive_execution_errors,
                        threshold=app_settings.live_max_consecutive_execution_errors,
                    )
                    await emit_runtime_counters(
                        {
                            "stageA_execute_fail_count": 1,
                            f"stageA_execute_fail_reason:{fail_reason}": 1,
                        }
                    )
                    if (
                        not app_settings.dry_run
                        and consecutive_execution_errors >= app_settings.live_max_consecutive_execution_errors
                    ):
                        await open_execution_circuit(
                            reason="consecutive_execution_errors",
                            error=str(error),
                        )
                    continue

                if result.status == "rate_limited":
                    execution_rate_limited_count += 1
                    fail_reason = FAIL_REASON_RATE_LIMITED
                    retry_after_seconds = to_float(result.metadata.get("retry_after_seconds"), 0.0)
                    adaptive_backoff_seconds = compute_rate_limit_backoff_seconds(
                        rate_limited_count=execution_rate_limited_count
                    )
                    transient_backoff_seconds = max(
                        transient_backoff_seconds,
                        retry_after_seconds,
                        adaptive_backoff_seconds,
                    )
                    log_event(
                        logger,
                        level="warning",
                        event="execution_rate_limited",
                        message="Execution skipped due to Jito rate limiting",
                        rate_limited_count=execution_rate_limited_count,
                        consecutive_errors=consecutive_execution_errors,
                        retry_after_seconds=retry_after_seconds if retry_after_seconds > 0 else None,
                        adaptive_backoff_seconds=round(adaptive_backoff_seconds, 3),
                        applied_backoff_seconds=round(transient_backoff_seconds, 3),
                        reason=result.reason,
                        fail_reason=fail_reason,
                    )
                    await emit_runtime_counters(
                        {
                            "stageA_execute_fail_count": 1,
                            "rate_limited_count": 1,
                            f"stageA_execute_fail_reason:{fail_reason}": 1,
                        }
                    )
                    await guarded_call(
                        storage.update_heartbeat,
                        logger=logger,
                        event="execution_rate_limited_heartbeat_failed",
                        message="Failed to update heartbeat during execution rate-limit backoff",
                    )
                else:
                    execution_rate_limited_count = 0

                if result.status == "failed":
                    consecutive_execution_errors += 1
                    if (
                        not app_settings.dry_run
                        and consecutive_execution_errors >= app_settings.live_max_consecutive_execution_errors
                    ):
                        await open_execution_circuit(
                            reason="execution_result_failed",
                            error=result.reason,
                        )
                else:
                    consecutive_execution_errors = 0

                result_fail_reason = normalize_fail_reason(
                    fail_reason_from_execution_result(result.status, result.reason)
                )
                if result.status in {"filled", "dry_run"}:
                    await emit_runtime_counters({"stageA_execute_success_count": 1})
                elif result.status != "rate_limited" and result_fail_reason:
                    await emit_runtime_counters(
                        {
                            "stageA_execute_fail_count": 1,
                            f"stageA_execute_fail_reason:{result_fail_reason}": 1,
                        }
                    )

                is_non_execution_result = (
                    result.status.startswith("skipped_")
                    or result.status in {"rate_limited", "pending_confirmation", "not_landed"}
                )
                realized_pnl_lamports = to_float(result.metadata.get("pnl_delta"), 0.0)
                if (
                    not app_settings.dry_run
                    and realized_pnl_lamports == 0.0
                    and not is_non_execution_result
                    and estimated_fee_lamports > 0
                ):
                    realized_pnl_lamports = -float(estimated_fee_lamports)
                result_metadata, metrics_source, plan_expected_net_bps, requote_decay_bps = (
                    prepare_execution_result_metadata(
                        result_metadata=result.metadata if isinstance(result.metadata, dict) else None,
                        initial_expected_net_bps=decision.expected_net_bps_single,
                    )
                )
                if result_fail_reason:
                    result_metadata.setdefault("fail_reason", result_fail_reason)
                result_metadata.setdefault("stageA_pass", stage_a_pass)
                result_metadata.setdefault("stageB_pass", stage_b_pass)
                result_metadata.setdefault("stageA_ok", stage_a_ok)
                result_metadata.setdefault("stageB_ok", stage_b_ok)
                result_metadata.setdefault("allow_stageb_fail_probe", allow_stageb_fail_probe)
                result_metadata.setdefault("is_probe_trade", is_probe_trade)
                result_metadata.setdefault("probe_limit_fail_reason", probe_limit_fail_reason or "")
                result_metadata.setdefault(
                    "probe_limits",
                    {
                        "max_neg_net_bps": PROBE_MAX_NEG_NET_BPS,
                        "max_loss_lamports": PROBE_MAX_LOSS_LAMPORTS,
                        "max_base_amount_lamports": PROBE_MAX_BASE_AMOUNT_LAMPORTS,
                    },
                )
                result_metadata.setdefault("initial_fail_reason", initial_fail_reason)

                initial_spread_bps = (
                    float(observation.best_spread_pre_requote)
                    if observation.best_spread_pre_requote is not None
                    else float(decision.spread_bps)
                )
                requote_median_spread_bps = (
                    float(observation.requote_median_spread_bps)
                    if observation.requote_median_spread_bps is not None
                    else float(decision.spread_bps)
                )
                plan_spread_bps = to_float(
                    result_metadata.get("plan_spread_bps")
                    or result_metadata.get("observed_spread_bps"),
                    None,
                )
                decay_initial_to_requote_bps = requote_median_spread_bps - initial_spread_bps
                decay_requote_to_plan_bps = (
                    (float(plan_spread_bps) - requote_median_spread_bps)
                    if plan_spread_bps is not None
                    else None
                )
                decay_initial_to_plan_bps = (
                    (float(plan_spread_bps) - initial_spread_bps)
                    if plan_spread_bps is not None
                    else None
                )

                initial_forward_route_hash = str(observation.forward_route_hash or "").strip()
                initial_reverse_route_hash = str(observation.reverse_route_hash or "").strip()
                plan_forward_route_hash = str(
                    result_metadata.get("plan_forward_route_hash")
                    or result_metadata.get("forward_route_hash")
                    or ""
                ).strip()
                plan_reverse_route_hash = str(
                    result_metadata.get("plan_reverse_route_hash")
                    or result_metadata.get("reverse_route_hash")
                    or ""
                ).strip()
                route_hash_match_forward = (
                    bool(initial_forward_route_hash and plan_forward_route_hash)
                    and initial_forward_route_hash == plan_forward_route_hash
                )
                route_hash_match_reverse = (
                    bool(initial_reverse_route_hash and plan_reverse_route_hash)
                    and initial_reverse_route_hash == plan_reverse_route_hash
                )

                result_metadata.setdefault("initial_spread_bps", initial_spread_bps)
                result_metadata.setdefault("requote_median_spread_bps", requote_median_spread_bps)
                result_metadata.setdefault("plan_spread_bps", plan_spread_bps)
                result_metadata.setdefault("decay_initial_to_requote_bps", decay_initial_to_requote_bps)
                result_metadata.setdefault("decay_requote_to_plan_bps", decay_requote_to_plan_bps)
                result_metadata.setdefault("decay_initial_to_plan_bps", decay_initial_to_plan_bps)
                result_metadata.setdefault("forward_route_hash", initial_forward_route_hash)
                result_metadata.setdefault("reverse_route_hash", initial_reverse_route_hash)
                result_metadata.setdefault("plan_forward_route_hash", plan_forward_route_hash)
                result_metadata.setdefault("plan_reverse_route_hash", plan_reverse_route_hash)
                result_metadata.setdefault("route_hash_match_forward", route_hash_match_forward)
                result_metadata.setdefault("route_hash_match_reverse", route_hash_match_reverse)

                if runtime_config.enable_decay_metrics_logging:
                    log_event(
                        logger,
                        level="info",
                        event="spread_decay_breakdown",
                        message="Spread decay breakdown across initial -> requote -> plan",
                        pair=pair.symbol,
                        plan_id=result_metadata.get("plan_id"),
                        initial_spread_bps=round(initial_spread_bps, 6),
                        requote_median_spread_bps=round(requote_median_spread_bps, 6),
                        plan_spread_bps=(
                            round(float(plan_spread_bps), 6)
                            if plan_spread_bps is not None
                            else None
                        ),
                        decay_initial_to_requote_bps=round(decay_initial_to_requote_bps, 6),
                        decay_requote_to_plan_bps=(
                            round(float(decay_requote_to_plan_bps), 6)
                            if decay_requote_to_plan_bps is not None
                            else None
                        ),
                        decay_initial_to_plan_bps=(
                            round(float(decay_initial_to_plan_bps), 6)
                            if decay_initial_to_plan_bps is not None
                            else None
                        ),
                        forward_route_hash=initial_forward_route_hash,
                        reverse_route_hash=initial_reverse_route_hash,
                        plan_forward_route_hash=plan_forward_route_hash,
                        plan_reverse_route_hash=plan_reverse_route_hash,
                        route_hash_match_forward=route_hash_match_forward,
                        route_hash_match_reverse=route_hash_match_reverse,
                    )

                if (
                    runtime_config.enable_route_instability_cooldown
                    and not app_settings.dry_run
                    and decay_requote_to_plan_bps is not None
                    and decay_requote_to_plan_bps <= -runtime_config.route_instability_cooldown_decay_plan_bps
                ):
                    async def arm_plan_route_cooldown() -> None:
                        trader_engine.watcher.register_route_instability_cooldown(
                            forward_route_hash=plan_forward_route_hash or initial_forward_route_hash,
                            reverse_route_hash=plan_reverse_route_hash or initial_reverse_route_hash,
                            decay_bps=float(decay_requote_to_plan_bps),
                            cooldown_seconds=runtime_config.route_instability_cooldown_plan_seconds,
                            source="plan_decay",
                            plan_id=str(result_metadata.get("plan_id") or ""),
                        )

                    await guarded_call(
                        arm_plan_route_cooldown,
                        logger=logger,
                        event="route_instability_cooldown_plan_arming_failed",
                        message="Failed to arm route instability cooldown from plan decay",
                        level="warning",
                    )

                if plan_expected_net_bps is not None:
                    if (
                        requote_decay_bps is not None
                        and requote_decay_bps <= -app_settings.live_requote_decay_warn_bps
                    ):
                        log_event(
                            logger,
                            level="warning",
                            event="requote_decay_detected",
                            message="Re-quote edge degraded compared with initial decision",
                            pair=pair.symbol,
                            initial_expected_net_bps=round(decision.expected_net_bps_single, 6),
                            plan_expected_net_bps=round(plan_expected_net_bps, 6),
                            requote_decay_bps=round(requote_decay_bps, 6),
                            initial_spread_bps=round(initial_spread_bps, 6),
                            requote_median_spread_bps=round(requote_median_spread_bps, 6),
                            plan_spread_bps=(
                                round(float(plan_spread_bps), 6)
                                if plan_spread_bps is not None
                                else None
                            ),
                            decay_initial_to_requote_bps=round(decay_initial_to_requote_bps, 6),
                            decay_requote_to_plan_bps=(
                                round(float(decay_requote_to_plan_bps), 6)
                                if decay_requote_to_plan_bps is not None
                                else None
                            ),
                            decay_initial_to_plan_bps=(
                                round(float(decay_initial_to_plan_bps), 6)
                                if decay_initial_to_plan_bps is not None
                                else None
                            ),
                            route_hash_match_forward=route_hash_match_forward,
                            route_hash_match_reverse=route_hash_match_reverse,
                            plan_id=result_metadata.get("plan_id"),
                        )
                        if (
                            not app_settings.dry_run
                            and app_settings.live_requote_decay_guard_trigger_bps > 0
                            and requote_decay_bps <= -app_settings.live_requote_decay_guard_trigger_bps
                        ):
                            guard_pause_seconds = app_settings.live_requote_decay_guard_pause_seconds
                            if guard_pause_seconds > 0:
                                requote_decay_guard_until = max(
                                    requote_decay_guard_until,
                                    loop.time() + guard_pause_seconds,
                                )
                            requote_decay_required_net_bps_boost = max(
                                requote_decay_required_net_bps_boost,
                                app_settings.live_requote_decay_guard_required_net_bps_boost,
                            )
                            log_event(
                                logger,
                                level="warning",
                                event="requote_decay_guard_armed",
                                message="Requote-decay guard was armed due to severe execution edge decay",
                                pair=pair.symbol,
                                plan_id=result_metadata.get("plan_id"),
                                requote_decay_bps=round(requote_decay_bps, 6),
                                trigger_bps=round(app_settings.live_requote_decay_guard_trigger_bps, 6),
                                pause_seconds=round(guard_pause_seconds, 3),
                                required_net_bps_boost=round(
                                    requote_decay_required_net_bps_boost,
                                    6,
                                ),
                                guard_until_epoch=round(requote_decay_guard_until, 3)
                                if requote_decay_guard_until > 0
                                else None,
                            )
                await storage.record_position(
                    {
                        "pair": pair.symbol,
                        "status": result.status,
                        "reason": result.reason,
                        "fail_reason": result_fail_reason,
                        "spread_bps": decision.spread_bps,
                        "required_spread_bps": decision.required_spread_bps,
                        "order_id": result.order_id,
                        "idempotency_key": result.idempotency_key,
                    }
                )
                await storage.record_trade(
                    trade={
                        "order_id": result.order_id,
                        "idempotency_key": result.idempotency_key,
                        "pair": pair.symbol,
                        "status": result.status,
                        "reason": result.reason,
                        "fail_reason": result_fail_reason,
                        "tx_signature": result.tx_signature,
                        "spread_bps": decision.spread_bps,
                        "required_spread_bps": decision.required_spread_bps,
                        "total_fee_bps": decision.total_fee_bps,
                        "priority_fee_micro_lamports": result.priority_fee_micro_lamports,
                        "config_schema_version": runtime_config.config_schema_version,
                        "metrics_source": metrics_source,
                        "pnl_delta": realized_pnl_lamports,
                        "allow_stageb_fail_probe": allow_stageb_fail_probe,
                        "stagea_ok": stage_a_ok,
                        "stageb_ok": stage_b_ok,
                        "is_probe_trade": is_probe_trade,
                        "probe_limit_fail_reason": probe_limit_fail_reason or "",
                        "probe_limits": {
                            "max_neg_net_bps": PROBE_MAX_NEG_NET_BPS,
                            "max_loss_lamports": PROBE_MAX_LOSS_LAMPORTS,
                            "max_base_amount_lamports": PROBE_MAX_BASE_AMOUNT_LAMPORTS,
                        },
                        "metadata": result_metadata,
                    },
                    trade_id=result.order_id,
                )
                if storage.settings.firestore_publish_order_execution_events:
                    await storage.publish_event(
                        level="INFO",
                        event="order_execution",
                        message="Order execution attempted",
                        details={
                            "pair": pair.symbol,
                            "decision_reason": decision.reason,
                            "execution": result.to_dict(),
                            "priority_fee_plan": priority_fee_plan.to_dict(),
                        },
                        event_id=f"order_execution:{result.order_id}",
                    )
                if not is_non_execution_result:
                    execution_time = loop.time()
                    recent_execution_timestamps.append(execution_time)
                    last_execution_at = execution_time
                    if not app_settings.dry_run and estimated_fee_lamports > 0:
                        recent_execution_fees.append((execution_time, estimated_fee_lamports))
                        recent_execution_fee_total += estimated_fee_lamports
                    if not app_settings.dry_run and realized_pnl_lamports != 0.0:
                        record_realized_pnl(
                            now_time=execution_time,
                            pnl_delta_lamports=realized_pnl_lamports,
                        )
                        current_drawdown_lamports = refresh_recent_pnl(execution_time)
                        if (
                            app_settings.live_max_drawdown_lamports > 0
                            and current_drawdown_lamports >= app_settings.live_max_drawdown_lamports
                        ):
                            await open_drawdown_circuit(
                                reason="post_execution_drawdown_limit",
                                current_drawdown_lamports=current_drawdown_lamports,
                            )

            if not should_attempt_execution:
                if not runtime_config.trade_enabled:
                    preflight_skip_reason = "trade_disabled"
                elif decision.blocked_by_fee_cap:
                    preflight_skip_reason = "blocked_by_fee_cap"
                elif not initial_gate_passed:
                    preflight_skip_reason = initial_fail_reason or "initial_gate_filtered"
                else:
                    preflight_skip_reason = probe_limit_fail_reason or initial_fail_reason or "not_executable"
                emit_execution_preflight_summary(
                    runtime_config=runtime_config,
                    initial_gate_passed=initial_gate_passed,
                    stage_a_pass=stage_a_pass,
                    stage_b_pass=stage_b_pass,
                    stage_a_ok=stage_a_ok,
                    stage_b_ok=stage_b_ok,
                    is_probe_trade=is_probe_trade,
                    expected_net_bps_stage_a=expected_net_bps_stage_a,
                    expected_net_bps_stage_b=expected_net_bps_stage_b,
                    expected_net_lamports_single=decision.expected_net_lamports_single,
                    expected_net_bps=decision.expected_net_bps_single,
                    expected_net_lamports=decision.expected_net_lamports_single,
                    final_skip_reason=preflight_skip_reason,
                    rate_limit_pause_until_epoch=rate_limit_pause_until_epoch,
                )

            if decision.blocked_by_fee_cap and decision.profitable:
                log_event(
                    logger,
                    level="info",
                    event="max_fee_guard_triggered",
                    message="Opportunity skipped by max_fee guard",
                    pair=pair.symbol,
                    spread_bps=decision.spread_bps,
                    required_spread_bps=decision.required_spread_bps,
                    priority_fee_recommended=priority_fee_plan.recommended_micro_lamports,
                    max_fee=priority_fee_plan.max_fee_micro_lamports,
                )

            if (
                runtime_config.trade_enabled
                and not decision.blocked_by_fee_cap
                and not initial_gate_passed
            ):
                log_event(
                    logger,
                    level="info",
                    event="initial_gate_filtered",
                    message="Opportunity skipped by initial requote prefilter",
                    pair=pair.symbol,
                    initial_expected_net_bps=round(decision.expected_net_bps_single, 6),
                    initial_expected_net_lamports=int(decision.expected_net_lamports_single),
                    required_initial_gate_bps=round(initial_requote_gate_bps, 6),
                    required_initial_profit_lamports=required_initial_profit_lamports,
                    initial_profit_gate_relaxed_enabled=runtime_config.enable_initial_profit_gate_relaxed,
                    initial_profit_gate_relaxed_lamports=int(runtime_config.initial_profit_gate_relaxed_lamports),
                    gate_basis="single",
                    stageA_gate_mode=stage_a_gate_mode,
                    stageA_pass=stage_a_pass,
                    stageA_ok=stage_a_ok,
                    stageA_margin_bps=(
                        round(float(stage_a_margin_bps), 6)
                        if stage_a_margin_bps is not None
                        else None
                    ),
                    stageB_pass=stage_b_pass,
                    stageB_ok=stage_b_ok,
                    stageB_margin_bps=(
                        round(float(stage_b_margin_bps), 6)
                        if stage_b_margin_bps is not None
                        else None
                    ),
                    fail_reason=initial_fail_reason,
                    allow_stageb_fail_probe=allow_stageb_fail_probe,
                    is_probe_trade=is_probe_trade,
                    probe_limit_fail_reason=probe_limit_fail_reason or "",
                    probe_limits={
                        "max_neg_net_bps": PROBE_MAX_NEG_NET_BPS,
                        "max_loss_lamports": PROBE_MAX_LOSS_LAMPORTS,
                        "max_base_amount_lamports": PROBE_MAX_BASE_AMOUNT_LAMPORTS,
                    },
                    exploration_mode=observation.exploration_mode,
                )

            if decision.profitable and not runtime_config.trade_enabled and not decision.blocked_by_fee_cap:
                log_event(
                    logger,
                    level="info",
                    event="opportunity_detected",
                    message="Opportunity detected while trade is disabled",
                    pair=pair.symbol,
                    spread_bps=decision.spread_bps,
                    required_spread_bps=decision.required_spread_bps,
                )

        except HeliusRateLimitError as error:
            retry_after_seconds = max(0.0, float(error.retry_after_seconds or 0.0))
            watcher_pause_until_epoch = 0.0
            watcher_pause_getter = getattr(trader_engine.watcher, "rate_limit_pause_until_epoch", None)
            if callable(watcher_pause_getter):
                with contextlib.suppress(Exception):
                    watcher_pause_until_epoch = max(0.0, float(watcher_pause_getter()))
            pause_until_epoch = max(
                watcher_pause_until_epoch,
                time.time() + max(60.0, retry_after_seconds),
            )
            pause_remaining_seconds = max(0.0, pause_until_epoch - time.time())
            await guarded_call(
                lambda: storage.set_rate_limit_pause_until(
                    pause_until_epoch=pause_until_epoch,
                ),
                logger=logger,
                event="rate_limit_pause_write_failed",
                message="Failed to persist quote rate-limit pause window to Redis",
                level="warning",
            )
            transient_backoff_seconds = max(
                app_settings.error_backoff_seconds,
                pause_remaining_seconds,
            )
            provider = getattr(error, "provider", "quote")
            await emit_runtime_counters(
                {
                    "rate_limited_count": 1,
                    f"fail_reason:{FAIL_REASON_RATE_LIMITED}": 1,
                }
            )
            log_event(
                logger,
                level="warning",
                event=f"{provider}_rate_limited",
                message=f"{provider.capitalize()} quote rate-limited; keeping order intake active and backing off",
                error=str(error),
                retry_after_seconds=round(retry_after_seconds, 3) if retry_after_seconds > 0 else None,
                pause_until_epoch=round(pause_until_epoch, 3),
                backoff_seconds=round(transient_backoff_seconds, 3),
            )
            emit_execution_preflight_summary(
                runtime_config=runtime_config,
                initial_gate_passed=False,
                guard_block_reason="rate_limited_pause_active",
                final_skip_reason="rate_limited_pause_active",
                rate_limit_pause_until_epoch=pause_until_epoch,
            )
            await guarded_call(
                storage.update_heartbeat,
                logger=logger,
                event="helius_rate_limited_heartbeat_failed",
                message="Failed to update heartbeat during rate-limit backoff",
            )
        except Exception as error:
            pause_reason = str(error)
            order_intake_paused = True
            await emit_runtime_counters({f"fail_reason:{FAIL_REASON_OTHER}": 1})

            log_event(
                logger,
                level="exception",
                event="main_loop_error",
                message="Main loop failed and order intake has been paused",
                error=str(error),
            )
            await guarded_call(
                lambda: storage.record_position(
                    {
                        "pair": pair.symbol,
                        "status": "order_intake_paused",
                        "reason": pause_reason,
                    }
                ),
                logger=logger,
                event="main_loop_pause_position_record_failed",
                message="Failed to record paused position state",
            )
            await guarded_call(
                lambda: storage.publish_event(
                    level="ERROR",
                    event="order_intake_paused",
                    message="Order intake paused due to dependency or runtime error",
                    details={"error": str(error), "pair": pair.symbol},
                ),
                logger=logger,
                event="main_loop_pause_publish_failed",
                message="Failed to publish order_intake_paused event",
            )
        finally:
            next_tick += app_settings.watch_interval_seconds
            now = loop.time()
            if next_tick <= now:
                missed_cycles = int((now - next_tick) / app_settings.watch_interval_seconds) + 1
                next_tick += missed_cycles * app_settings.watch_interval_seconds

            delay_seconds = max(0.0, next_tick - now)
            if order_intake_paused:
                delay_seconds = max(delay_seconds, app_settings.error_backoff_seconds)
            if transient_backoff_seconds > 0:
                delay_seconds = max(delay_seconds, transient_backoff_seconds)

            await wait_with_stop(stop_event, delay_seconds)


__all__ = [
    "bootstrap_dependencies",
    "run_trading_loop",
]
