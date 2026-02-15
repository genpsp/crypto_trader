from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


def to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def to_int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def parse_string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in result:
                result.append(text)
        return tuple(result)

    raw = str(value).strip()
    if not raw:
        return ()

    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
            return parse_string_list(parsed)
        except json.JSONDecodeError:
            pass

    result: list[str] = []
    for part in raw.split(","):
        text = part.strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def parse_float_list(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        result: list[float] = []
        for item in value:
            try:
                parsed = float(str(item).strip())
            except ValueError:
                continue
            if parsed > 0:
                result.append(parsed)
        return tuple(result)

    raw = str(value).strip()
    if not raw:
        return ()

    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
            return parse_float_list(parsed)
        except json.JSONDecodeError:
            pass

    result: list[float] = []
    for part in raw.split(","):
        try:
            parsed = float(part.strip())
        except ValueError:
            continue
        if parsed > 0:
            result.append(parsed)
    return tuple(result)


def parse_int_list(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        result: list[int] = []
        for item in value:
            parsed = to_int(item, 0)
            if parsed > 0 and parsed not in result:
                result.append(parsed)
        return tuple(result)

    raw = str(value).strip()
    if not raw:
        return ()

    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
            return parse_int_list(parsed)
        except json.JSONDecodeError:
            pass

    result: list[int] = []
    for part in raw.split(","):
        parsed = to_int(part.strip(), 0)
        if parsed > 0 and parsed not in result:
            result.append(parsed)
    return tuple(result)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp_percentile(value: float) -> float:
    return max(0.0, min(1.0, value))


def percentile_value(values: list[int], percentile: float) -> int:
    if not values:
        return 0

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    p = clamp_percentile(percentile)
    index = max(0, min(len(sorted_values) - 1, math.ceil(len(sorted_values) * p) - 1))
    return sorted_values[index]


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


def normalize_quote_exploration_mode(value: str) -> str:
    mode = (value or "").strip().lower()
    if mode in {"tier1_only", "tier1"}:
        return "TIER1_ONLY"
    return "MIXED"


def make_order_id(idempotency_key: str, blockhash: str | None = None) -> str:
    tail = blockhash[:16] if blockhash else datetime.now(timezone.utc).strftime("%H%M%S%f")
    return f"ord-{idempotency_key[:18]}-{tail}"


FAIL_REASON_BELOW_STAGEA_REQUIRED = "BELOW_STAGEA_REQUIRED"
FAIL_REASON_BELOW_STAGEB_REQUIRED = "BELOW_STAGEB_REQUIRED"
FAIL_REASON_ROUTE_UNSTABLE_REQUOTE_DROPPED = "ROUTE_UNSTABLE_REQUOTE_DROPPED"
FAIL_REASON_RATE_LIMITED = "RATE_LIMITED"
FAIL_REASON_SIMULATION_FAIL = "SIMULATION_FAIL"
FAIL_REASON_TX_FAIL = "TX_FAIL"
FAIL_REASON_NOT_LANDED = "NOT_LANDED"
FAIL_REASON_PROBE_LIMIT_NEG_NET = "PROBE_LIMIT_NEG_NET"
FAIL_REASON_PROBE_LIMIT_LOSS_LAMPORTS = "PROBE_LIMIT_LOSS_LAMPORTS"
FAIL_REASON_PROBE_LIMIT_MAX_BASE_AMOUNT = "PROBE_LIMIT_MAX_BASE_AMOUNT"
FAIL_REASON_SLIPPAGE_EXCEEDED = "SLIPPAGE_EXCEEDED"
FAIL_REASON_ACCOUNT_LIMIT = "ACCOUNT_LIMIT"
FAIL_REASON_OTHER = "OTHER"


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
            base_decimals=to_int(os.getenv("PAIR_BASE_DECIMALS"), 9),
            quote_decimals=to_int(os.getenv("PAIR_QUOTE_DECIMALS"), 6),
            base_amount=to_int(os.getenv("PAIR_BASE_AMOUNT"), 1_000_000_000),
            slippage_bps=to_int(os.getenv("PAIR_SLIPPAGE_BPS"), 20),
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
    quote_params: dict[str, Any] = field(default_factory=dict)
    forward_route_dexes: tuple[str, ...] = ()
    reverse_route_dexes: tuple[str, ...] = ()
    forward_route_hash: str = ""
    reverse_route_hash: str = ""
    stage_a_pass: bool | None = None
    stage_a_margin_bps: float | None = None
    stage_b_pass: bool | None = None
    stage_b_margin_bps: float | None = None
    fail_reason: str = ""
    route_candidate_count: int = 0
    route_sampled_count: int = 0
    route_cooldown_skipped_count: int = 0
    exploration_mode: str = "MIXED"
    median_requote_applied: bool = False
    median_requote_sample_count: int = 0
    median_requote_range_bps: float | None = None
    requote_spread_samples_bps: tuple[float, ...] = ()
    requote_samples_route_hashes_forward: tuple[str, ...] = ()
    requote_samples_route_hashes_reverse: tuple[str, ...] = ()
    requote_median_spread_bps: float | None = None
    best_spread_pre_requote: float | None = None
    best_spread_post_requote: float | None = None
    unstable_drop_threshold_bps: float | None = None
    is_probe_trade: bool = False


@dataclass(slots=True, frozen=True)
class RuntimeConfig:
    config_schema_version: int
    min_spread_bps: float
    initial_min_spread_bps: float
    dex_fee_bps: float
    priority_fee_micro_lamports: int
    priority_compute_units: int
    priority_fee_percentile: float
    priority_fee_multiplier: float
    max_fee_micro_lamports: int
    trade_enabled: bool
    order_guard_ttl_seconds: int
    execution_mode: str
    atomic_send_mode: str
    atomic_expiry_ms: int
    atomic_margin_bps: float
    initial_atomic_margin_bps: float
    min_stagea_margin_bps: float
    atomic_bundle_min_expected_net_bps: float
    jito_block_engine_url: str
    jito_tip_lamports_max: int
    jito_tip_lamports_recommended: int
    jito_tip_share: float
    quote_default_params_json: str
    quote_initial_params_json: str
    quote_plan_params_json: str
    quote_dex_allowlist_forward: tuple[str, ...]
    quote_dex_allowlist_reverse: tuple[str, ...]
    quote_dex_excludelist: tuple[str, ...]
    quote_exploration_mode: str
    quote_dex_sweep_topk: int
    quote_dex_sweep_topk_max: int
    quote_dex_sweep_combo_limit: int
    quote_near_miss_expand_bps: float
    quote_median_requote_max_range_bps: float
    max_requote_range_bps: float
    min_improvement_bps: float
    quote_max_rps: float
    quote_exploration_max_rps: float
    quote_execution_max_rps: float
    quote_cache_ttl_ms: int
    quote_no_routes_cache_ttl_seconds: int
    quote_probe_max_routes: int
    quote_probe_base_amounts_raw: tuple[int, ...]
    quote_dynamic_allowlist_topk: int
    quote_dynamic_allowlist_good_candidate_alpha: float
    quote_dynamic_allowlist_ttl_seconds: int
    quote_dynamic_allowlist_refresh_seconds: float
    quote_negative_fallback_streak_threshold: int
    route_instability_cooldown_requote_seconds: float
    route_instability_cooldown_plan_seconds: float
    route_instability_cooldown_decay_requote_bps: float
    route_instability_cooldown_decay_plan_bps: float
    initial_profit_gate_relaxed_lamports: int
    enable_probe_unconstrained: bool
    enable_probe_multi_amount: bool
    enable_decay_metrics_logging: bool
    enable_priority_fee_breakdown_logging: bool
    enable_stagea_relaxed_gate: bool
    enable_initial_profit_gate_relaxed: bool
    enable_route_instability_cooldown: bool
    enable_legacy_swap_fallback: bool
    allow_stageb_fail_probe: bool
    base_amount_sweep_candidates_raw: tuple[int, ...]
    base_amount_sweep_multipliers: tuple[float, ...]
    base_amount_max_raw: int
    min_expected_profit_lamports: int

    @classmethod
    def from_env_defaults(cls) -> "RuntimeConfig":
        execution_min_spread_bps = to_float(os.getenv("MIN_SPREAD_BPS"), 5.0)
        execution_atomic_margin_bps = max(0.0, to_float(os.getenv("ATOMIC_MARGIN_BPS"), 20.0))
        quote_global_max_rps = max(0.1, to_float(os.getenv("QUOTE_MAX_RPS"), 0.4))
        quote_exploration_max_rps = max(
            0.05,
            to_float(
                os.getenv("QUOTE_EXPLORATION_MAX_RPS"),
                min(0.3, quote_global_max_rps),
            ),
        )
        quote_execution_max_rps = max(
            0.05,
            to_float(
                os.getenv("QUOTE_EXECUTION_MAX_RPS"),
                min(0.15, quote_global_max_rps),
            ),
        )
        quote_exploration_max_rps = min(quote_exploration_max_rps, quote_global_max_rps)
        quote_execution_max_rps = min(quote_execution_max_rps, quote_global_max_rps)
        max_requote_range_default = max(
            0.0,
            to_float(
                os.getenv("MAX_REQUOTE_RANGE_BPS"),
                to_float(os.getenv("QUOTE_MEDIAN_REQUOTE_MAX_RANGE_BPS"), 0.6),
            ),
        )
        return cls(
            config_schema_version=max(1, to_int(os.getenv("CONFIG_SCHEMA_VERSION"), 1)),
            min_spread_bps=execution_min_spread_bps,
            initial_min_spread_bps=max(
                0.0,
                to_float(os.getenv("INITIAL_MIN_SPREAD_BPS"), execution_min_spread_bps),
            ),
            dex_fee_bps=to_float(os.getenv("DEX_FEE_BPS"), 4.0),
            priority_fee_micro_lamports=to_int(os.getenv("PRIORITY_FEE_MICRO_LAMPORTS"), 10_000),
            priority_compute_units=to_int(os.getenv("PRIORITY_COMPUTE_UNITS"), 200_000),
            priority_fee_percentile=clamp_percentile(to_float(os.getenv("PRIORITY_FEE_PERCENTILE"), 0.75)),
            priority_fee_multiplier=max(0.0, to_float(os.getenv("PRIORITY_FEE_MULTIPLIER"), 1.15)),
            max_fee_micro_lamports=to_int(os.getenv("MAX_FEE_MICRO_LAMPORTS"), 80_000),
            trade_enabled=to_bool(os.getenv("TRADE_ENABLED"), False),
            order_guard_ttl_seconds=max(1, to_int(os.getenv("ORDER_GUARD_TTL_SECONDS"), 20)),
            execution_mode=normalize_execution_mode(os.getenv("EXECUTION_MODE", "atomic")),
            atomic_send_mode=normalize_atomic_send_mode(os.getenv("ATOMIC_SEND_MODE", "auto")),
            atomic_expiry_ms=max(250, to_int(os.getenv("ATOMIC_EXPIRY_MS"), 5_000)),
            atomic_margin_bps=execution_atomic_margin_bps,
            initial_atomic_margin_bps=max(
                0.0,
                to_float(os.getenv("INITIAL_ATOMIC_MARGIN_BPS"), execution_atomic_margin_bps),
            ),
            min_stagea_margin_bps=max(
                0.0,
                to_float(os.getenv("MIN_STAGEA_MARGIN_BPS"), 0.5),
            ),
            atomic_bundle_min_expected_net_bps=max(
                0.0,
                to_float(os.getenv("ATOMIC_BUNDLE_MIN_EXPECTED_NET_BPS"), 2.0),
            ),
            jito_block_engine_url=os.getenv("JITO_BLOCK_ENGINE_URL", "").strip(),
            jito_tip_lamports_max=max(0, to_int(os.getenv("JITO_TIP_LAMPORTS_MAX"), 100_000)),
            jito_tip_lamports_recommended=max(
                0,
                to_int(os.getenv("JITO_TIP_LAMPORTS_RECOMMENDED"), 20_000),
            ),
            jito_tip_share=max(
                0.0,
                min(1.0, to_float(os.getenv("JITO_TIP_SHARE"), 0.2)),
            ),
            quote_default_params_json=os.getenv("QUOTE_DEFAULT_PARAMS_JSON", "").strip(),
            quote_initial_params_json=os.getenv("QUOTE_INITIAL_PARAMS_JSON", "").strip(),
            quote_plan_params_json=os.getenv("QUOTE_PLAN_PARAMS_JSON", "").strip(),
            quote_dex_allowlist_forward=parse_string_list(os.getenv("QUOTE_DEX_ALLOWLIST_FORWARD", "")),
            quote_dex_allowlist_reverse=parse_string_list(os.getenv("QUOTE_DEX_ALLOWLIST_REVERSE", "")),
            quote_dex_excludelist=parse_string_list(os.getenv("QUOTE_DEX_EXCLUDELIST", "")),
            quote_exploration_mode=normalize_quote_exploration_mode(
                os.getenv("QUOTE_EXPLORATION_MODE", "mixed")
            ),
            quote_dex_sweep_topk=max(1, to_int(os.getenv("QUOTE_DEX_SWEEP_TOPK"), 3)),
            quote_dex_sweep_topk_max=max(
                1,
                to_int(os.getenv("QUOTE_DEX_SWEEP_TOPK_MAX"), 8),
            ),
            quote_dex_sweep_combo_limit=max(1, to_int(os.getenv("QUOTE_DEX_SWEEP_COMBO_LIMIT"), 9)),
            quote_near_miss_expand_bps=max(
                0.0,
                to_float(os.getenv("QUOTE_NEAR_MISS_EXPAND_BPS"), 0.5),
            ),
            quote_median_requote_max_range_bps=max(
                0.0,
                to_float(os.getenv("QUOTE_MEDIAN_REQUOTE_MAX_RANGE_BPS"), 0.6),
            ),
            max_requote_range_bps=max_requote_range_default,
            min_improvement_bps=max(
                0.0,
                to_float(
                    os.getenv("MIN_IMPROVEMENT_BPS"),
                    to_float(os.getenv("QUOTE_MIN_IMPROVEMENT_BPS"), 0.2),
                ),
            ),
            quote_max_rps=quote_global_max_rps,
            quote_exploration_max_rps=quote_exploration_max_rps,
            quote_execution_max_rps=quote_execution_max_rps,
            quote_cache_ttl_ms=max(0, to_int(os.getenv("QUOTE_CACHE_TTL_MS"), 500)),
            quote_no_routes_cache_ttl_seconds=max(
                1,
                to_int(os.getenv("QUOTE_NO_ROUTES_CACHE_TTL_SECONDS"), 120),
            ),
            quote_probe_max_routes=max(
                1,
                to_int(os.getenv("QUOTE_PROBE_MAX_ROUTES"), 50),
            ),
            quote_probe_base_amounts_raw=(
                parse_int_list(os.getenv("QUOTE_PROBE_BASE_AMOUNTS_RAW", "10000000,20000000,40000000"))
                or (10_000_000, 20_000_000, 40_000_000)
            ),
            quote_dynamic_allowlist_topk=max(
                1,
                to_int(os.getenv("QUOTE_DYNAMIC_ALLOWLIST_TOPK"), 10),
            ),
            quote_dynamic_allowlist_good_candidate_alpha=max(
                0.0,
                to_float(os.getenv("QUOTE_DYNAMIC_ALLOWLIST_GOOD_ALPHA"), 2.0),
            ),
            quote_dynamic_allowlist_ttl_seconds=max(
                1,
                to_int(os.getenv("QUOTE_DYNAMIC_ALLOWLIST_TTL_SECONDS"), 300),
            ),
            quote_dynamic_allowlist_refresh_seconds=max(
                0.5,
                to_float(os.getenv("QUOTE_DYNAMIC_ALLOWLIST_REFRESH_SECONDS"), 5.0),
            ),
            quote_negative_fallback_streak_threshold=max(
                1,
                to_int(os.getenv("QUOTE_NEGATIVE_FALLBACK_STREAK_THRESHOLD"), 10),
            ),
            route_instability_cooldown_requote_seconds=max(
                0.0,
                to_float(os.getenv("ROUTE_INSTABILITY_COOLDOWN_REQUOTE_SECONDS"), 60.0),
            ),
            route_instability_cooldown_plan_seconds=max(
                0.0,
                to_float(os.getenv("ROUTE_INSTABILITY_COOLDOWN_PLAN_SECONDS"), 120.0),
            ),
            route_instability_cooldown_decay_requote_bps=max(
                0.0,
                to_float(os.getenv("ROUTE_INSTABILITY_COOLDOWN_DECAY_REQUOTE_BPS"), 2.0),
            ),
            route_instability_cooldown_decay_plan_bps=max(
                0.0,
                to_float(os.getenv("ROUTE_INSTABILITY_COOLDOWN_DECAY_PLAN_BPS"), 2.0),
            ),
            initial_profit_gate_relaxed_lamports=max(
                0,
                to_int(os.getenv("INITIAL_PROFIT_GATE_RELAXED_LAMPORTS"), 1000),
            ),
            enable_probe_unconstrained=to_bool(os.getenv("ENABLE_PROBE_UNCONSTRAINED"), True),
            enable_probe_multi_amount=to_bool(os.getenv("ENABLE_PROBE_MULTI_AMOUNT"), True),
            enable_decay_metrics_logging=to_bool(os.getenv("ENABLE_DECAY_METRICS_LOGGING"), True),
            enable_priority_fee_breakdown_logging=to_bool(
                os.getenv("ENABLE_PRIORITY_FEE_BREAKDOWN_LOGGING"), True
            ),
            enable_stagea_relaxed_gate=to_bool(os.getenv("ENABLE_STAGEA_RELAXED_GATE"), True),
            enable_initial_profit_gate_relaxed=to_bool(
                os.getenv("ENABLE_INITIAL_PROFIT_GATE_RELAXED"), True
            ),
            enable_route_instability_cooldown=to_bool(
                os.getenv("ENABLE_ROUTE_INSTABILITY_COOLDOWN"), True
            ),
            enable_legacy_swap_fallback=to_bool(
                os.getenv("ENABLE_LEGACY_SWAP_FALLBACK"), False
            ),
            allow_stageb_fail_probe=to_bool(os.getenv("ALLOW_STAGEB_FAIL_PROBE"), False),
            base_amount_sweep_candidates_raw=(
                parse_int_list(os.getenv("BASE_AMOUNT_SWEEP_CANDIDATES_RAW", ""))
            ),
            base_amount_sweep_multipliers=(
                parse_float_list(os.getenv("BASE_AMOUNT_SWEEP_MULTIPLIERS", "1,3,10"))
                or (1.0, 3.0, 10.0)
            ),
            base_amount_max_raw=max(0, to_int(os.getenv("BASE_AMOUNT_MAX_RAW"), 0)),
            min_expected_profit_lamports=max(0, to_int(os.getenv("MIN_EXPECTED_PROFIT_LAMPORTS"), 0)),
        )

    @classmethod
    def from_redis(cls, redis_config: dict[str, str], defaults: "RuntimeConfig") -> "RuntimeConfig":
        schema_raw = redis_config.get("schema_version") or redis_config.get("config_schema_version")
        max_fee_raw = redis_config.get("max_fee_micro_lamports") or redis_config.get("max_fee")

        def list_from_redis(key: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
            if key in redis_config:
                return parse_string_list(redis_config.get(key))
            return fallback

        def float_list_from_redis(key: str, fallback: tuple[float, ...]) -> tuple[float, ...]:
            if key in redis_config:
                parsed = parse_float_list(redis_config.get(key))
                return parsed or fallback
            return fallback

        def int_list_from_redis(key: str, fallback: tuple[int, ...]) -> tuple[int, ...]:
            if key in redis_config:
                parsed = parse_int_list(redis_config.get(key))
                return parsed or fallback
            return fallback

        return cls(
            config_schema_version=max(1, to_int(schema_raw, defaults.config_schema_version)),
            min_spread_bps=to_float(
                redis_config.get("min_spread_bps") or redis_config.get("min_spread"),
                defaults.min_spread_bps,
            ),
            initial_min_spread_bps=max(
                0.0,
                to_float(
                    redis_config.get("initial_min_spread_bps"),
                    defaults.initial_min_spread_bps,
                ),
            ),
            dex_fee_bps=to_float(redis_config.get("dex_fee_bps"), defaults.dex_fee_bps),
            priority_fee_micro_lamports=to_int(
                redis_config.get("priority_fee_micro_lamports"),
                defaults.priority_fee_micro_lamports,
            ),
            priority_compute_units=to_int(
                redis_config.get("priority_compute_units"),
                defaults.priority_compute_units,
            ),
            priority_fee_percentile=clamp_percentile(
                to_float(redis_config.get("priority_fee_percentile"), defaults.priority_fee_percentile)
            ),
            priority_fee_multiplier=max(
                0.0,
                to_float(redis_config.get("priority_fee_multiplier"), defaults.priority_fee_multiplier),
            ),
            max_fee_micro_lamports=to_int(max_fee_raw, defaults.max_fee_micro_lamports),
            trade_enabled=to_bool(redis_config.get("trade_enabled"), defaults.trade_enabled),
            order_guard_ttl_seconds=max(
                1,
                to_int(redis_config.get("order_guard_ttl_seconds"), defaults.order_guard_ttl_seconds),
            ),
            execution_mode=normalize_execution_mode(
                redis_config.get("execution_mode") or defaults.execution_mode
            ),
            atomic_send_mode=normalize_atomic_send_mode(
                redis_config.get("atomic_send_mode") or defaults.atomic_send_mode
            ),
            atomic_expiry_ms=max(
                250,
                to_int(redis_config.get("atomic_expiry_ms"), defaults.atomic_expiry_ms),
            ),
            atomic_margin_bps=max(
                0.0,
                to_float(redis_config.get("atomic_margin_bps"), defaults.atomic_margin_bps),
            ),
            initial_atomic_margin_bps=max(
                0.0,
                to_float(
                    redis_config.get("initial_atomic_margin_bps"),
                    defaults.initial_atomic_margin_bps,
                ),
            ),
            min_stagea_margin_bps=max(
                0.0,
                to_float(
                    redis_config.get("min_stagea_margin_bps"),
                    defaults.min_stagea_margin_bps,
                ),
            ),
            atomic_bundle_min_expected_net_bps=max(
                0.0,
                to_float(
                    redis_config.get("atomic_bundle_min_expected_net_bps"),
                    defaults.atomic_bundle_min_expected_net_bps,
                ),
            ),
            jito_block_engine_url=(
                redis_config.get("jito_block_engine_url")
                or redis_config.get("JITO_BLOCK_ENGINE_URL")
                or defaults.jito_block_engine_url
            ).strip(),
            jito_tip_lamports_max=max(
                0,
                to_int(redis_config.get("jito_tip_lamports_max"), defaults.jito_tip_lamports_max),
            ),
            jito_tip_lamports_recommended=max(
                0,
                to_int(
                    redis_config.get("jito_tip_lamports_recommended"),
                    defaults.jito_tip_lamports_recommended,
                ),
            ),
            jito_tip_share=max(
                0.0,
                min(
                    1.0,
                    to_float(
                        redis_config.get("jito_tip_share"),
                        defaults.jito_tip_share,
                    ),
                ),
            ),
            quote_default_params_json=(
                redis_config.get("quote_default_params_json")
                or defaults.quote_default_params_json
            ).strip(),
            quote_initial_params_json=(
                redis_config.get("quote_initial_params_json")
                or defaults.quote_initial_params_json
            ).strip(),
            quote_plan_params_json=(
                redis_config.get("quote_plan_params_json")
                or defaults.quote_plan_params_json
            ).strip(),
            quote_dex_allowlist_forward=list_from_redis(
                "quote_dex_allowlist_forward",
                defaults.quote_dex_allowlist_forward,
            ),
            quote_dex_allowlist_reverse=list_from_redis(
                "quote_dex_allowlist_reverse",
                defaults.quote_dex_allowlist_reverse,
            ),
            quote_dex_excludelist=list_from_redis(
                "quote_dex_excludelist",
                defaults.quote_dex_excludelist,
            ),
            quote_exploration_mode=normalize_quote_exploration_mode(
                redis_config.get("quote_exploration_mode")
                or defaults.quote_exploration_mode
            ),
            quote_dex_sweep_topk=max(
                1,
                to_int(redis_config.get("quote_dex_sweep_topk"), defaults.quote_dex_sweep_topk),
            ),
            quote_dex_sweep_topk_max=max(
                1,
                to_int(
                    redis_config.get("quote_dex_sweep_topk_max"),
                    defaults.quote_dex_sweep_topk_max,
                ),
            ),
            quote_dex_sweep_combo_limit=max(
                1,
                to_int(
                    redis_config.get("quote_dex_sweep_combo_limit"),
                    defaults.quote_dex_sweep_combo_limit,
                ),
            ),
            quote_near_miss_expand_bps=max(
                0.0,
                to_float(
                    redis_config.get("quote_near_miss_expand_bps"),
                    defaults.quote_near_miss_expand_bps,
                ),
            ),
            quote_median_requote_max_range_bps=max(
                0.0,
                to_float(
                    redis_config.get("quote_median_requote_max_range_bps"),
                    defaults.quote_median_requote_max_range_bps,
                ),
            ),
            max_requote_range_bps=max(
                0.0,
                to_float(
                    redis_config.get("max_requote_range_bps")
                    or redis_config.get("quote_median_requote_max_range_bps"),
                    defaults.max_requote_range_bps,
                ),
            ),
            min_improvement_bps=max(
                0.0,
                to_float(
                    redis_config.get("min_improvement_bps")
                    or redis_config.get("quote_min_improvement_bps"),
                    defaults.min_improvement_bps,
                ),
            ),
            quote_max_rps=max(
                0.1,
                to_float(redis_config.get("quote_max_rps"), defaults.quote_max_rps),
            ),
            quote_exploration_max_rps=max(
                0.05,
                min(
                    to_float(
                        redis_config.get("quote_exploration_max_rps"),
                        defaults.quote_exploration_max_rps,
                    ),
                    to_float(redis_config.get("quote_max_rps"), defaults.quote_max_rps),
                ),
            ),
            quote_execution_max_rps=max(
                0.05,
                min(
                    to_float(
                        redis_config.get("quote_execution_max_rps"),
                        defaults.quote_execution_max_rps,
                    ),
                    to_float(redis_config.get("quote_max_rps"), defaults.quote_max_rps),
                ),
            ),
            quote_cache_ttl_ms=max(
                0,
                to_int(redis_config.get("quote_cache_ttl_ms"), defaults.quote_cache_ttl_ms),
            ),
            quote_no_routes_cache_ttl_seconds=max(
                1,
                to_int(
                    redis_config.get("quote_no_routes_cache_ttl_seconds"),
                    defaults.quote_no_routes_cache_ttl_seconds,
                ),
            ),
            quote_probe_max_routes=max(
                1,
                to_int(
                    redis_config.get("quote_probe_max_routes"),
                    defaults.quote_probe_max_routes,
                ),
            ),
            quote_probe_base_amounts_raw=int_list_from_redis(
                "quote_probe_base_amounts_raw",
                defaults.quote_probe_base_amounts_raw,
            ),
            quote_dynamic_allowlist_topk=max(
                1,
                to_int(
                    redis_config.get("quote_dynamic_allowlist_topk"),
                    defaults.quote_dynamic_allowlist_topk,
                ),
            ),
            quote_dynamic_allowlist_good_candidate_alpha=max(
                0.0,
                to_float(
                    redis_config.get("quote_dynamic_allowlist_good_candidate_alpha")
                    or redis_config.get("quote_dynamic_allowlist_good_alpha"),
                    defaults.quote_dynamic_allowlist_good_candidate_alpha,
                ),
            ),
            quote_dynamic_allowlist_ttl_seconds=max(
                1,
                to_int(
                    redis_config.get("quote_dynamic_allowlist_ttl_seconds"),
                    defaults.quote_dynamic_allowlist_ttl_seconds,
                ),
            ),
            quote_dynamic_allowlist_refresh_seconds=max(
                0.5,
                to_float(
                    redis_config.get("quote_dynamic_allowlist_refresh_seconds"),
                    defaults.quote_dynamic_allowlist_refresh_seconds,
                ),
            ),
            quote_negative_fallback_streak_threshold=max(
                1,
                to_int(
                    redis_config.get("quote_negative_fallback_streak_threshold"),
                    defaults.quote_negative_fallback_streak_threshold,
                ),
            ),
            route_instability_cooldown_requote_seconds=max(
                0.0,
                to_float(
                    redis_config.get("route_instability_cooldown_requote_seconds"),
                    defaults.route_instability_cooldown_requote_seconds,
                ),
            ),
            route_instability_cooldown_plan_seconds=max(
                0.0,
                to_float(
                    redis_config.get("route_instability_cooldown_plan_seconds"),
                    defaults.route_instability_cooldown_plan_seconds,
                ),
            ),
            route_instability_cooldown_decay_requote_bps=max(
                0.0,
                to_float(
                    redis_config.get("route_instability_cooldown_decay_requote_bps"),
                    defaults.route_instability_cooldown_decay_requote_bps,
                ),
            ),
            route_instability_cooldown_decay_plan_bps=max(
                0.0,
                to_float(
                    redis_config.get("route_instability_cooldown_decay_plan_bps"),
                    defaults.route_instability_cooldown_decay_plan_bps,
                ),
            ),
            initial_profit_gate_relaxed_lamports=max(
                0,
                to_int(
                    redis_config.get("initial_profit_gate_relaxed_lamports"),
                    defaults.initial_profit_gate_relaxed_lamports,
                ),
            ),
            enable_probe_unconstrained=to_bool(
                redis_config.get("enable_probe_unconstrained"),
                defaults.enable_probe_unconstrained,
            ),
            enable_probe_multi_amount=to_bool(
                redis_config.get("enable_probe_multi_amount"),
                defaults.enable_probe_multi_amount,
            ),
            enable_decay_metrics_logging=to_bool(
                redis_config.get("enable_decay_metrics_logging"),
                defaults.enable_decay_metrics_logging,
            ),
            enable_priority_fee_breakdown_logging=to_bool(
                redis_config.get("enable_priority_fee_breakdown_logging"),
                defaults.enable_priority_fee_breakdown_logging,
            ),
            enable_stagea_relaxed_gate=to_bool(
                redis_config.get("enable_stagea_relaxed_gate"),
                defaults.enable_stagea_relaxed_gate,
            ),
            enable_initial_profit_gate_relaxed=to_bool(
                redis_config.get("enable_initial_profit_gate_relaxed"),
                defaults.enable_initial_profit_gate_relaxed,
            ),
            enable_route_instability_cooldown=to_bool(
                redis_config.get("enable_route_instability_cooldown"),
                defaults.enable_route_instability_cooldown,
            ),
            enable_legacy_swap_fallback=to_bool(
                redis_config.get("enable_legacy_swap_fallback"),
                defaults.enable_legacy_swap_fallback,
            ),
            allow_stageb_fail_probe=to_bool(
                redis_config.get("allow_stageb_fail_probe"),
                defaults.allow_stageb_fail_probe,
            ),
            base_amount_sweep_candidates_raw=int_list_from_redis(
                "base_amount_sweep_candidates_raw",
                defaults.base_amount_sweep_candidates_raw,
            ),
            base_amount_sweep_multipliers=float_list_from_redis(
                "base_amount_sweep_multipliers",
                defaults.base_amount_sweep_multipliers,
            ),
            base_amount_max_raw=max(
                0,
                to_int(redis_config.get("base_amount_max_raw"), defaults.base_amount_max_raw),
            ),
            min_expected_profit_lamports=max(
                0,
                to_int(
                    redis_config.get("min_expected_profit_lamports"),
                    defaults.min_expected_profit_lamports,
                ),
            ),
        )


@dataclass(slots=True, frozen=True)
class PriorityFeePlan:
    selected_micro_lamports: int
    recommended_micro_lamports: int
    max_fee_micro_lamports: int
    sample_size: int
    source: str
    exceeds_max: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class TradeDecision:
    profitable: bool
    should_execute: bool
    spread_bps: float
    required_spread_bps: float
    total_fee_bps: float
    required_spread_bps_single: float
    required_spread_bps_bundle: float
    total_fee_bps_single: float
    total_fee_bps_bundle: float
    reason: str
    blocked_by_fee_cap: bool
    priority_fee_micro_lamports: int
    tip_lamports: int
    tip_fee_bps: float
    atomic_margin_bps: float
    atomic_margin_single_bps: float
    atomic_margin_bundle_bps: float
    expected_net_bps: float
    expected_net_bps_single: float
    expected_net_bps_bundle: float
    expected_net_lamports_single: int
    expected_net_lamports_bundle: int


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
    order_id: str
    idempotency_key: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OrderGuardStore(Protocol):
    async def acquire_order_guard(
        self,
        *,
        guard_key: str,
        order_id: str,
        ttl_seconds: int,
    ) -> bool:
        ...

    async def get_order_guard(self, *, guard_key: str) -> dict[str, Any] | None:
        ...

    async def refresh_order_guard(
        self,
        *,
        guard_key: str,
        order_id: str,
        ttl_seconds: int,
    ) -> bool:
        ...

    async def release_order_guard(
        self,
        *,
        guard_key: str,
        order_id: str,
    ) -> bool:
        ...

    async def record_order_state(
        self,
        *,
        order_id: str,
        status: str,
        ttl_seconds: int,
        payload: dict[str, Any] | None = None,
        guard_key: str | None = None,
    ) -> None:
        ...

    async def list_order_records(
        self,
        *,
        statuses: set[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        ...

    async def save_pending_atomic(
        self,
        *,
        plan_id: str,
        status: str,
        order_id: str,
        guard_key: str,
        tx_signatures: list[str],
        ttl_seconds: int,
        mode: str,
        bundle_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        ...

    async def update_pending_atomic(
        self,
        *,
        plan_id: str,
        status: str,
        ttl_seconds: int,
        tx_signatures: list[str] | None = None,
        bundle_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        ...

    async def list_pending_atomic(
        self,
        *,
        statuses: set[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        ...

    async def delete_pending_atomic(self, *, plan_id: str) -> bool:
        ...


class OrderExecutor(Protocol):
    async def connect(self) -> None:
        ...

    async def close(self) -> None:
        ...

    async def healthcheck(self) -> None:
        ...

    async def resolve_priority_fee(self, *, runtime_config: RuntimeConfig) -> PriorityFeePlan:
        ...

    async def get_wallet_balance_lamports(self) -> int | None:
        ...

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
        ...


def build_idempotency_key(*, pair: PairConfig, observation: SpreadObservation) -> str:
    fingerprint = {
        "pair": pair.symbol,
        "input_mint": pair.base_mint,
        "output_mint": pair.quote_mint,
        "amount_in": pair.base_amount,
        "slippage_bps": pair.slippage_bps,
        "forward_out_amount": observation.forward_out_amount,
        "reverse_out_amount": observation.reverse_out_amount,
        "spread_bps": round(observation.spread_bps, 4),
        "forward_context_slot": observation.forward_quote.get("contextSlot"),
        "reverse_context_slot": observation.reverse_quote.get("contextSlot"),
        "quote_params": observation.quote_params,
        "forward_route_dexes": list(observation.forward_route_dexes),
        "reverse_route_dexes": list(observation.reverse_route_dexes),
    }
    encoded = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
