#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ModuleNotFoundError:
    DOTENV_AVAILABLE = False

    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "min_spread_bps": 0.8,
    "initial_min_spread_bps": 0.2,
    "dex_fee_bps": 0.0,
    "priority_fee_micro_lamports": 6_000,
    "priority_compute_units": 200_000,
    "priority_fee_percentile": 0.6,
    "priority_fee_multiplier": 1.1,
    "max_fee_micro_lamports": 30_000,
    "trade_enabled": True,
    "order_guard_ttl_seconds": 20,
    "execution_mode": "atomic",
    "atomic_send_mode": "auto",
    "atomic_expiry_ms": 6_000,
    "atomic_margin_bps": 1.2,
    "initial_atomic_margin_bps": 0.4,
    "min_stagea_margin_bps": 0.3,
    "allow_stageb_fail_probe": False,
    "atomic_bundle_min_expected_net_bps": 3.0,
    "jito_block_engine_url": "",
    "jito_tip_lamports_max": 8_000,
    "jito_tip_lamports_recommended": 3_000,
    "jito_tip_share": 0.1,
    "quote_default_params_json": '{"onlyDirectRoutes": false, "restrictIntermediateTokens": false, "maxAccounts": 64}',
    "quote_initial_params_json": '{"onlyDirectRoutes": false, "restrictIntermediateTokens": false, "maxAccounts": 64}',
    "quote_plan_params_json": '{"onlyDirectRoutes": false, "restrictIntermediateTokens": false, "maxAccounts": 48}',
    "quote_dex_allowlist_forward": ["Raydium CLMM", "Orca Whirlpool", "Meteora DLMM", "Raydium CPMM"],
    "quote_dex_allowlist_reverse": ["Raydium CLMM", "Orca Whirlpool", "Meteora DLMM", "Raydium CPMM"],
    "quote_dex_excludelist": [],
    "quote_exploration_mode": "TIER1_ONLY",
    "quote_dex_sweep_topk": 1,
    "quote_dex_sweep_topk_max": 1,
    "quote_dex_sweep_combo_limit": 1,
    "quote_near_miss_expand_bps": 999.0,
    "quote_median_requote_max_range_bps": 3.0,
    "max_requote_range_bps": 3.0,
    "min_improvement_bps": 0.2,
    "quote_max_rps": 0.4,
    "quote_exploration_max_rps": 0.3,
    "quote_execution_max_rps": 0.15,
    "quote_cache_ttl_ms": 1500,
    "quote_no_routes_cache_ttl_seconds": 300,
    "quote_probe_max_routes": 50,
    "quote_probe_base_amounts_raw": [10_000_000, 20_000_000, 40_000_000],
    "quote_dynamic_allowlist_topk": 10,
    "quote_dynamic_allowlist_good_candidate_alpha": 2.0,
    "quote_dynamic_allowlist_ttl_seconds": 300,
    "quote_dynamic_allowlist_refresh_seconds": 5.0,
    "quote_negative_fallback_streak_threshold": 10,
    "route_instability_cooldown_requote_seconds": 60.0,
    "route_instability_cooldown_plan_seconds": 120.0,
    "route_instability_cooldown_decay_requote_bps": 2.0,
    "route_instability_cooldown_decay_plan_bps": 2.0,
    "initial_profit_gate_relaxed_lamports": 500,
    "enable_probe_unconstrained": False,
    "enable_probe_multi_amount": False,
    "enable_decay_metrics_logging": True,
    "enable_priority_fee_breakdown_logging": True,
    "enable_stagea_relaxed_gate": True,
    "enable_initial_profit_gate_relaxed": True,
    "enable_route_instability_cooldown": True,
    "enable_legacy_swap_fallback": False,
    "base_amount_sweep_candidates_raw": [300_000_000],
    "base_amount_sweep_multipliers": [1, 3, 10],
    "base_amount_max_raw": 300_000_000,
    "min_expected_profit_lamports": 500,
}


def fallback_load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'") and len(value) >= 2:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(float(raw.strip()))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_string_list(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for item in parse_string_list(raw):
        try:
            parsed = float(item)
        except ValueError:
            continue
        if parsed > 0:
            values.append(parsed)
    return values


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for item in parse_string_list(raw):
        try:
            parsed = int(float(item))
        except ValueError:
            continue
        if parsed > 0 and parsed not in values:
            values.append(parsed)
    return values


def parse_args() -> argparse.Namespace:
    bot_collection = (os.getenv("BOT_COLLECTION", "bots").strip("/") or "bots")
    bot_id = (os.getenv("BOT_ID", "solana-bot").strip() or "solana-bot").replace("/", "-")
    default_config_doc = os.getenv("FIRESTORE_CONFIG_DOC") or f"{bot_collection}/{bot_id}/config/runtime"

    parser = argparse.ArgumentParser(
        description="Seed initial runtime config to Firestore for the Solana bot.",
    )

    parser.add_argument(
        "--project-id",
        default=os.getenv("FIRESTORE_PROJECT_ID", ""),
        help="GCP project id. Defaults to FIRESTORE_PROJECT_ID from env.",
    )
    parser.add_argument(
        "--config-doc",
        default=default_config_doc,
        help="Firestore target path. If odd segments are given, a doc id is auto-appended.",
    )
    parser.add_argument(
        "--leaf-doc-id",
        default=os.getenv("FIRESTORE_CONFIG_LEAF_DOC_ID", "runtime"),
        help="Doc id to append when --config-doc is a collection path.",
    )
    parser.add_argument(
        "--credentials",
        default=os.getenv("FIREBASE_CREDENTIALS", ""),
        help="Service account json path. Defaults to FIREBASE_CREDENTIALS from env.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the full document (merge=false).",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print resolved doc path and payload without writing to Firestore.",
    )

    parser.add_argument(
        "--schema-version",
        type=int,
        default=max(1, env_int("CONFIG_SCHEMA_VERSION", DEFAULT_CONFIG["schema_version"])),
    )
    parser.add_argument(
        "--min-spread-bps",
        type=float,
        default=env_float("MIN_SPREAD_BPS", DEFAULT_CONFIG["min_spread_bps"]),
    )
    parser.add_argument(
        "--initial-min-spread-bps",
        type=float,
        default=env_float("INITIAL_MIN_SPREAD_BPS", DEFAULT_CONFIG["initial_min_spread_bps"]),
    )
    parser.add_argument(
        "--dex-fee-bps",
        type=float,
        default=env_float("DEX_FEE_BPS", DEFAULT_CONFIG["dex_fee_bps"]),
    )
    parser.add_argument(
        "--priority-fee-micro-lamports",
        type=int,
        default=env_int(
            "PRIORITY_FEE_MICRO_LAMPORTS",
            DEFAULT_CONFIG["priority_fee_micro_lamports"],
        ),
    )
    parser.add_argument(
        "--priority-compute-units",
        type=int,
        default=max(1, env_int("PRIORITY_COMPUTE_UNITS", DEFAULT_CONFIG["priority_compute_units"])),
    )
    parser.add_argument(
        "--priority-fee-percentile",
        type=float,
        default=env_float("PRIORITY_FEE_PERCENTILE", DEFAULT_CONFIG["priority_fee_percentile"]),
    )
    parser.add_argument(
        "--priority-fee-multiplier",
        type=float,
        default=env_float("PRIORITY_FEE_MULTIPLIER", DEFAULT_CONFIG["priority_fee_multiplier"]),
    )
    parser.add_argument(
        "--max-fee-micro-lamports",
        type=int,
        default=env_int("MAX_FEE_MICRO_LAMPORTS", DEFAULT_CONFIG["max_fee_micro_lamports"]),
    )
    parser.add_argument(
        "--trade-enabled",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG["trade_enabled"],
        help="Set trade_enabled true/false. Defaults to true (override with --no-trade-enabled).",
    )
    parser.add_argument(
        "--order-guard-ttl-seconds",
        type=int,
        default=max(1, env_int("ORDER_GUARD_TTL_SECONDS", DEFAULT_CONFIG["order_guard_ttl_seconds"])),
    )

    parser.add_argument(
        "--execution-mode",
        choices=["atomic", "legacy"],
        default=os.getenv("EXECUTION_MODE", DEFAULT_CONFIG["execution_mode"]),
    )
    parser.add_argument(
        "--atomic-send-mode",
        choices=["single_tx", "bundle", "auto"],
        default=os.getenv("ATOMIC_SEND_MODE", DEFAULT_CONFIG["atomic_send_mode"]),
    )
    parser.add_argument(
        "--atomic-expiry-ms",
        type=int,
        default=env_int("ATOMIC_EXPIRY_MS", DEFAULT_CONFIG["atomic_expiry_ms"]),
    )
    parser.add_argument(
        "--atomic-margin-bps",
        type=float,
        default=env_float("ATOMIC_MARGIN_BPS", DEFAULT_CONFIG["atomic_margin_bps"]),
    )
    parser.add_argument(
        "--initial-atomic-margin-bps",
        type=float,
        default=env_float("INITIAL_ATOMIC_MARGIN_BPS", DEFAULT_CONFIG["initial_atomic_margin_bps"]),
    )
    parser.add_argument(
        "--min-stagea-margin-bps",
        type=float,
        default=env_float("MIN_STAGEA_MARGIN_BPS", DEFAULT_CONFIG["min_stagea_margin_bps"]),
    )
    parser.add_argument(
        "--allow-stageb-fail-probe",
        action=argparse.BooleanOptionalAction,
        default=env_bool("ALLOW_STAGEB_FAIL_PROBE", DEFAULT_CONFIG["allow_stageb_fail_probe"]),
    )
    parser.add_argument(
        "--atomic-bundle-min-expected-net-bps",
        type=float,
        default=env_float(
            "ATOMIC_BUNDLE_MIN_EXPECTED_NET_BPS",
            DEFAULT_CONFIG["atomic_bundle_min_expected_net_bps"],
        ),
    )
    parser.add_argument(
        "--jito-block-engine-url",
        default=os.getenv("JITO_BLOCK_ENGINE_URL", DEFAULT_CONFIG["jito_block_engine_url"]),
    )
    parser.add_argument(
        "--jito-tip-lamports-max",
        type=int,
        default=env_int("JITO_TIP_LAMPORTS_MAX", DEFAULT_CONFIG["jito_tip_lamports_max"]),
    )
    parser.add_argument(
        "--jito-tip-lamports-recommended",
        type=int,
        default=env_int(
            "JITO_TIP_LAMPORTS_RECOMMENDED",
            DEFAULT_CONFIG["jito_tip_lamports_recommended"],
        ),
    )
    parser.add_argument(
        "--jito-tip-share",
        type=float,
        default=env_float("JITO_TIP_SHARE", DEFAULT_CONFIG["jito_tip_share"]),
    )
    parser.add_argument(
        "--quote-default-params-json",
        default=os.getenv("QUOTE_DEFAULT_PARAMS_JSON", DEFAULT_CONFIG["quote_default_params_json"]),
    )
    parser.add_argument(
        "--quote-initial-params-json",
        default=os.getenv("QUOTE_INITIAL_PARAMS_JSON", DEFAULT_CONFIG["quote_initial_params_json"]),
    )
    parser.add_argument(
        "--quote-plan-params-json",
        default=os.getenv("QUOTE_PLAN_PARAMS_JSON", DEFAULT_CONFIG["quote_plan_params_json"]),
    )
    parser.add_argument(
        "--quote-dex-allowlist-forward",
        default=os.getenv(
            "QUOTE_DEX_ALLOWLIST_FORWARD",
            json.dumps(DEFAULT_CONFIG["quote_dex_allowlist_forward"], ensure_ascii=False),
        ),
    )
    parser.add_argument(
        "--quote-dex-allowlist-reverse",
        default=os.getenv(
            "QUOTE_DEX_ALLOWLIST_REVERSE",
            json.dumps(DEFAULT_CONFIG["quote_dex_allowlist_reverse"], ensure_ascii=False),
        ),
    )
    parser.add_argument(
        "--quote-dex-excludelist",
        default=os.getenv(
            "QUOTE_DEX_EXCLUDELIST",
            json.dumps(DEFAULT_CONFIG["quote_dex_excludelist"], ensure_ascii=False),
        ),
    )
    parser.add_argument(
        "--quote-exploration-mode",
        default=os.getenv(
            "QUOTE_EXPLORATION_MODE",
            DEFAULT_CONFIG["quote_exploration_mode"],
        ),
    )
    parser.add_argument(
        "--quote-dex-sweep-topk",
        type=int,
        default=max(1, env_int("QUOTE_DEX_SWEEP_TOPK", DEFAULT_CONFIG["quote_dex_sweep_topk"])),
    )
    parser.add_argument(
        "--quote-dex-sweep-topk-max",
        type=int,
        default=max(
            1,
            env_int("QUOTE_DEX_SWEEP_TOPK_MAX", DEFAULT_CONFIG["quote_dex_sweep_topk_max"]),
        ),
    )
    parser.add_argument(
        "--quote-dex-sweep-combo-limit",
        type=int,
        default=max(
            1,
            env_int("QUOTE_DEX_SWEEP_COMBO_LIMIT", DEFAULT_CONFIG["quote_dex_sweep_combo_limit"]),
        ),
    )
    parser.add_argument(
        "--quote-near-miss-expand-bps",
        type=float,
        default=max(
            0.0,
            env_float("QUOTE_NEAR_MISS_EXPAND_BPS", DEFAULT_CONFIG["quote_near_miss_expand_bps"]),
        ),
    )
    parser.add_argument(
        "--quote-median-requote-max-range-bps",
        type=float,
        default=max(
            0.0,
            env_float(
                "QUOTE_MEDIAN_REQUOTE_MAX_RANGE_BPS",
                DEFAULT_CONFIG["quote_median_requote_max_range_bps"],
            ),
        ),
    )
    parser.add_argument(
        "--max-requote-range-bps",
        type=float,
        default=max(
            0.0,
            env_float("MAX_REQUOTE_RANGE_BPS", DEFAULT_CONFIG["max_requote_range_bps"]),
        ),
    )
    parser.add_argument(
        "--min-improvement-bps",
        type=float,
        default=max(
            0.0,
            env_float("MIN_IMPROVEMENT_BPS", DEFAULT_CONFIG["min_improvement_bps"]),
        ),
    )
    parser.add_argument(
        "--quote-max-rps",
        type=float,
        default=max(0.1, env_float("QUOTE_MAX_RPS", DEFAULT_CONFIG["quote_max_rps"])),
    )
    parser.add_argument(
        "--quote-exploration-max-rps",
        type=float,
        default=max(
            0.05,
            env_float(
                "QUOTE_EXPLORATION_MAX_RPS",
                DEFAULT_CONFIG["quote_exploration_max_rps"],
            ),
        ),
    )
    parser.add_argument(
        "--quote-execution-max-rps",
        type=float,
        default=max(
            0.05,
            env_float(
                "QUOTE_EXECUTION_MAX_RPS",
                DEFAULT_CONFIG["quote_execution_max_rps"],
            ),
        ),
    )
    parser.add_argument(
        "--quote-cache-ttl-ms",
        type=int,
        default=max(0, env_int("QUOTE_CACHE_TTL_MS", DEFAULT_CONFIG["quote_cache_ttl_ms"])),
    )
    parser.add_argument(
        "--quote-no-routes-cache-ttl-seconds",
        type=int,
        default=max(
            1,
            env_int(
                "QUOTE_NO_ROUTES_CACHE_TTL_SECONDS",
                DEFAULT_CONFIG["quote_no_routes_cache_ttl_seconds"],
            ),
        ),
    )
    parser.add_argument(
        "--quote-probe-max-routes",
        type=int,
        default=max(
            1,
            env_int(
                "QUOTE_PROBE_MAX_ROUTES",
                DEFAULT_CONFIG["quote_probe_max_routes"],
            ),
        ),
    )
    parser.add_argument(
        "--quote-probe-base-amounts-raw",
        default=os.getenv(
            "QUOTE_PROBE_BASE_AMOUNTS_RAW",
            json.dumps(DEFAULT_CONFIG["quote_probe_base_amounts_raw"], ensure_ascii=False),
        ),
    )
    parser.add_argument(
        "--quote-dynamic-allowlist-topk",
        type=int,
        default=max(
            1,
            env_int("QUOTE_DYNAMIC_ALLOWLIST_TOPK", DEFAULT_CONFIG["quote_dynamic_allowlist_topk"]),
        ),
    )
    parser.add_argument(
        "--quote-dynamic-allowlist-good-candidate-alpha",
        type=float,
        default=max(
            0.0,
            env_float(
                "QUOTE_DYNAMIC_ALLOWLIST_GOOD_ALPHA",
                DEFAULT_CONFIG["quote_dynamic_allowlist_good_candidate_alpha"],
            ),
        ),
    )
    parser.add_argument(
        "--quote-dynamic-allowlist-ttl-seconds",
        type=int,
        default=max(
            1,
            env_int(
                "QUOTE_DYNAMIC_ALLOWLIST_TTL_SECONDS",
                DEFAULT_CONFIG["quote_dynamic_allowlist_ttl_seconds"],
            ),
        ),
    )
    parser.add_argument(
        "--quote-dynamic-allowlist-refresh-seconds",
        type=float,
        default=max(
            0.5,
            env_float(
                "QUOTE_DYNAMIC_ALLOWLIST_REFRESH_SECONDS",
                DEFAULT_CONFIG["quote_dynamic_allowlist_refresh_seconds"],
            ),
        ),
    )
    parser.add_argument(
        "--quote-negative-fallback-streak-threshold",
        type=int,
        default=max(
            1,
            env_int(
                "QUOTE_NEGATIVE_FALLBACK_STREAK_THRESHOLD",
                DEFAULT_CONFIG["quote_negative_fallback_streak_threshold"],
            ),
        ),
    )
    parser.add_argument(
        "--route-instability-cooldown-requote-seconds",
        type=float,
        default=max(
            0.0,
            env_float(
                "ROUTE_INSTABILITY_COOLDOWN_REQUOTE_SECONDS",
                DEFAULT_CONFIG["route_instability_cooldown_requote_seconds"],
            ),
        ),
    )
    parser.add_argument(
        "--route-instability-cooldown-plan-seconds",
        type=float,
        default=max(
            0.0,
            env_float(
                "ROUTE_INSTABILITY_COOLDOWN_PLAN_SECONDS",
                DEFAULT_CONFIG["route_instability_cooldown_plan_seconds"],
            ),
        ),
    )
    parser.add_argument(
        "--route-instability-cooldown-decay-requote-bps",
        type=float,
        default=max(
            0.0,
            env_float(
                "ROUTE_INSTABILITY_COOLDOWN_DECAY_REQUOTE_BPS",
                DEFAULT_CONFIG["route_instability_cooldown_decay_requote_bps"],
            ),
        ),
    )
    parser.add_argument(
        "--route-instability-cooldown-decay-plan-bps",
        type=float,
        default=max(
            0.0,
            env_float(
                "ROUTE_INSTABILITY_COOLDOWN_DECAY_PLAN_BPS",
                DEFAULT_CONFIG["route_instability_cooldown_decay_plan_bps"],
            ),
        ),
    )
    parser.add_argument(
        "--initial-profit-gate-relaxed-lamports",
        type=int,
        default=max(
            0,
            env_int(
                "INITIAL_PROFIT_GATE_RELAXED_LAMPORTS",
                DEFAULT_CONFIG["initial_profit_gate_relaxed_lamports"],
            ),
        ),
    )
    parser.add_argument(
        "--enable-probe-unconstrained",
        action=argparse.BooleanOptionalAction,
        default=env_bool("ENABLE_PROBE_UNCONSTRAINED", DEFAULT_CONFIG["enable_probe_unconstrained"]),
    )
    parser.add_argument(
        "--enable-probe-multi-amount",
        action=argparse.BooleanOptionalAction,
        default=env_bool("ENABLE_PROBE_MULTI_AMOUNT", DEFAULT_CONFIG["enable_probe_multi_amount"]),
    )
    parser.add_argument(
        "--enable-decay-metrics-logging",
        action=argparse.BooleanOptionalAction,
        default=env_bool("ENABLE_DECAY_METRICS_LOGGING", DEFAULT_CONFIG["enable_decay_metrics_logging"]),
    )
    parser.add_argument(
        "--enable-priority-fee-breakdown-logging",
        action=argparse.BooleanOptionalAction,
        default=env_bool(
            "ENABLE_PRIORITY_FEE_BREAKDOWN_LOGGING",
            DEFAULT_CONFIG["enable_priority_fee_breakdown_logging"],
        ),
    )
    parser.add_argument(
        "--enable-stagea-relaxed-gate",
        action=argparse.BooleanOptionalAction,
        default=env_bool("ENABLE_STAGEA_RELAXED_GATE", DEFAULT_CONFIG["enable_stagea_relaxed_gate"]),
    )
    parser.add_argument(
        "--enable-initial-profit-gate-relaxed",
        action=argparse.BooleanOptionalAction,
        default=env_bool(
            "ENABLE_INITIAL_PROFIT_GATE_RELAXED",
            DEFAULT_CONFIG["enable_initial_profit_gate_relaxed"],
        ),
    )
    parser.add_argument(
        "--enable-route-instability-cooldown",
        action=argparse.BooleanOptionalAction,
        default=env_bool(
            "ENABLE_ROUTE_INSTABILITY_COOLDOWN",
            DEFAULT_CONFIG["enable_route_instability_cooldown"],
        ),
    )
    parser.add_argument(
        "--enable-legacy-swap-fallback",
        action=argparse.BooleanOptionalAction,
        default=env_bool(
            "ENABLE_LEGACY_SWAP_FALLBACK",
            DEFAULT_CONFIG["enable_legacy_swap_fallback"],
        ),
    )
    parser.add_argument(
        "--base-amount-sweep-candidates-raw",
        default=os.getenv(
            "BASE_AMOUNT_SWEEP_CANDIDATES_RAW",
            json.dumps(DEFAULT_CONFIG["base_amount_sweep_candidates_raw"], ensure_ascii=False),
        ),
    )
    parser.add_argument(
        "--base-amount-sweep-multipliers",
        default=os.getenv(
            "BASE_AMOUNT_SWEEP_MULTIPLIERS",
            json.dumps(DEFAULT_CONFIG["base_amount_sweep_multipliers"], ensure_ascii=False),
        ),
    )
    parser.add_argument(
        "--base-amount-max-raw",
        type=int,
        default=max(0, env_int("BASE_AMOUNT_MAX_RAW", DEFAULT_CONFIG["base_amount_max_raw"])),
    )
    parser.add_argument(
        "--min-expected-profit-lamports",
        type=int,
        default=max(
            0,
            env_int("MIN_EXPECTED_PROFIT_LAMPORTS", DEFAULT_CONFIG["min_expected_profit_lamports"]),
        ),
    )

    return parser.parse_args()


def resolve_credentials_path(raw_path: str, repo_root: Path) -> str:
    path = raw_path.strip()
    if not path:
        return ""

    if path.startswith("/app/"):
        mapped = repo_root / path.removeprefix("/app/")
        if mapped.exists():
            return str(mapped)

    return path


def normalize_doc_path(doc_path: str, leaf_doc_id: str) -> tuple[str, bool]:
    normalized = doc_path.strip("/")
    if not normalized:
        raise ValueError("FIRESTORE_CONFIG_DOC is empty.")

    segments = [part for part in normalized.split("/") if part]
    if len(segments) % 2 == 0:
        return normalized, False

    return f"{normalized}/{leaf_doc_id}", True


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": max(1, int(args.schema_version)),
        "min_spread_bps": float(args.min_spread_bps),
        "initial_min_spread_bps": max(0.0, float(args.initial_min_spread_bps)),
        "dex_fee_bps": float(args.dex_fee_bps),
        "priority_fee_micro_lamports": int(args.priority_fee_micro_lamports),
        "priority_compute_units": max(1, int(args.priority_compute_units)),
        "priority_fee_percentile": float(args.priority_fee_percentile),
        "priority_fee_multiplier": float(args.priority_fee_multiplier),
        "max_fee_micro_lamports": int(args.max_fee_micro_lamports),
        "trade_enabled": bool(args.trade_enabled),
        "order_guard_ttl_seconds": int(args.order_guard_ttl_seconds),
        "execution_mode": str(args.execution_mode),
        "atomic_send_mode": str(args.atomic_send_mode),
        "atomic_expiry_ms": max(250, int(args.atomic_expiry_ms)),
        "atomic_margin_bps": max(0.0, float(args.atomic_margin_bps)),
        "initial_atomic_margin_bps": max(0.0, float(args.initial_atomic_margin_bps)),
        "min_stagea_margin_bps": max(0.0, float(args.min_stagea_margin_bps)),
        "allow_stageb_fail_probe": bool(args.allow_stageb_fail_probe),
        "atomic_bundle_min_expected_net_bps": max(0.0, float(args.atomic_bundle_min_expected_net_bps)),
        "jito_block_engine_url": str(args.jito_block_engine_url).strip(),
        "jito_tip_lamports_max": max(0, int(args.jito_tip_lamports_max)),
        "jito_tip_lamports_recommended": max(0, int(args.jito_tip_lamports_recommended)),
        "jito_tip_share": max(0.0, min(1.0, float(args.jito_tip_share))),
        "quote_default_params_json": str(args.quote_default_params_json).strip(),
        "quote_initial_params_json": str(args.quote_initial_params_json).strip(),
        "quote_plan_params_json": str(args.quote_plan_params_json).strip(),
        "quote_dex_allowlist_forward": parse_string_list(str(args.quote_dex_allowlist_forward)),
        "quote_dex_allowlist_reverse": parse_string_list(str(args.quote_dex_allowlist_reverse)),
        "quote_dex_excludelist": parse_string_list(str(args.quote_dex_excludelist)),
        "quote_exploration_mode": str(args.quote_exploration_mode).strip().upper() or "MIXED",
        "quote_dex_sweep_topk": max(1, int(args.quote_dex_sweep_topk)),
        "quote_dex_sweep_topk_max": max(1, int(args.quote_dex_sweep_topk_max)),
        "quote_dex_sweep_combo_limit": max(1, int(args.quote_dex_sweep_combo_limit)),
        "quote_near_miss_expand_bps": max(0.0, float(args.quote_near_miss_expand_bps)),
        "quote_median_requote_max_range_bps": max(
            0.0,
            float(args.quote_median_requote_max_range_bps),
        ),
        "max_requote_range_bps": max(0.0, float(args.max_requote_range_bps)),
        "min_improvement_bps": max(0.0, float(args.min_improvement_bps)),
        "quote_max_rps": max(0.1, float(args.quote_max_rps)),
        "quote_exploration_max_rps": max(
            0.05,
            min(float(args.quote_exploration_max_rps), float(args.quote_max_rps)),
        ),
        "quote_execution_max_rps": max(
            0.05,
            min(float(args.quote_execution_max_rps), float(args.quote_max_rps)),
        ),
        "quote_cache_ttl_ms": max(0, int(args.quote_cache_ttl_ms)),
        "quote_no_routes_cache_ttl_seconds": max(1, int(args.quote_no_routes_cache_ttl_seconds)),
        "quote_probe_max_routes": max(1, int(args.quote_probe_max_routes)),
        "quote_probe_base_amounts_raw": parse_int_list(str(args.quote_probe_base_amounts_raw)),
        "quote_dynamic_allowlist_topk": max(1, int(args.quote_dynamic_allowlist_topk)),
        "quote_dynamic_allowlist_good_candidate_alpha": max(
            0.0,
            float(args.quote_dynamic_allowlist_good_candidate_alpha),
        ),
        "quote_dynamic_allowlist_ttl_seconds": max(1, int(args.quote_dynamic_allowlist_ttl_seconds)),
        "quote_dynamic_allowlist_refresh_seconds": max(
            0.5,
            float(args.quote_dynamic_allowlist_refresh_seconds),
        ),
        "quote_negative_fallback_streak_threshold": max(
            1,
            int(args.quote_negative_fallback_streak_threshold),
        ),
        "route_instability_cooldown_requote_seconds": max(
            0.0,
            float(args.route_instability_cooldown_requote_seconds),
        ),
        "route_instability_cooldown_plan_seconds": max(
            0.0,
            float(args.route_instability_cooldown_plan_seconds),
        ),
        "route_instability_cooldown_decay_requote_bps": max(
            0.0,
            float(args.route_instability_cooldown_decay_requote_bps),
        ),
        "route_instability_cooldown_decay_plan_bps": max(
            0.0,
            float(args.route_instability_cooldown_decay_plan_bps),
        ),
        "initial_profit_gate_relaxed_lamports": max(
            0,
            int(args.initial_profit_gate_relaxed_lamports),
        ),
        "enable_probe_unconstrained": bool(args.enable_probe_unconstrained),
        "enable_probe_multi_amount": bool(args.enable_probe_multi_amount),
        "enable_decay_metrics_logging": bool(args.enable_decay_metrics_logging),
        "enable_priority_fee_breakdown_logging": bool(args.enable_priority_fee_breakdown_logging),
        "enable_stagea_relaxed_gate": bool(args.enable_stagea_relaxed_gate),
        "enable_initial_profit_gate_relaxed": bool(args.enable_initial_profit_gate_relaxed),
        "enable_route_instability_cooldown": bool(args.enable_route_instability_cooldown),
        "enable_legacy_swap_fallback": bool(args.enable_legacy_swap_fallback),
        "base_amount_sweep_candidates_raw": parse_int_list(str(args.base_amount_sweep_candidates_raw)),
        "base_amount_sweep_multipliers": parse_float_list(str(args.base_amount_sweep_multipliers)) or [1.0],
        "base_amount_max_raw": max(0, int(args.base_amount_max_raw)),
        "min_expected_profit_lamports": max(0, int(args.min_expected_profit_lamports)),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"

    if DOTENV_AVAILABLE:
        load_dotenv(env_path)
    else:
        fallback_load_env_file(env_path)

    args = parse_args()

    target_doc_path, path_auto_fixed = normalize_doc_path(args.config_doc, args.leaf_doc_id)
    payload = build_payload(args)

    if path_auto_fixed:
        print(
            f"[info] --config-doc '{args.config_doc}' is a collection path. "
            f"Using document path '{target_doc_path}'."
        )

    credentials_path = resolve_credentials_path(args.credentials, repo_root)
    if credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    project_id = args.project_id.strip()
    if not project_id:
        raise ValueError("FIRESTORE_PROJECT_ID is required (set env or --project-id).")

    print(f"[info] project_id={project_id}")
    print(f"[info] target_doc={target_doc_path}")
    print(f"[info] merge={not args.replace}")
    print("[info] payload=")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.print_only:
        print("[info] print-only mode: skipped Firestore write")
        return

    try:
        from google.cloud import firestore
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "google-cloud-firestore is not installed. Run `pip install -r requirements.txt` first."
        ) from error

    client = firestore.Client(project=project_id)
    doc_ref = client.document(target_doc_path)
    doc_ref.set(payload, merge=not args.replace)

    print("[ok] Firestore config seeded successfully")


if __name__ == "__main__":
    main()
