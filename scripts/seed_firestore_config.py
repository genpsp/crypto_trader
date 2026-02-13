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
    "min_spread_bps": 2.2,
    "dex_fee_bps": 0.0,
    "priority_fee_micro_lamports": 6_000,
    "priority_compute_units": 200_000,
    "priority_fee_percentile": 0.6,
    "priority_fee_multiplier": 1.1,
    "max_fee_micro_lamports": 30_000,
    "trade_enabled": False,
    "order_guard_ttl_seconds": 20,
    "execution_mode": "atomic",
    "atomic_send_mode": "auto",
    "atomic_expiry_ms": 5_000,
    "atomic_margin_bps": 3.0,
    "atomic_bundle_min_expected_net_bps": 3.0,
    "jito_block_engine_url": "",
    "jito_tip_lamports_max": 20_000,
    "jito_tip_lamports_recommended": 5_000,
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
        default=env_bool("TRADE_ENABLED", DEFAULT_CONFIG["trade_enabled"]),
        help="Set trade_enabled true/false. Defaults to TRADE_ENABLED env or false.",
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
        "atomic_bundle_min_expected_net_bps": max(0.0, float(args.atomic_bundle_min_expected_net_bps)),
        "jito_block_engine_url": str(args.jito_block_engine_url).strip(),
        "jito_tip_lamports_max": max(0, int(args.jito_tip_lamports_max)),
        "jito_tip_lamports_recommended": max(0, int(args.jito_tip_lamports_recommended)),
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
