#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ModuleNotFoundError:
    DOTENV_AVAILABLE = False

    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


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


def to_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_iso_datetime(value: str) -> datetime | None:
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        return parse_iso_datetime(value)
    return None


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    p = min(1.0, max(0.0, ratio))
    idx = (len(sorted_values) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_values[lo]

    weight = idx - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * weight


def resolve_credentials_path(raw_path: str, repo_root: Path) -> str:
    path = raw_path.strip()
    if not path:
        return ""

    if path.startswith("/app/"):
        mapped = repo_root / path.removeprefix("/app/")
        if mapped.exists():
            return str(mapped)
    return path


@dataclass(slots=True)
class TradeRow:
    trade_id: str
    created_at: datetime | None
    run_id: str
    pair: str
    status: str
    reason: str
    spread_bps: float | None
    total_fee_bps: float | None
    required_spread_bps: float | None
    priority_fee_micro_lamports: int | None
    amount_in_raw: int
    net_bps: float | None
    est_pnl_raw: float | None
    est_pnl_base: float | None


@dataclass(slots=True)
class AggregateSummary:
    scanned_docs: int
    matched_docs: int
    window_start_utc: str
    window_end_utc: str
    pair: str
    run_id: str
    statuses: list[str]
    status_counts: dict[str, int]
    reason_counts_top10: list[tuple[str, int]]
    spread_bps_avg: float
    total_fee_bps_avg: float
    net_bps_avg: float
    net_bps_p10: float
    net_bps_p50: float
    net_bps_p90: float
    net_positive_rate: float
    estimated_pnl_raw_total: float
    estimated_pnl_base_total: float


def parse_args() -> argparse.Namespace:
    bot_collection = (os.getenv("BOT_COLLECTION", "bots").strip("/") or "bots")
    bot_id = (os.getenv("BOT_ID", "solana-bot").strip() or "solana-bot").replace("/", "-")

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Firestore trades and compute dry-run quality metrics "
            "(net_bps / estimated pnl)."
        )
    )
    parser.add_argument(
        "--project-id",
        default=os.getenv("FIRESTORE_PROJECT_ID", ""),
        help="GCP project id. Defaults to FIRESTORE_PROJECT_ID.",
    )
    parser.add_argument(
        "--credentials",
        default=os.getenv("FIREBASE_CREDENTIALS", ""),
        help="Service account json path. Defaults to FIREBASE_CREDENTIALS.",
    )
    parser.add_argument("--bot-collection", default=bot_collection)
    parser.add_argument("--bot-id", default=bot_id)
    parser.add_argument(
        "--trades-collection",
        default=os.getenv("BOT_TRADES_COLLECTION", "trades"),
    )
    parser.add_argument(
        "--pair",
        default=os.getenv("PAIR_SYMBOL", ""),
        help="Filter by pair (ex: SOL/USDC). Empty means all.",
    )
    parser.add_argument(
        "--run-id",
        default=os.getenv("BOT_RUN_ID", ""),
        help="Filter by run_id. Empty means all runs.",
    )
    parser.add_argument(
        "--status",
        action="append",
        dest="statuses",
        default=[],
        help="Filter by status. Repeat to include multiple values.",
    )
    parser.add_argument(
        "--since-hours",
        type=float,
        default=24.0,
        help="Lookback window in hours. Use <=0 for no time filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Fetch at most N recent docs by created_at. Use <=0 for unlimited.",
    )
    parser.add_argument(
        "--base-amount",
        type=int,
        default=env_int("PAIR_BASE_AMOUNT", 1_000_000_000),
        help="Fallback notional (raw units) used for estimated pnl.",
    )
    parser.add_argument(
        "--base-decimals",
        type=int,
        default=env_int("PAIR_BASE_DECIMALS", 9),
        help="Decimals for base token (for estimated_pnl_base display).",
    )
    parser.add_argument(
        "--target-net-bps",
        type=float,
        default=1.0,
        help="Target edge used for min_spread recommendation.",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Optional CSV output path for matched rows.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print summary JSON only.",
    )
    return parser.parse_args()


def extract_created_at(payload: dict[str, Any], snapshot: Any) -> datetime | None:
    for key in ("created_at", "updated_at", "timestamp"):
        parsed = coerce_datetime(payload.get(key))
        if parsed is not None:
            return parsed
    snapshot_time = getattr(snapshot, "create_time", None)
    parsed_snapshot_time = coerce_datetime(snapshot_time)
    return parsed_snapshot_time


def parse_trade_row(
    payload: dict[str, Any],
    *,
    trade_id: str,
    created_at: datetime | None,
    base_amount_default: int,
    base_decimals: int,
) -> TradeRow:
    spread_bps = to_float(payload.get("spread_bps"))
    total_fee_bps = to_float(payload.get("total_fee_bps"))
    required_spread_bps = to_float(payload.get("required_spread_bps"))
    net_bps: float | None = None
    if spread_bps is not None and total_fee_bps is not None:
        net_bps = spread_bps - total_fee_bps

    metadata = payload.get("metadata")
    amount_in_raw = to_int(payload.get("amount_in"), None)
    if amount_in_raw is None and isinstance(metadata, dict):
        amount_in_raw = to_int(metadata.get("amount_in"), None)
    if amount_in_raw is None:
        amount_in_raw = base_amount_default

    est_pnl_raw: float | None = None
    est_pnl_base: float | None = None
    if net_bps is not None:
        est_pnl_raw = amount_in_raw * net_bps / 10_000
        est_pnl_base = est_pnl_raw / (10**base_decimals)

    return TradeRow(
        trade_id=trade_id,
        created_at=created_at,
        run_id=str(payload.get("run_id") or ""),
        pair=str(payload.get("pair") or ""),
        status=str(payload.get("status") or ""),
        reason=str(payload.get("reason") or ""),
        spread_bps=spread_bps,
        total_fee_bps=total_fee_bps,
        required_spread_bps=required_spread_bps,
        priority_fee_micro_lamports=to_int(payload.get("priority_fee_micro_lamports"), None),
        amount_in_raw=amount_in_raw,
        net_bps=net_bps,
        est_pnl_raw=est_pnl_raw,
        est_pnl_base=est_pnl_base,
    )


def build_summary(
    *,
    rows: list[TradeRow],
    scanned_docs: int,
    window_start_utc: datetime,
    window_end_utc: datetime,
    pair: str,
    run_id: str,
    statuses: list[str],
) -> AggregateSummary:
    status_counts = Counter(row.status for row in rows)
    reason_counts = Counter(row.reason for row in rows if row.reason)

    spreads = [row.spread_bps for row in rows if row.spread_bps is not None]
    fees = [row.total_fee_bps for row in rows if row.total_fee_bps is not None]
    nets = [row.net_bps for row in rows if row.net_bps is not None]
    pnl_raw_values = [row.est_pnl_raw for row in rows if row.est_pnl_raw is not None]
    pnl_base_values = [row.est_pnl_base for row in rows if row.est_pnl_base is not None]

    net_positive = sum(1 for value in nets if value > 0)
    net_positive_rate = (net_positive / len(nets)) if nets else 0.0

    return AggregateSummary(
        scanned_docs=scanned_docs,
        matched_docs=len(rows),
        window_start_utc=window_start_utc.isoformat(),
        window_end_utc=window_end_utc.isoformat(),
        pair=pair,
        run_id=run_id,
        statuses=statuses,
        status_counts=dict(status_counts),
        reason_counts_top10=reason_counts.most_common(10),
        spread_bps_avg=(sum(spreads) / len(spreads)) if spreads else 0.0,
        total_fee_bps_avg=(sum(fees) / len(fees)) if fees else 0.0,
        net_bps_avg=(sum(nets) / len(nets)) if nets else 0.0,
        net_bps_p10=percentile(nets, 0.10),
        net_bps_p50=percentile(nets, 0.50),
        net_bps_p90=percentile(nets, 0.90),
        net_positive_rate=net_positive_rate,
        estimated_pnl_raw_total=sum(pnl_raw_values),
        estimated_pnl_base_total=sum(pnl_base_values),
    )


def build_recommendations(summary: AggregateSummary, *, target_net_bps: float) -> list[str]:
    suggestions: list[str] = []
    if summary.matched_docs == 0:
        return ["No trades matched the filter. Expand --since-hours or remove filters."]

    current_min_spread = env_float("MIN_SPREAD_BPS", 5.0)
    suggested_min_spread = summary.total_fee_bps_avg + target_net_bps
    delta = suggested_min_spread - current_min_spread

    if summary.matched_docs < 30:
        suggestions.append("Samples are small (<30). Increase lookback window before final tuning.")

    if summary.net_bps_avg <= 0 or summary.net_positive_rate < 0.5:
        suggestions.append(
            (
                "Quality is weak. Increase min_spread_bps. "
                f"Current={current_min_spread:.3f}, suggested start={max(current_min_spread + 1.0, suggested_min_spread):.3f}"
            )
        )
    elif summary.net_positive_rate >= 0.7 and summary.net_bps_p10 > 0:
        suggestions.append(
            (
                "Quality is strong. You can test lowering min_spread_bps slightly for volume. "
                f"Current={current_min_spread:.3f}, trial={max(0.1, current_min_spread - 0.5):.3f}"
            )
        )
    else:
        direction = "increase" if delta > 0.25 else "keep"
        suggestions.append(
            (
                f"Target-net model says {direction} min_spread_bps. "
                f"Current={current_min_spread:.3f}, target={suggested_min_spread:.3f}"
            )
        )

    if summary.net_bps_p10 < -2.0:
        suggestions.append("Tail risk is high (p10 net_bps < -2). Reduce PAIR_BASE_AMOUNT or raise min_spread_bps.")

    if summary.matched_docs > 0 and summary.status_counts.get("dry_run", 0) == summary.matched_docs:
        suggestions.append("All rows are dry_run. Before production, implement real swap path in LiveOrderExecutor.")

    return suggestions


def write_csv(path: Path, rows: list[TradeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trade_id",
                "created_at",
                "run_id",
                "pair",
                "status",
                "reason",
                "spread_bps",
                "total_fee_bps",
                "required_spread_bps",
                "priority_fee_micro_lamports",
                "amount_in_raw",
                "net_bps",
                "est_pnl_raw",
                "est_pnl_base",
            ],
        )
        writer.writeheader()
        for row in rows:
            raw = asdict(row)
            created_at = raw.pop("created_at")
            raw["created_at"] = created_at.isoformat() if isinstance(created_at, datetime) else ""
            writer.writerow(raw)


def fetch_snapshots(collection: Any, *, limit: int) -> tuple[list[Any], str]:
    try:
        from google.cloud import firestore
    except ModuleNotFoundError:
        return list(collection.stream()), "full_scan"

    try:
        query = collection.order_by("created_at", direction=firestore.Query.DESCENDING)
        if limit > 0:
            query = query.limit(limit)
        return list(query.stream()), "ordered"
    except Exception as error:
        print(f"[warn] ordered query failed; fallback to full scan: {error}")
        return list(collection.stream()), "full_scan"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    if DOTENV_AVAILABLE:
        load_dotenv(env_path)
    else:
        fallback_load_env_file(env_path)

    args = parse_args()
    project_id = args.project_id.strip()
    if not project_id:
        raise ValueError("FIRESTORE_PROJECT_ID is required (set env or --project-id).")

    credentials_path = resolve_credentials_path(args.credentials, repo_root)
    if credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    try:
        from google.cloud import firestore
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "google-cloud-firestore is not installed. Run `pip install -r requirements.txt` first."
        ) from error

    now_utc = datetime.now(timezone.utc)
    cutoff = None
    if args.since_hours > 0:
        cutoff = now_utc - timedelta(hours=float(args.since_hours))

    client = firestore.Client(project=project_id)
    collection = (
        client.collection(args.bot_collection)
        .document(args.bot_id)
        .collection(args.trades_collection)
    )

    snapshots, query_mode = fetch_snapshots(collection, limit=int(args.limit))
    scanned_docs = 0

    rows: list[TradeRow] = []
    status_filter = {value.strip() for value in args.statuses if value.strip()}
    pair_filter = args.pair.strip()
    run_filter = args.run_id.strip()

    for snapshot in snapshots:
        scanned_docs += 1
        payload = snapshot.to_dict() or {}
        created_at = extract_created_at(payload, snapshot)
        if cutoff is not None and created_at is not None and created_at < cutoff:
            continue
        if pair_filter and str(payload.get("pair") or "") != pair_filter:
            continue
        if run_filter and str(payload.get("run_id") or "") != run_filter:
            continue
        status = str(payload.get("status") or "")
        if status_filter and status not in status_filter:
            continue

        rows.append(
            parse_trade_row(
                payload,
                trade_id=str(snapshot.id),
                created_at=created_at,
                base_amount_default=max(1, int(args.base_amount)),
                base_decimals=max(0, int(args.base_decimals)),
            )
        )

    rows.sort(
        key=lambda item: item.created_at or datetime.fromtimestamp(0, tz=timezone.utc),
        reverse=True,
    )
    if args.limit > 0 and len(rows) > args.limit:
        rows = rows[: args.limit]

    window_start = cutoff or datetime.fromtimestamp(0, tz=timezone.utc)
    summary = build_summary(
        rows=rows,
        scanned_docs=scanned_docs,
        window_start_utc=window_start,
        window_end_utc=now_utc,
        pair=pair_filter,
        run_id=run_filter,
        statuses=sorted(status_filter),
    )
    recommendations = build_recommendations(summary, target_net_bps=args.target_net_bps)

    if args.csv:
        write_csv((repo_root / args.csv).resolve(), rows)

    if args.json:
        output = {
            "summary": asdict(summary),
            "recommendations": recommendations,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    print(f"[info] project_id={project_id}")
    print(f"[info] query_mode={query_mode}")
    print(f"[info] scanned_docs={summary.scanned_docs} matched_docs={summary.matched_docs}")
    print(f"[info] window_start_utc={summary.window_start_utc}")
    print(f"[info] window_end_utc={summary.window_end_utc}")
    print(f"[info] pair_filter={summary.pair or '(all)'} run_filter={summary.run_id or '(all)'}")
    print(f"[info] status_filter={summary.statuses or ['(all)']}")
    print("")
    print("[summary]")
    print(
        f"  spread_bps_avg={summary.spread_bps_avg:.4f} total_fee_bps_avg={summary.total_fee_bps_avg:.4f} "
        f"net_bps_avg={summary.net_bps_avg:.4f}"
    )
    print(
        f"  net_bps_p10={summary.net_bps_p10:.4f} p50={summary.net_bps_p50:.4f} "
        f"p90={summary.net_bps_p90:.4f} positive_rate={summary.net_positive_rate:.2%}"
    )
    print(
        f"  estimated_pnl_raw_total={summary.estimated_pnl_raw_total:.6f} "
        f"estimated_pnl_base_total={summary.estimated_pnl_base_total:.10f}"
    )
    print("")
    print("[status_counts]")
    for key, value in sorted(summary.status_counts.items(), key=lambda item: item[0]):
        print(f"  {key}: {value}")
    print("")
    print("[reason_top10]")
    for reason, count in summary.reason_counts_top10:
        print(f"  {reason}: {count}")
    print("")
    print("[recommendations]")
    for recommendation in recommendations:
        print(f"  - {recommendation}")
    if args.csv:
        print("")
        print(f"[info] csv={str((repo_root / args.csv).resolve())}")


if __name__ == "__main__":
    main()
