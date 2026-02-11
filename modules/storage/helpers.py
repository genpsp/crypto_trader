from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_day_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def to_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def serialize_for_redis(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
