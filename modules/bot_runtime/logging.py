from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

STANDARD_LOG_FIELDS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
}

URL_TOKEN_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
API_KEY_ASSIGNMENT_RE = re.compile(r"(?i)(api[-_]?key\s*[:=]\s*)([^\s,;\"'&]+)")
API_KEY_QUERY_RE = re.compile(r"(?i)([?&](?:api[-_]?key)=)([^&#\s]+)")


def _sanitize_url_token(token: str) -> str:
    candidate = token
    trailing = ""
    while candidate and candidate[-1] in ".,);]}":
        trailing = candidate[-1] + trailing
        candidate = candidate[:-1]

    parsed = urlsplit(candidate)
    if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
        candidate = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    return f"{candidate}{trailing}"


def _sanitize_text(value: str) -> str:
    masked = URL_TOKEN_RE.sub(lambda match: _sanitize_url_token(match.group(0)), value)
    masked = API_KEY_QUERY_RE.sub(r"\1***", masked)
    masked = API_KEY_ASSIGNMENT_RE.sub(r"\1***", masked)
    return masked


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, dict):
        return {key: _sanitize_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_value(item) for item in value)
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": _sanitize_text(record.getMessage()),
        }

        for key, value in record.__dict__.items():
            if key in STANDARD_LOG_FIELDS or key.startswith("_"):
                continue
            if key in {"message", "asctime"}:
                continue
            payload[key] = _sanitize_value(value)

        if record.exc_info:
            payload["exception"] = _sanitize_text(self.formatException(record.exc_info))

        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("solana_bot")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False

    return logger
