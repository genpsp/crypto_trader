from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

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


def log_event(
    logger: logging.Logger,
    *,
    level: str,
    event: str,
    message: str,
    **fields: Any,
) -> None:
    extra = {"event": _sanitize_value(event)}
    extra.update({key: _sanitize_value(value) for key, value in fields.items()})
    safe_message = _sanitize_text(message)

    if level == "debug":
        logger.debug(safe_message, extra=extra)
        return
    if level == "info":
        logger.info(safe_message, extra=extra)
        return
    if level == "warning":
        logger.warning(safe_message, extra=extra)
        return
    if level == "error":
        logger.error(safe_message, extra=extra)
        return
    if level == "critical":
        logger.critical(safe_message, extra=extra)
        return
    if level == "exception":
        logger.exception(safe_message, extra=extra)
        return

    logger.log(logging.INFO, safe_message, extra=extra)
