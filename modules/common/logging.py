from __future__ import annotations

import logging
from typing import Any


def log_event(
    logger: logging.Logger,
    *,
    level: str,
    event: str,
    message: str,
    **fields: Any,
) -> None:
    extra = {"event": event, **fields}

    if level == "debug":
        logger.debug(message, extra=extra)
        return
    if level == "info":
        logger.info(message, extra=extra)
        return
    if level == "warning":
        logger.warning(message, extra=extra)
        return
    if level == "error":
        logger.error(message, extra=extra)
        return
    if level == "critical":
        logger.critical(message, extra=extra)
        return
    if level == "exception":
        logger.exception(message, extra=extra)
        return

    logger.log(logging.INFO, message, extra=extra)
