from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, TypeVar

from .logging import log_event

T = TypeVar("T")


async def guarded_call(
    action: Callable[[], Awaitable[T]],
    *,
    logger: logging.Logger,
    event: str,
    message: str,
    level: str = "warning",
    default: T | None = None,
    reraise: bool = False,
    **fields: Any,
) -> T | None:
    try:
        return await action()
    except asyncio.CancelledError:
        raise
    except Exception as error:
        log_event(
            logger,
            level=level,
            event=event,
            message=message,
            error=str(error),
            **fields,
        )
        if reraise:
            raise
        return default
