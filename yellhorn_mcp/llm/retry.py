"""Retry utilities for provider API calls."""

import asyncio
import logging
from typing import Protocol, runtime_checkable

from google.api_core import exceptions as google_exceptions
from openai import RateLimitError
from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class HasCodeAndMessage(Protocol):
    code: int
    message: str


@runtime_checkable
class HasName(Protocol):
    __name__: str


def log_retry_attempt(retry_state: RetryCallState) -> None:
    if retry_state.outcome is None:
        return

    attempt = retry_state.attempt_number
    try:
        wt = retry_state.outcome_timestamp
        st = retry_state.start_time
        wait_time = (
            float(wt - st) if isinstance(wt, (int, float)) and isinstance(st, (int, float)) else 0.0
        )
    except Exception:
        wait_time = 0.0

    fn_name = "<unknown>"
    if retry_state.fn is not None:
        fn_name = (
            retry_state.fn.__name__
            if isinstance(retry_state.fn, HasName)
            else type(retry_state.fn).__name__
        )

    exc_text = ""
    try:
        exc = retry_state.outcome.exception()
        exc_text = str(exc) if exc is not None else ""
    except Exception:
        exc_text = ""

    logger.warning(
        f"Retrying {fn_name} after {wait_time:.1f} seconds (attempt {attempt}): {exc_text}"
    )


def is_retryable_error(exception: BaseException) -> bool:
    # Client errors exposing code/message (e.g., some Google client errors)
    if isinstance(exception, HasCodeAndMessage):
        msg = exception.message.lower()
        if exception.code == 429 or "resource_exhausted" in msg or "quota" in msg:
            return True

    # Standard retryable classes
    if isinstance(
        exception,
        (
            RateLimitError,
            google_exceptions.ResourceExhausted,
            google_exceptions.TooManyRequests,
            ConnectionError,
            asyncio.TimeoutError,
        ),
    ):
        return True

    # Message matching
    msg = str(exception).lower()
    if any(
        term in msg for term in ("resource_exhausted", "quota", "rate limit", "too many requests")
    ):
        return True

    return False


api_retry = retry(
    retry=retry_if_exception(is_retryable_error),
    wait=wait_exponential(multiplier=1, min=4, max=60, exp_base=2),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
