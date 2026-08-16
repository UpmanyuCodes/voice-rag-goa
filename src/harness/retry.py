"""Resilience and Retry Management.

Implements exponential backoff, jitter, and fallback handlers using tenacity
to ensure high availability during network hiccups or rate limits.
"""

import logging
from typing import Callable, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


def create_retry_decorator(
    max_attempts: int = 3,
    min_seconds: float = 0.1,
    max_seconds: float = 1.0,
    retry_exceptions: tuple = (Exception,)
) -> Callable:
    """Creates a configured tenacity retry decorator with exponential backoff and jitter."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=min_seconds, max=max_seconds),
        retry=retry_if_exception_type(retry_exceptions),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
