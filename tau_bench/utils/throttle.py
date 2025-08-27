# Copyright Sierra

import time
import threading
from typing import Callable, TypeVar, Any
from functools import wraps
from litellm import completion

T = TypeVar("T")

# Global lock and last request time for throttling
_throttle_lock = threading.Lock()
_last_request_time = 0.0


def throttled_completion(
    delay_seconds: float = 0.5,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that adds throttling to API calls to prevent rate limiting.

    Args:
        delay_seconds: Minimum time to wait between API calls

    Returns:
        Decorated function with throttling
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            global _last_request_time

            with _throttle_lock:
                current_time = time.time()
                time_since_last = current_time - _last_request_time

                if time_since_last < delay_seconds:
                    sleep_time = delay_seconds - time_since_last
                    time.sleep(sleep_time)

                _last_request_time = time.time()

            return func(*args, **kwargs)

        return wrapper

    return decorator


def create_throttled_completion(delay_seconds: float = 0.5):
    """
    Create a throttled version of litellm.completion.

    Args:
        delay_seconds: Minimum time to wait between API calls

    Returns:
        Throttled completion function
    """

    @throttled_completion(delay_seconds=delay_seconds)
    def throttled_litellm_completion(*args, **kwargs):
        return completion(*args, **kwargs)

    return throttled_litellm_completion
