"""Timing utilities for performance monitoring."""
import functools
import time
from typing import Callable, TypeVar, ParamSpec, Optional

from loguru import logger

P = ParamSpec("P")
R = TypeVar("R")


def linetimer(func: Optional[Callable[P, R]] = None) -> Callable:
    """
    Decorator to measure and log execution time of a function.
    Can be used with or without parentheses.

    Usage:
        @linetimer
        def my_function():
            pass

        @linetimer()
        def my_function():
            pass

    Args:
        func: The function to be timed (optional)

    Returns:
        Wrapped function that logs its execution time
    """
    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                logger.debug(
                    f"{f.__module__}.{f.__name__} took {elapsed_time:.4f} seconds"
                )
        return wrapper

    if func is None:
        # Called with parentheses: @linetimer()
        return decorator
    else:
        # Called without parentheses: @linetimer
        return decorator(func)
