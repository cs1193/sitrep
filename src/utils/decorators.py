"""Cross-cutting decorators: logging, retry, timing, error handling, memoization."""
from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Tuple, Type, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
_logger = logging.getLogger("sitrep.decorators")


def log_execution(func: F) -> F:
    """Log entry, exit, and exceptions of *func* at DEBUG."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _logger.debug("→ %s", func.__qualname__)
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            _logger.debug("← %s (%.3fs)", func.__qualname__, time.perf_counter() - start)
            return result
        except Exception:
            _logger.exception("✗ %s", func.__qualname__)
            raise

    return wrapper  # type: ignore[return-value]


def measure_time(metric_name: str = "") -> Callable[[F], F]:
    """Time *func* and record the duration via the metrics collector (if present)."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                name = metric_name or f"duration.{func.__name__}"
                try:
                    from src.infrastructure.monitoring.metrics import get_metrics

                    get_metrics().observe(name, elapsed)
                except Exception:  # pragma: no cover - metrics optional
                    pass
                _logger.debug("%s took %.3fs", func.__qualname__, elapsed)

        return wrapper  # type: ignore[return-value]

    return decorator


def retry(
    tries: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> Callable[[F], F]:
    """Retry *func* on *exceptions* with exponential backoff."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt, current_delay = 0, delay
            while True:
                attempt += 1
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt >= tries:
                        _logger.error("retry exhausted for %s: %s", func.__qualname__, exc)
                        raise
                    _logger.warning(
                        "retry %d/%d for %s: %s", attempt, tries, func.__qualname__, exc
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper  # type: ignore[return-value]

    return decorator


def handle_errors(default: Any = None, log: bool = True) -> Callable[[F], F]:
    """Swallow exceptions from *func*, returning *default* (logging when *log*)."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if log:
                    _logger.exception("handled error in %s: %s", func.__qualname__, exc)
                return default

        return wrapper  # type: ignore[return-value]

    return decorator


def memoize(func: F) -> F:
    """Memoize *func* on its positional + keyword arguments (hashable only)."""
    cache: dict = {}

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    def cache_clear() -> None:  # pragma: no cover - helper
        cache.clear()

    wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]
