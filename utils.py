"""Utility helpers: logging setup, signal handlers, performance timers."""

from __future__ import annotations

import signal
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from typing import Generator, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Module-level shutdown event shared by all threads
# ---------------------------------------------------------------------------
shutdown_event: Event = Event()

# Unique session identifier generated once per process
SESSION_ID: str = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None, max_days: int = 7) -> None:
    """Configure loguru for console and optional file output.

    Args:
        log_level: Minimum log level string (e.g. "INFO", "DEBUG").
        log_file: Optional path for rotating file sink.
        max_days: Number of days to retain log files.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            level=log_level,
            rotation="1 day",
            retention=f"{max_days} days",
            compression="zip",
            enqueue=True,  # thread-safe
            serialize=False,
        )
        logger.info("File logging enabled → {}", log_file)


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def register_signal_handlers() -> None:
    """Register SIGINT / SIGTERM handlers that set the shutdown_event.

    Safe to call from the main thread only.
    """

    def _handler(signum: int, _frame: object) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received {} — initiating graceful shutdown …", sig_name)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Performance timing
# ---------------------------------------------------------------------------

class PerfTimer:
    """Simple wall-clock timer with millisecond resolution.

    Example:
        >>> timer = PerfTimer()
        >>> timer.start()
        >>> elapsed = timer.stop_ms()
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self._stop: float = 0.0

    def start(self) -> "PerfTimer":
        """Start the timer and return self for chaining."""
        self._start = time.perf_counter()
        return self

    def stop_ms(self) -> float:
        """Stop the timer and return elapsed milliseconds."""
        self._stop = time.perf_counter()
        return (self._stop - self._start) * 1000.0

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in ms without stopping the timer."""
        return (time.perf_counter() - self._start) * 1000.0


@contextmanager
def timed(label: str = "") -> Generator[PerfTimer, None, None]:
    """Context manager that logs elapsed time.

    Args:
        label: Optional label printed with the timing result.

    Yields:
        A started PerfTimer.
    """
    timer = PerfTimer().start()
    try:
        yield timer
    finally:
        ms = timer.stop_ms()
        if label:
            logger.debug("{} took {:.1f} ms", label, ms)
