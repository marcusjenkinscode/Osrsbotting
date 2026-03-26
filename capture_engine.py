"""Screen capture engine using MSS for high-performance region capture.

Features
--------
* Captures only the OSRS chat region — not the full screen.
* ROI pixel-diff check: skips OCR if nothing changed since last frame.
* Randomised capture intervals with ±jitter to mimic human behaviour.
* Producer thread puts raw PIL images into a ``queue.Queue``.
"""

from __future__ import annotations

import queue
import random
import time
from threading import Thread
from typing import Optional, Tuple

import mss
import numpy as np
from loguru import logger
from PIL import Image

from config_manager import CaptureSettings
from utils import PerfTimer, shutdown_event

# Pixel difference threshold below which a frame is considered unchanged.
_PIXEL_DIFF_THRESHOLD = 50

# Maximum number of frames buffered in the queue before dropping oldest.
_QUEUE_MAXSIZE = 10


class CaptureEngine:
    """Producer thread: captures the OSRS chat region at a configurable rate.

    Args:
        settings: Capture configuration (region, FPS, …).
        frame_queue: Queue into which raw BGR numpy arrays are placed.
        jitter: Maximum random jitter added to the capture interval (seconds).

    Example:
        >>> import queue
        >>> q = queue.Queue()
        >>> engine = CaptureEngine(CaptureSettings(), q)
        >>> engine.start()
        >>> engine.stop()
    """

    def __init__(
        self,
        settings: CaptureSettings,
        frame_queue: "queue.Queue[np.ndarray]",
        jitter: float = 0.5,
    ) -> None:
        import threading as _threading

        self._settings = settings
        self._queue = frame_queue
        self._jitter = jitter
        self._thread: Optional[Thread] = None
        self._last_frame: Optional[np.ndarray] = None
        self._stop_event = _threading.Event()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background capture thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("CaptureEngine is already running.")
            return
        self._thread = Thread(target=self._run, name="capture-thread", daemon=True)
        self._thread.start()
        logger.info("CaptureEngine started (region={})", self._settings.region)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the capture thread to stop and wait for it to finish.

        Args:
            timeout: Seconds to wait for the thread to join.
        """
        self._stop_event.set()
        shutdown_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("CaptureEngine stopped.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main loop executed in the background thread."""
        interval = 1.0 / max(self._settings.capture_fps, 0.01)

        try:
            with mss.mss() as sct:
                region = self._settings.region
                logger.debug("MSS capture region: {}", region)

                while not self._stop_event.is_set() and not shutdown_event.is_set():
                    timer = PerfTimer().start()

                    frame = self._capture_frame(sct, region)
                    if frame is not None:
                        if self._has_changed(frame):
                            self._enqueue(frame)
                        else:
                            logger.trace("Frame unchanged — skipping OCR")
                        self._last_frame = frame

                    elapsed = timer.stop_ms() / 1000.0
                    sleep_time = max(0.0, interval - elapsed + self._random_jitter())
                    time.sleep(sleep_time)

        except Exception:
            logger.exception("CaptureEngine encountered an unhandled error")
        finally:
            logger.debug("CaptureEngine thread exiting")

    @staticmethod
    def _capture_frame(sct: mss.mss, region: dict) -> Optional[np.ndarray]:
        """Capture a single frame.

        Args:
            sct: Active MSS context.
            region: MSS-compatible region dict.

        Returns:
            BGR numpy array or None on error.
        """
        try:
            shot = sct.grab(region)
            # MSS returns BGRA; drop alpha channel → BGR
            frame = np.array(shot)[:, :, :3]
            return frame
        except mss.exception.ScreenShotError as exc:
            logger.warning("Screen capture failed (monitor disconnect?): {}", exc)
            return None
        except Exception:
            logger.exception("Unexpected error during frame capture")
            return None

    def _has_changed(self, frame: np.ndarray) -> bool:
        """Return True if *frame* differs enough from the previous frame.

        Args:
            frame: Current BGR frame.

        Returns:
            True when the frame differs above the pixel-diff threshold.
        """
        if self._last_frame is None:
            return True
        if frame.shape != self._last_frame.shape:
            return True
        diff = np.mean(np.abs(frame.astype(np.int16) - self._last_frame.astype(np.int16)))
        return float(diff) > _PIXEL_DIFF_THRESHOLD

    def _enqueue(self, frame: np.ndarray) -> None:
        """Put *frame* on the queue; drop the oldest entry if full.

        Args:
            frame: BGR numpy array to enqueue.
        """
        if self._queue.full():
            try:
                self._queue.get_nowait()
                logger.debug("Frame queue full — dropped oldest frame")
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            pass  # extremely unlikely after the get_nowait above

    def _random_jitter(self) -> float:
        """Return a small random jitter value within ±self._jitter seconds."""
        return random.uniform(-self._jitter, self._jitter)


def frame_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert a BGR numpy array to a PIL RGB image.

    Args:
        frame: BGR numpy array from MSS.

    Returns:
        PIL Image in RGB mode.
    """
    import cv2  # local import keeps module importable without opencv

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
