"""n8n webhook dispatcher with batching, retry/backoff, and SQLite offline queue.

Architecture
------------
* A background **dispatcher thread** drains the outbound queue.
* Messages are accumulated until ``batch_size`` is reached or
  ``batch_timeout_ms`` elapses, then sent in a single HTTP POST.
* If the POST fails, messages are persisted to a local SQLite database and
  retried with exponential back-off (max 3 attempts).
* A periodic **health-check** pings the webhook every 60 s.
"""

from __future__ import annotations

import asyncio
import json
import queue
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger

from chat_parser import ParsedMessage
from config_manager import CaptureSettings, WebhookSettings
from utils import SESSION_ID, PerfTimer, shutdown_event

# ---------------------------------------------------------------------------
# Payload model
# ---------------------------------------------------------------------------


@dataclass
class WebhookPayload:
    """A single chat message payload as expected by the n8n workflow.

    Args:
        message: Parsed OSRS chat message.
        capture_time_ms: Time taken to capture the frame.
        processing_time_ms: Time taken to process (OCR) the frame.
        capture_settings: Capture configuration (used for resolution metadata).
    """

    message: ParsedMessage
    capture_time_ms: float = 0.0
    processing_time_ms: float = 0.0
    capture_settings: Optional[CaptureSettings] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dict matching the spec payload."""
        msg = self.message
        cs = self.capture_settings
        return {
            "timestamp": datetime.fromtimestamp(msg.timestamp, tz=timezone.utc).isoformat(),
            "chat_type": msg.chat_type,
            "author": msg.author,
            "message": msg.message,
            "raw_ocr": msg.raw_ocr,
            "confidence": 1.0,  # populated externally if available
            "game_window": {
                "resolution": cs.resolution_str if cs else "unknown",
                "client_mode": "resizable",
                "region_captured": cs.region if cs else {},
            },
            "metadata": {
                "capture_time_ms": self.capture_time_ms,
                "processing_time_ms": self.processing_time_ms,
                "session_id": SESSION_ID,
            },
        }


# ---------------------------------------------------------------------------
# SQLite offline queue
# ---------------------------------------------------------------------------

_DB_PATH = Path("./data/offline_queue.db")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pending_messages (
    id          TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    created_at  REAL NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 0
);
"""


@contextmanager
def _db_connection(path: Path):  # type: ignore[return]
    """Yield an open SQLite connection, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()
        yield conn
    finally:
        conn.close()


class OfflineQueue:
    """Persist unsent payloads to SQLite for later retry.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self._db_path = db_path

    def push(self, payload_dict: Dict[str, Any]) -> None:
        """Persist a single payload dict.

        Args:
            payload_dict: JSON-serialisable payload.
        """
        row_id = str(uuid.uuid4())
        with _db_connection(self._db_path) as conn:
            conn.execute(
                "INSERT INTO pending_messages (id, payload_json, created_at, attempts) "
                "VALUES (?, ?, ?, 0)",
                (row_id, json.dumps(payload_dict), time.time()),
            )
            conn.commit()
        logger.debug("Offline queue: stored payload id={}", row_id)

    def pop_batch(self, n: int = 20) -> List[tuple]:
        """Return up to *n* pending rows (id, payload_json, attempts).

        Args:
            n: Maximum batch size.

        Returns:
            List of (id, payload_dict, attempts) tuples.
        """
        with _db_connection(self._db_path) as conn:
            rows = conn.execute(
                "SELECT id, payload_json, attempts FROM pending_messages "
                "ORDER BY created_at ASC LIMIT ?",
                (n,),
            ).fetchall()
        return [(r[0], json.loads(r[1]), r[2]) for r in rows]

    def delete(self, row_id: str) -> None:
        """Remove a successfully delivered row.

        Args:
            row_id: UUID of the row to delete.
        """
        with _db_connection(self._db_path) as conn:
            conn.execute("DELETE FROM pending_messages WHERE id = ?", (row_id,))
            conn.commit()

    def increment_attempts(self, row_id: str) -> None:
        """Increment the retry counter for a row.

        Args:
            row_id: UUID of the row to update.
        """
        with _db_connection(self._db_path) as conn:
            conn.execute(
                "UPDATE pending_messages SET attempts = attempts + 1 WHERE id = ?", (row_id,)
            )
            conn.commit()


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_BACKOFF_BASE_S = 2.0  # 2, 4, 8 seconds


class N8nDispatcher:
    """Sends batched chat messages to an n8n webhook.

    Args:
        settings: Webhook configuration.
        capture_settings: Capture configuration for metadata.
        db_path: Path for the offline SQLite queue.

    Example:
        >>> dispatcher = N8nDispatcher(WebhookSettings(), CaptureSettings())
        >>> dispatcher.start()
        >>> dispatcher.enqueue(payload)
        >>> dispatcher.stop()
    """

    def __init__(
        self,
        settings: WebhookSettings,
        capture_settings: Optional[CaptureSettings] = None,
        db_path: Path = _DB_PATH,
    ) -> None:
        import threading as _threading

        self._settings = settings
        self._capture_settings = capture_settings
        self._offline_queue = OfflineQueue(db_path)
        self._batch: List[Dict[str, Any]] = []
        self._inbound: "queue.Queue[Optional[WebhookPayload]]" = queue.Queue()
        self._thread: Optional[Thread] = None
        self._last_batch_flush = time.time()
        self._stop_event = _threading.Event()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background dispatcher thread."""
        self._thread = Thread(target=self._run, name="dispatcher-thread", daemon=True)
        self._thread.start()
        logger.info("N8nDispatcher started → {}", self._settings.n8n_webhook_url)

    def stop(self, timeout: float = 10.0) -> None:
        """Flush pending messages and stop the thread.

        Args:
            timeout: Seconds to wait for the thread to join.
        """
        self._stop_event.set()
        self._inbound.put(None)  # sentinel
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("N8nDispatcher stopped.")

    def enqueue(self, payload: WebhookPayload) -> None:
        """Add a payload to the outbound queue.

        Args:
            payload: Chat message payload to send.
        """
        self._inbound.put(payload)

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Dispatcher loop: accumulate a batch, flush when ready."""
        batch_timeout_s = self._settings.batch_timeout_ms / 1000.0
        health_interval = self._settings.health_check_interval_s
        offline_retry_interval = 30.0
        last_health = time.time()
        last_offline_retry = time.time()

        asyncio.set_event_loop(asyncio.new_event_loop())

        while not self._stop_event.is_set() and not shutdown_event.is_set():
            # Drain inbox into local batch
            self._drain_inbox()

            now = time.time()
            batch_full = len(self._batch) >= self._settings.batch_size
            batch_timeout = (now - self._last_batch_flush) >= batch_timeout_s

            if (batch_full or (batch_timeout and self._batch)):
                self._flush_batch()

            # Health check
            if now - last_health >= health_interval:
                self._health_check()
                last_health = now

            # Retry offline queue periodically
            if now - last_offline_retry >= offline_retry_interval:
                self._retry_offline()
                last_offline_retry = now

            time.sleep(0.1)

        # Final flush
        self._drain_inbox()
        if self._batch:
            self._flush_batch()

    def _drain_inbox(self) -> None:
        """Move items from the inbound queue into the local batch."""
        try:
            while True:
                item = self._inbound.get_nowait()
                if item is None:
                    self._stop_event.set()
                    return
                self._batch.append(item.to_dict())
        except queue.Empty:
            pass

    def _flush_batch(self) -> None:
        """Send the current batch to n8n."""
        if not self._batch:
            return
        batch = list(self._batch)
        self._batch.clear()
        self._last_batch_flush = time.time()

        success = self._send_with_retry(batch)
        if not success:
            logger.warning("Batch send failed — persisting {} messages to offline queue", len(batch))
            for payload in batch:
                self._offline_queue.push(payload)

    def _send_with_retry(self, batch: List[Dict[str, Any]]) -> bool:
        """Send *batch* with exponential back-off.

        Args:
            batch: List of payload dicts to POST.

        Returns:
            True on success, False after all retries exhausted.
        """
        headers = {
            "Content-Type": "application/json",
            self._settings.n8n_secret_header: self._settings.n8n_secret_value,
        }
        body = json.dumps({"messages": batch})

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    self._async_post(self._settings.n8n_webhook_url, headers, body)
                )
                loop.close()
                if result:
                    logger.debug(
                        "Batch of {} sent on attempt {}", len(batch), attempt
                    )
                    return True
            except Exception:
                logger.exception("HTTP POST attempt {} failed", attempt)
            if attempt < _MAX_RETRIES:
                backoff = _BACKOFF_BASE_S ** attempt
                logger.debug("Retrying in {:.1f}s …", backoff)
                time.sleep(backoff)

        return False

    async def _async_post(self, url: str, headers: dict, body: str) -> bool:
        """Perform a non-blocking HTTP POST.

        Args:
            url: Target URL.
            headers: HTTP headers.
            body: JSON-encoded request body.

        Returns:
            True if the server returned 2xx.
        """
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, data=body, headers=headers) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    logger.warning("n8n returned HTTP {}: {}", resp.status, text[:200])
                    return False
                return True

    def _health_check(self) -> None:
        """Ping n8n to verify connectivity."""
        try:
            loop = asyncio.new_event_loop()
            ok = loop.run_until_complete(
                self._async_post(
                    self._settings.n8n_webhook_url,
                    {
                        "Content-Type": "application/json",
                        self._settings.n8n_secret_header: self._settings.n8n_secret_value,
                    },
                    json.dumps({"health_check": True, "session_id": SESSION_ID}),
                )
            )
            loop.close()
            logger.info("n8n health check: {}", "OK" if ok else "FAILED")
        except Exception:
            logger.warning("n8n health check failed")

    def _retry_offline(self) -> None:
        """Attempt to re-send messages from the offline SQLite queue."""
        rows = self._offline_queue.pop_batch(n=20)
        if not rows:
            return
        logger.info("Retrying {} offline messages …", len(rows))
        for row_id, payload_dict, attempts in rows:
            if attempts >= _MAX_RETRIES:
                logger.warning("Dropping offline message {} after {} attempts", row_id, attempts)
                self._offline_queue.delete(row_id)
                continue
            success = self._send_with_retry([payload_dict])
            if success:
                self._offline_queue.delete(row_id)
            else:
                self._offline_queue.increment_attempts(row_id)
