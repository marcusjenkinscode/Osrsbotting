"""Tests for n8n_dispatcher.py"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chat_parser import ParsedMessage
from config_manager import CaptureSettings, WebhookSettings
from n8n_dispatcher import N8nDispatcher, OfflineQueue, WebhookPayload


# ---------------------------------------------------------------------------
# WebhookPayload
# ---------------------------------------------------------------------------

def _make_payload() -> WebhookPayload:
    msg = ParsedMessage(
        chat_type="public",
        author="TestUser",
        message="Hello world",
        raw_ocr="[TestUser] Hello world",
    )
    return WebhookPayload(
        message=msg,
        capture_time_ms=10.0,
        processing_time_ms=50.0,
        capture_settings=CaptureSettings(),
    )


class TestWebhookPayload:
    def test_to_dict_structure(self):
        p = _make_payload()
        d = p.to_dict()
        assert "timestamp" in d
        assert d["chat_type"] == "public"
        assert d["author"] == "TestUser"
        assert d["message"] == "Hello world"
        assert "game_window" in d
        assert "metadata" in d
        assert "session_id" in d["metadata"]

    def test_to_dict_resolution(self):
        p = _make_payload()
        d = p.to_dict()
        assert d["game_window"]["resolution"] == CaptureSettings().resolution_str

    def test_to_dict_region(self):
        p = _make_payload()
        d = p.to_dict()
        region = d["game_window"]["region_captured"]
        assert "left" in region and "width" in region


# ---------------------------------------------------------------------------
# OfflineQueue (SQLite)
# ---------------------------------------------------------------------------

class TestOfflineQueue:
    def test_push_and_pop(self, tmp_path):
        db = tmp_path / "test.db"
        oq = OfflineQueue(db_path=db)
        payload = {"message": "hello", "timestamp": "2024-01-01T00:00:00Z"}
        oq.push(payload)
        rows = oq.pop_batch(n=10)
        assert len(rows) == 1
        row_id, loaded, attempts = rows[0]
        assert loaded["message"] == "hello"
        assert attempts == 0

    def test_delete(self, tmp_path):
        db = tmp_path / "test.db"
        oq = OfflineQueue(db_path=db)
        oq.push({"msg": "test"})
        rows = oq.pop_batch()
        row_id = rows[0][0]
        oq.delete(row_id)
        assert oq.pop_batch() == []

    def test_increment_attempts(self, tmp_path):
        db = tmp_path / "test.db"
        oq = OfflineQueue(db_path=db)
        oq.push({"msg": "test"})
        rows = oq.pop_batch()
        row_id = rows[0][0]
        oq.increment_attempts(row_id)
        rows2 = oq.pop_batch()
        assert rows2[0][2] == 1  # attempts == 1

    def test_multiple_messages_order(self, tmp_path):
        db = tmp_path / "test.db"
        oq = OfflineQueue(db_path=db)
        for i in range(5):
            oq.push({"index": i})
            time.sleep(0.01)  # ensure ordering by created_at
        rows = oq.pop_batch(n=5)
        assert len(rows) == 5
        indices = [r[1]["index"] for r in rows]
        assert indices == sorted(indices)


# ---------------------------------------------------------------------------
# N8nDispatcher send logic
# ---------------------------------------------------------------------------

class TestN8nDispatcher:
    def test_enqueue_and_batch_flush(self, tmp_path):
        """Messages enqueued up to batch_size trigger a flush."""
        settings = WebhookSettings(
            n8n_webhook_url="http://localhost:5678/webhook/test",
            batch_size=2,
            batch_timeout_ms=60000,
        )
        dispatcher = N8nDispatcher(
            settings=settings,
            capture_settings=CaptureSettings(),
            db_path=tmp_path / "offline.db",
        )

        sent_batches = []

        async def _fake_post(url, headers, body):
            sent_batches.append(json.loads(body))
            return True

        dispatcher._async_post = _fake_post

        dispatcher.start()
        dispatcher.enqueue(_make_payload())
        dispatcher.enqueue(_make_payload())

        # Give the dispatcher thread time to flush
        time.sleep(1.5)
        dispatcher.stop()

        assert len(sent_batches) >= 1
        total_msgs = sum(len(b.get("messages", [])) for b in sent_batches)
        assert total_msgs == 2

    def test_failed_post_stores_offline(self, tmp_path):
        """When POST fails, messages are stored in the offline queue."""
        settings = WebhookSettings(
            n8n_webhook_url="http://localhost:5678/webhook/test",
            batch_size=1,
            batch_timeout_ms=500,
        )
        db_path = tmp_path / "offline.db"
        dispatcher = N8nDispatcher(
            settings=settings,
            capture_settings=CaptureSettings(),
            db_path=db_path,
        )

        async def _fail_post(url, headers, body):
            raise ConnectionError("n8n unreachable")

        dispatcher._async_post = _fail_post

        dispatcher.start()
        dispatcher.enqueue(_make_payload())
        time.sleep(3.0)  # wait for retries + store
        dispatcher.stop()

        oq = OfflineQueue(db_path=db_path)
        rows = oq.pop_batch()
        assert len(rows) >= 1
