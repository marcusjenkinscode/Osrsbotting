"""Tests for config_manager.py"""

import os
from pathlib import Path

import pytest

from config_manager import (
    AppSettings,
    CaptureSettings,
    FilterSettings,
    LoggingSettings,
    OCRSettings,
    WebhookSettings,
    get_settings,
)


def test_capture_settings_defaults():
    s = CaptureSettings()
    assert s.capture_fps == 0.5
    assert s.chat_region_width == 520
    assert s.chat_region_height == 130
    assert s.region["left"] == s.chat_region_left


def test_capture_settings_region_dict():
    s = CaptureSettings()
    r = s.region
    assert set(r.keys()) == {"left", "top", "width", "height"}


def test_capture_settings_resolution_str():
    s = CaptureSettings()
    assert "x" in s.resolution_str


def test_ocr_settings_defaults():
    s = OCRSettings()
    assert s.min_confidence == 0.75
    assert s.enable_easyocr is False


def test_ocr_settings_tesseract_path_none():
    s = OCRSettings(tesseract_path=None)
    assert s.tesseract_path is None


def test_ocr_settings_tesseract_path_str():
    s = OCRSettings(tesseract_path="/usr/bin/tesseract")
    assert isinstance(s.tesseract_path, Path)


def test_webhook_settings_defaults():
    s = WebhookSettings()
    assert s.batch_size == 5
    assert s.n8n_secret_header == "X-OSRS-Secret"


def test_webhook_settings_invalid_url():
    with pytest.raises(Exception):
        WebhookSettings(n8n_webhook_url="not-a-url")


def test_filter_settings_csv():
    s = FilterSettings(keyword_filters="buying,selling,wts")
    assert "buying" in s.keyword_filters
    assert "selling" in s.keyword_filters
    assert len(s.keyword_filters) == 3


def test_filter_settings_empty():
    s = FilterSettings(keyword_filters="")
    assert s.keyword_filters == []


def test_logging_settings_invalid_level():
    with pytest.raises(Exception):
        LoggingSettings(log_level="NOTVALID")


def test_logging_settings_valid_level():
    s = LoggingSettings(log_level="debug")
    assert s.log_level == "DEBUG"


def test_get_settings_returns_singleton():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_app_settings_has_sub_settings():
    s = AppSettings()
    assert isinstance(s.capture, CaptureSettings)
    assert isinstance(s.ocr, OCRSettings)
    assert isinstance(s.webhook, WebhookSettings)
    assert isinstance(s.filters, FilterSettings)
    assert isinstance(s.logging, LoggingSettings)
