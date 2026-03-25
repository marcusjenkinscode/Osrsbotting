"""Tests for ocr_processor.py"""

import numpy as np
import pytest

from ocr_processor import OCRProcessor, OCRResult
from config_manager import OCRSettings


def _make_frame(h: int = 130, w: int = 520, color: tuple = (0, 0, 0)) -> np.ndarray:
    """Create a solid-color BGR numpy array."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def _make_yellow_text_frame() -> np.ndarray:
    """Create a frame with a yellow (#FFFF00) rectangle to simulate OSRS text."""
    frame = _make_frame()  # black background
    # Paint a yellow region (BGR order: 0, 255, 255)
    frame[30:50, 20:100] = (0, 255, 255)
    return frame


class TestOCRResult:
    def test_is_usable_true(self):
        r = OCRResult(raw_text="Hello", confidence=0.9)
        assert r.is_usable

    def test_is_usable_empty_text(self):
        r = OCRResult(raw_text="   ", confidence=0.9)
        assert not r.is_usable

    def test_is_usable_zero_confidence(self):
        r = OCRResult(raw_text="Hello", confidence=0.0)
        assert not r.is_usable


class TestOCRProcessorPreprocessing:
    """Test the image preprocessing pipeline independently (no Tesseract needed)."""

    def test_preprocess_black_frame_returns_array(self):
        settings = OCRSettings(enable_easyocr=False)
        processor = OCRProcessor(settings)
        frame = _make_frame()
        result = processor._preprocess(frame)
        assert result is not None
        assert result.ndim == 2  # grayscale

    def test_preprocess_scales_up(self):
        settings = OCRSettings(enable_easyocr=False)
        processor = OCRProcessor(settings)
        h, w = 130, 520
        frame = _make_frame(h, w)
        result = processor._preprocess(frame)
        assert result is not None
        # Should be 3x the original dimensions
        assert result.shape == (h * 3, w * 3)

    def test_preprocess_none_frame(self):
        settings = OCRSettings(enable_easyocr=False)
        processor = OCRProcessor(settings)
        result = processor._preprocess(None)
        assert result is None

    def test_preprocess_empty_frame(self):
        settings = OCRSettings(enable_easyocr=False)
        processor = OCRProcessor(settings)
        result = processor._preprocess(np.array([]))
        assert result is None

    def test_yellow_mask_activates(self):
        """Yellow pixels should produce non-zero output after preprocessing."""
        import cv2

        settings = OCRSettings(enable_easyocr=False)
        processor = OCRProcessor(settings)
        frame = _make_yellow_text_frame()
        result = processor._preprocess(frame)
        assert result is not None
        # Some pixels should be white (255) from the yellow region
        assert result.max() == 255


class TestOCRProcessorFallback:
    """Ensure processor handles missing Tesseract gracefully."""

    def test_process_returns_none_without_tesseract(self, monkeypatch):
        """If Tesseract is missing and EasyOCR disabled, process() returns None."""
        settings = OCRSettings(enable_easyocr=False)
        processor = OCRProcessor(settings)
        # Force the availability flag to False
        processor._tesseract_available = False
        frame = _make_frame()
        result = processor.process(frame)
        assert result is None
