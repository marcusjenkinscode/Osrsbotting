"""OCR processor for OSRS chat text.

Pipeline
--------
1. Receive BGR frame from the capture engine.
2. Convert BGR → HSV.
3. Build colour masks for yellow (#FFFF00) and white (#FFFFFF) OSRS text.
4. Combine masks and apply to isolate text pixels.
5. Resize 3× with INTER_CUBIC interpolation.
6. Grayscale + Otsu thresholding.
7. Morphological denoising.
8. Run Tesseract (or EasyOCR) with OSRS-tuned config.
9. Return raw text + confidence score.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from config_manager import OCRSettings

# ---------------------------------------------------------------------------
# HSV bounds for text colour masks
# ---------------------------------------------------------------------------
# Yellow (#FFFF00) in HSV:  hue ≈ 30 (OpenCV 0-180), sat high, val high
_YELLOW_LOWER = np.array([20, 150, 150], dtype=np.uint8)
_YELLOW_UPPER = np.array([40, 255, 255], dtype=np.uint8)

# White (#FFFFFF) in HSV: low saturation, high value
_WHITE_LOWER = np.array([0, 0, 180], dtype=np.uint8)
_WHITE_UPPER = np.array([180, 60, 255], dtype=np.uint8)

# Tesseract config optimised for OSRS pixelated font
_TESS_CONFIG = (
    "--psm 6 --oem 3 "
    "-c tessedit_char_whitelist="
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "0123456789 :.,!?'\"[]()-_/@#"
)

# Scale factor applied before Tesseract
_SCALE = 3


@dataclass
class OCRResult:
    """Result of a single OCR pass.

    Attributes:
        raw_text: Raw string returned by the OCR engine.
        confidence: Normalised confidence in [0, 1].
        engine: Name of the OCR engine used ("tesseract" or "easyocr").
        processing_time_ms: Time taken by the OCR step in milliseconds.
    """

    raw_text: str
    confidence: float
    engine: str = "tesseract"
    processing_time_ms: float = 0.0

    @property
    def is_usable(self) -> bool:
        """Return True when text is non-empty and confidence is positive."""
        return bool(self.raw_text.strip()) and self.confidence > 0


class OCRProcessor:
    """Converts raw BGR frames into OCR results.

    Args:
        settings: OCR configuration.

    Example:
        >>> processor = OCRProcessor(OCRSettings())
        >>> result = processor.process(frame)
        >>> print(result.raw_text)
    """

    def __init__(self, settings: OCRSettings) -> None:
        self._settings = settings
        self._tesseract_available: bool = self._check_tesseract()
        self._easyocr_reader: Optional[object] = None

        if settings.enable_easyocr:
            self._easyocr_reader = self._init_easyocr()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> Optional[OCRResult]:
        """Run the full preprocessing + OCR pipeline on *frame*.

        Args:
            frame: BGR numpy array from the capture engine.

        Returns:
            OCRResult or None if preprocessing or OCR fails.
        """
        import time

        t0 = time.perf_counter()

        try:
            preprocessed = self._preprocess(frame)
            if preprocessed is None:
                return None
        except Exception:
            logger.exception("Image preprocessing failed — skipping frame")
            return None

        elapsed_pre = (time.perf_counter() - t0) * 1000.0

        if self._settings.enable_easyocr and self._easyocr_reader is not None:
            result = self._run_easyocr(preprocessed)
        elif self._tesseract_available:
            result = self._run_tesseract(preprocessed)
        else:
            logger.error(
                "No OCR engine available.  Install Tesseract or set ENABLE_EASYOCR=true."
            )
            return None

        if result is not None:
            result.processing_time_ms += elapsed_pre

        return result

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Return a binary image ready for Tesseract.

        Args:
            frame: BGR input frame.

        Returns:
            Grayscale thresholded image or None on failure.
        """
        if frame is None or frame.size == 0:
            return None

        # Step 1 — BGR → HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Step 2 — colour masks
        yellow_mask = cv2.inRange(hsv, _YELLOW_LOWER, _YELLOW_UPPER)
        white_mask = cv2.inRange(hsv, _WHITE_LOWER, _WHITE_UPPER)
        combined_mask = cv2.bitwise_or(yellow_mask, white_mask)

        # Step 3 — isolate text pixels (white text on black background)
        result = np.zeros_like(frame[:, :, 0])
        result[combined_mask > 0] = 255

        # Step 4 — upscale 3× with cubic interpolation
        h, w = result.shape
        upscaled = cv2.resize(result, (w * _SCALE, h * _SCALE), interpolation=cv2.INTER_CUBIC)

        # Step 5 — Otsu binarisation
        _, binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 6 — morphological noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        return cleaned

    # ------------------------------------------------------------------
    # OCR engines
    # ------------------------------------------------------------------

    def _run_tesseract(self, image: np.ndarray) -> Optional[OCRResult]:
        """Run Tesseract on a preprocessed image.

        Args:
            image: Grayscale binary image.

        Returns:
            OCRResult or None on failure.
        """
        import time

        try:
            import pytesseract
        except ImportError:
            logger.error("pytesseract is not installed.")
            return None

        if self._settings.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = str(self._settings.tesseract_path)

        t0 = time.perf_counter()
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self._settings.ocr_lang,
                config=_TESS_CONFIG,
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            logger.exception("Tesseract OCR failed")
            return None
        elapsed = (time.perf_counter() - t0) * 1000.0

        words = []
        confidences = []
        for txt, conf in zip(data["text"], data["conf"]):
            conf_int = int(conf)
            if conf_int > 0 and txt.strip():
                words.append(txt)
                confidences.append(conf_int)

        raw_text = " ".join(words)
        avg_conf = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0

        return OCRResult(
            raw_text=raw_text,
            confidence=avg_conf,
            engine="tesseract",
            processing_time_ms=elapsed,
        )

    def _run_easyocr(self, image: np.ndarray) -> Optional[OCRResult]:
        """Run EasyOCR on a preprocessed image.

        Args:
            image: Grayscale binary image.

        Returns:
            OCRResult or None on failure.
        """
        import time

        if self._easyocr_reader is None:
            return None

        t0 = time.perf_counter()
        try:
            results = self._easyocr_reader.readtext(image, detail=1)  # type: ignore[attr-defined]
        except Exception:
            logger.exception("EasyOCR failed")
            return None
        elapsed = (time.perf_counter() - t0) * 1000.0

        texts = []
        confidences = []
        for (_bbox, text, prob) in results:
            if text.strip():
                texts.append(text)
                confidences.append(float(prob))

        raw_text = " ".join(texts)
        avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0

        return OCRResult(
            raw_text=raw_text,
            confidence=avg_conf,
            engine="easyocr",
            processing_time_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_tesseract(self) -> bool:
        """Verify that the Tesseract binary is accessible.

        Returns:
            True if Tesseract is available.
        """
        binary = str(self._settings.tesseract_path) if self._settings.tesseract_path else "tesseract"
        found = shutil.which(binary) is not None
        if not found:
            logger.warning(
                "Tesseract binary '{}' not found on PATH.  "
                "Install Tesseract or set TESSERACT_PATH in .env.",
                binary,
            )
        return found

    def _init_easyocr(self) -> Optional[object]:
        """Attempt to import and initialise EasyOCR.

        Returns:
            EasyOCR Reader instance or None if unavailable.
        """
        try:
            import easyocr  # type: ignore[import]

            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            logger.info("EasyOCR initialised successfully.")
            return reader
        except ImportError:
            logger.warning("EasyOCR not installed.  Falling back to Tesseract.")
            return None
        except Exception:
            logger.exception("EasyOCR initialisation failed")
            return None
