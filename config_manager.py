"""Configuration manager for the OSRS Chat Logger.

Loads settings from environment variables / .env file using Pydantic Settings.
All configuration is sourced from the environment — no secrets in code.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CaptureSettings(BaseSettings):
    """Screen-capture and chat-region configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    capture_fps: float = Field(default=0.5, ge=0.1, le=10.0, description="Captures per second")
    chat_region_left: int = Field(default=10, ge=0, description="Chat box left edge (px)")
    chat_region_top: int = Field(default=850, ge=0, description="Chat box top edge (px)")
    chat_region_width: int = Field(default=520, ge=50, description="Chat box width (px)")
    chat_region_height: int = Field(default=130, ge=20, description="Chat box height (px)")
    calibration_mode: bool = Field(default=False, description="Enable live calibration window")

    @property
    def region(self) -> dict:
        """Return MSS-compatible region dict."""
        return {
            "left": self.chat_region_left,
            "top": self.chat_region_top,
            "width": self.chat_region_width,
            "height": self.chat_region_height,
        }

    @property
    def resolution_str(self) -> str:
        """Human-readable capture resolution."""
        return f"{self.chat_region_width}x{self.chat_region_height}"


class OCRSettings(BaseSettings):
    """OCR engine configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    tesseract_path: Optional[Path] = Field(
        default=None, description="Path to Tesseract binary (None = auto-detect)"
    )
    ocr_lang: str = Field(default="eng", description="Tesseract language code")
    enable_easyocr: bool = Field(default=False, description="Use EasyOCR as fallback/primary")
    min_confidence: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Minimum OCR confidence threshold"
    )

    @field_validator("tesseract_path", mode="before")
    @classmethod
    def _parse_tesseract_path(cls, v: object) -> Optional[Path]:
        if v is None or v == "":
            return None
        return Path(str(v))


class WebhookSettings(BaseSettings):
    """n8n webhook configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    n8n_webhook_url: str = Field(
        default="https://n8n.example.com/webhook/osrs-chat",
        description="Target n8n webhook URL",
    )
    n8n_secret_header: str = Field(
        default="X-OSRS-Secret", description="Authentication header name"
    )
    n8n_secret_value: str = Field(default="", description="Authentication header value")
    batch_size: int = Field(default=5, ge=1, le=100, description="Messages per webhook batch")
    batch_timeout_ms: int = Field(
        default=10000, ge=100, description="Max wait time before flushing batch (ms)"
    )
    health_check_interval_s: int = Field(
        default=60, ge=10, description="Seconds between n8n health pings"
    )

    @field_validator("n8n_webhook_url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("n8n_webhook_url must start with http:// or https://")
        return v


class FilterSettings(BaseSettings):
    """Chat message filter configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    keyword_filters: List[str] = Field(
        default_factory=list,
        description="Only forward messages containing these keywords (empty = all)",
    )
    ignore_players: List[str] = Field(
        default_factory=list, description="Ignore messages from these usernames"
    )

    @field_validator("keyword_filters", "ignore_players", mode="before")
    @classmethod
    def _split_csv(cls, v: object) -> List[str]:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        if isinstance(v, list):
            return v
        return []


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Path = Field(default=Path("./logs/osrs_chat.log"), description="Log file path")
    max_log_days: int = Field(default=7, ge=1, description="Log retention in days")

    @field_validator("log_level")
    @classmethod
    def _validate_level(cls, v: str) -> str:
        valid = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper


class AppSettings(BaseSettings):
    """Root application settings aggregating all sub-settings.

    Example:
        >>> settings = AppSettings()
        >>> print(settings.capture.capture_fps)
        0.5
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    capture: CaptureSettings = Field(default_factory=CaptureSettings)
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    webhook: WebhookSettings = Field(default_factory=WebhookSettings)
    filters: FilterSettings = Field(default_factory=FilterSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @model_validator(mode="after")
    def _propagate_env(self) -> "AppSettings":
        """Re-initialise sub-settings so they each pick up the .env file."""
        self.capture = CaptureSettings()
        self.ocr = OCRSettings()
        self.webhook = WebhookSettings()
        self.filters = FilterSettings()
        self.logging = LoggingSettings()
        return self


_settings: Optional[AppSettings] = None


def get_settings(env_file: Optional[Path] = None) -> AppSettings:
    """Return a cached AppSettings instance.

    Args:
        env_file: Optional path to a custom .env file.  If provided the cache
            is refreshed.

    Returns:
        The global AppSettings singleton.
    """
    global _settings
    if _settings is None or env_file is not None:
        if env_file is not None:
            import os

            os.environ["ENV_FILE"] = str(env_file)
        _settings = AppSettings()
    return _settings
