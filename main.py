"""OSRS Chat Logger — main entry point.

Usage
-----
::

    python main.py                   # run with default .env
    python main.py --config my.env   # use a custom config file
    python main.py --calibrate       # launch calibration tool
    python main.py --debug           # verbose debug logging
"""

from __future__ import annotations

import argparse
import queue
import sys
import time
from pathlib import Path
from typing import Optional

from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from capture_engine import CaptureEngine
from chat_parser import ChatParser, ParsedMessage
from config_manager import AppSettings, CaptureSettings, get_settings
from n8n_dispatcher import N8nDispatcher, WebhookPayload
from ocr_processor import OCRProcessor
from utils import SESSION_ID, PerfTimer, register_signal_handlers, setup_logging, shutdown_event

console = Console()


# ---------------------------------------------------------------------------
# Rich status dashboard
# ---------------------------------------------------------------------------


def _build_status_table(
    session_id: str,
    total_captured: int,
    total_processed: int,
    total_sent: int,
    last_messages: list,
    fps: float,
) -> Table:
    """Build a Rich table showing live application status.

    Args:
        session_id: Current session UUID.
        total_captured: Frames captured so far.
        total_processed: Frames processed through OCR.
        total_sent: Messages dispatched to n8n.
        last_messages: Recent ParsedMessage objects.
        fps: Current capture rate.

    Returns:
        A formatted Rich Table.
    """
    table = Table(title="OSRS Chat Logger", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Session", session_id[:8] + "…")
    table.add_row("Captured frames", str(total_captured))
    table.add_row("OCR processed", str(total_processed))
    table.add_row("Messages sent", str(total_sent))
    table.add_row("Capture FPS", f"{fps:.2f}")

    if last_messages:
        table.add_section()
        table.add_row("[bold]Recent messages[/bold]", "")
        for msg in last_messages[-5:]:
            author = msg.author or "(system)"
            table.add_row(
                f"  [{msg.chat_type}] {author}",
                msg.message[:80],
            )

    return table


# ---------------------------------------------------------------------------
# Processing thread helper
# ---------------------------------------------------------------------------


def _process_frames(
    frame_queue: "queue.Queue",
    processor: OCRProcessor,
    parser: ChatParser,
    dispatcher: N8nDispatcher,
    min_confidence: float,
    capture_region_settings: "CaptureSettings",
    stats: dict,
) -> None:
    """Consumer: pull frames from queue, run OCR, parse, dispatch.

    Args:
        frame_queue: Queue of BGR numpy arrays from the capture engine.
        processor: OCR processor instance.
        parser: Chat line parser instance.
        dispatcher: n8n dispatcher instance.
        min_confidence: Minimum OCR confidence threshold (0–1).
        capture_region_settings: Capture region config included in payloads.
        stats: Shared mutable dict for counters and recent messages.
    """
    while not shutdown_event.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        capture_timer = PerfTimer().start()
        ocr_result = processor.process(frame)
        stats["total_processed"] += 1

        if ocr_result is None or not ocr_result.is_usable:
            continue

        if ocr_result.confidence < min_confidence:
            logger.debug("OCR confidence {:.2f} too low — skip", ocr_result.confidence)
            continue

        messages = parser.parse_lines(ocr_result.raw_text, confidence=ocr_result.confidence)
        for msg in messages:
            payload = WebhookPayload(
                message=msg,
                capture_time_ms=capture_timer.elapsed_ms,
                processing_time_ms=ocr_result.processing_time_ms,
                capture_settings=capture_region_settings,
            )
            dispatcher.enqueue(payload)
            stats["total_sent"] += 1
            stats["last_messages"].append(msg)
            if len(stats["last_messages"]) > 20:
                stats["last_messages"].pop(0)
            logger.info("[{}] {}: {}", msg.chat_type, msg.author or "(system)", msg.message)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="OSRS Chat Logger — capture in-game chat via OCR and send to n8n."
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="Path to a custom .env configuration file."
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Launch the interactive region calibration tool and exit.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG-level logging."
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    """Application entry point.

    Args:
        argv: Optional argument list for testing.

    Returns:
        Exit code (0 = success, non-zero = error).
    """
    args = parse_args(argv)

    # ------------------------------------------------------------------ config
    settings = get_settings(env_file=args.config)

    # ------------------------------------------------------------------ logging
    log_level = "DEBUG" if args.debug else settings.logging.log_level
    setup_logging(
        log_level=log_level,
        log_file=settings.logging.log_file,
        max_days=settings.logging.max_log_days,
    )

    # ------------------------------------------------------------------ calibrate
    if args.calibrate:
        from calibration_tool import run_calibration

        env_path = args.config or Path(".env")
        run_calibration(env_path=env_path)
        return 0

    # ------------------------------------------------------------------ signals
    register_signal_handlers()

    logger.info("OSRS Chat Logger starting — session {}", SESSION_ID)
    logger.info("Config: {}", settings)

    # ------------------------------------------------------------------ components
    frame_queue: "queue.Queue" = queue.Queue(maxsize=10)

    capture_engine = CaptureEngine(
        settings=settings.capture,
        frame_queue=frame_queue,
    )

    ocr_processor = OCRProcessor(settings=settings.ocr)

    chat_parser = ChatParser(
        keyword_filters=settings.filters.keyword_filters,
        ignore_players=settings.filters.ignore_players,
        min_confidence=settings.ocr.min_confidence,
    )

    dispatcher = N8nDispatcher(
        settings=settings.webhook,
        capture_settings=settings.capture,
    )

    # ------------------------------------------------------------------ shared stats
    stats: dict = {
        "total_captured": 0,
        "total_processed": 0,
        "total_sent": 0,
        "last_messages": [],
        "start_time": time.time(),
    }

    # ------------------------------------------------------------------ start
    capture_engine.start()
    dispatcher.start()

    import threading

    proc_thread = threading.Thread(
        target=_process_frames,
        args=(
            frame_queue,
            ocr_processor,
            chat_parser,
            dispatcher,
            settings.ocr.min_confidence,
            settings.capture,
            stats,
        ),
        name="processing-thread",
        daemon=True,
    )
    proc_thread.start()

    # ------------------------------------------------------------------ dashboard
    try:
        with Live(console=console, refresh_per_second=2, screen=False) as live:
            while not shutdown_event.is_set():
                elapsed = time.time() - stats["start_time"]
                fps = stats["total_captured"] / elapsed if elapsed > 0 else 0.0
                live.update(
                    Panel(
                        _build_status_table(
                            session_id=SESSION_ID,
                            total_captured=stats["total_captured"],
                            total_processed=stats["total_processed"],
                            total_sent=stats["total_sent"],
                            last_messages=stats["last_messages"],
                            fps=fps,
                        ),
                        title="[bold green]OSRS Chat Logger[/bold green]",
                        subtitle="Press Ctrl+C to stop",
                    )
                )
                time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown_event.set()

    # ------------------------------------------------------------------ shutdown
    logger.info("Shutting down …")
    capture_engine.stop()
    dispatcher.stop()
    proc_thread.join(timeout=5.0)

    logger.info(
        "Session complete — captured={} processed={} sent={}",
        stats["total_captured"],
        stats["total_processed"],
        stats["total_sent"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
