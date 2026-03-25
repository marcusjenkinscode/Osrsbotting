"""Interactive calibration tool for the OSRS chat region.

Launch with::

    python calibration_tool.py

Or via the main entry point::

    python main.py --calibrate

Controls
--------
* Arrow keys  — move the region 1 px in that direction.
* Shift+Arrow — resize the region by 1 px (right/down = grow, left/up = shrink).
* ``s``       — save current region to ``.env`` and quit.
* ``q`` / ESC — quit without saving.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

try:
    import cv2
    import mss
    import numpy as np
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

from config_manager import CaptureSettings


def _save_region_to_env(region: dict, env_path: Path = Path(".env")) -> None:
    """Update or create .env entries for the chat region.

    Args:
        region: Dict with keys ``left``, ``top``, ``width``, ``height``.
        env_path: Path to the .env file.
    """
    keys = {
        "CHAT_REGION_LEFT": str(region["left"]),
        "CHAT_REGION_TOP": str(region["top"]),
        "CHAT_REGION_WIDTH": str(region["width"]),
        "CHAT_REGION_HEIGHT": str(region["height"]),
    }

    existing_lines: list = []
    if env_path.exists():
        existing_lines = env_path.read_text().splitlines()

    updated_keys = set()
    new_lines = []
    for line in existing_lines:
        stripped = line.strip()
        matched = False
        for k in keys:
            if stripped.startswith(f"{k}=") or stripped.startswith(f"{k} ="):
                new_lines.append(f"{k}={keys[k]}")
                updated_keys.add(k)
                matched = True
                break
        if not matched:
            new_lines.append(line)

    for k, v in keys.items():
        if k not in updated_keys:
            new_lines.append(f"{k}={v}")

    env_path.write_text("\n".join(new_lines) + "\n")
    logger.info("Saved region {} to {}", region, env_path)


def run_calibration(env_path: Path = Path(".env")) -> None:
    """Launch the interactive OpenCV calibration window.

    Args:
        env_path: Path to the .env file that will be updated on save.

    Raises:
        SystemExit: If required dependencies are missing.
    """
    if not _DEPS_OK:
        logger.error("OpenCV and/or MSS are required for calibration.  Run: pip install opencv-python mss")
        sys.exit(1)

    settings = CaptureSettings()
    region = dict(settings.region)  # mutable copy

    window_name = "OSRS Chat Calibration  |  Arrows=move  Shift+Arrows=resize  S=save  Q=quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 300)

    logger.info("Calibration started.  Window: {}", window_name)
    print("\n=== OSRS Chat Region Calibration ===")
    print("  Arrow keys      : move region")
    print("  Shift+Arrow     : resize region")
    print("  S               : save & quit")
    print("  Q / ESC         : quit without saving")
    print("=" * 38)

    try:
        with mss.mss() as sct:
            while True:
                # Capture current region
                try:
                    shot = sct.grab(region)
                    frame = np.array(shot)[:, :, :3]
                except Exception as exc:
                    logger.warning("Capture error: {}", exc)
                    # Create placeholder frame
                    frame = np.zeros((region["height"], region["width"], 3), dtype=np.uint8)

                # Draw overlay info
                display = frame.copy()
                info = (
                    f"Region: L={region['left']} T={region['top']} "
                    f"W={region['width']} H={region['height']}"
                )
                cv2.putText(
                    display, info, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
                )

                # Draw border
                cv2.rectangle(
                    display, (0, 0),
                    (display.shape[1] - 1, display.shape[0] - 1),
                    (0, 255, 0), 2,
                )

                cv2.imshow(window_name, display)

                key = cv2.waitKey(100) & 0xFF

                if key == ord("q") or key == 27:  # q or ESC
                    print("Quitting without saving.")
                    break
                elif key == ord("s"):
                    _save_region_to_env(region, env_path)
                    print(f"Saved!  Region: {region}")
                    break

                # Arrow key codes (OpenCV on Linux/Windows: 81-84, Mac differs)
                elif key == 81 or key == ord("a"):   # left → move left
                    region["left"] = max(0, region["left"] - 1)
                elif key == 83 or key == ord("d"):   # right → move right
                    region["left"] += 1
                elif key == 82 or key == ord("w"):   # up → move up
                    region["top"] = max(0, region["top"] - 1)
                elif key == 84:                       # down arrow → move down
                    region["top"] += 1

                # Shift+Arrow: resize (approximated by checking for uppercase letters)
                elif key == ord("A"):   # Shift+left → shrink width
                    region["width"] = max(50, region["width"] - 1)
                elif key == ord("D"):   # Shift+right → grow width
                    region["width"] += 1
                elif key == ord("W"):   # Shift+up → shrink height
                    region["height"] = max(20, region["height"] - 1)
                elif key == ord("S"):   # Shift+down → grow height
                    region["height"] += 1

    finally:
        cv2.destroyAllWindows()
        logger.info("Calibration window closed.")


if __name__ == "__main__":
    run_calibration()
