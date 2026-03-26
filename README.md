# OSRS Chat Logger

Capture Old School RuneScape in-game chat via screen OCR and forward structured messages to an [n8n](https://n8n.io/) automation workflow.

> **Read-only by design** — no keystrokes or clicks are sent to the game client.

---

## Features

| Feature | Detail |
|---|---|
| **High-speed capture** | `mss` region capture at configurable FPS |
| **OSRS-tuned OCR** | Yellow/white text HSV masking → Tesseract with game-specific config |
| **Chat parsing** | Handles public, clan, guest-clan, private (in/out), trade messages |
| **Deduplication** | Fuzzy string matching + TTL buffer prevents duplicate events |
| **n8n integration** | Batch HTTP POST with exponential back-off retry |
| **Offline queue** | SQLite buffer when n8n is unreachable; retried automatically |
| **Calibration tool** | Interactive OpenCV window to tune the chat region |
| **Rich dashboard** | Live terminal status with message feed |
| **Safe** | No global hotkeys; runs at low process priority; read-only |

---

## Requirements

- Python 3.9+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed and on `PATH`
- A running [n8n](https://docs.n8n.io/hosting/) instance (optional for local testing)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/marcusjenkinscode/Osrsbotting.git
cd Osrsbotting

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy the example config and edit it
cp .env.example .env
```

Edit `.env` with your settings (n8n URL, chat region, etc.).

---

## Configuration

All configuration is loaded from `.env` (or a file passed via `--config`).  
See [`.env.example`](.env.example) for all available options.

Key settings:

| Variable | Default | Description |
|---|---|---|
| `CHAT_REGION_LEFT` | `10` | Left edge of chat box (px) |
| `CHAT_REGION_TOP` | `850` | Top edge of chat box (px) |
| `CHAT_REGION_WIDTH` | `520` | Width of chat box (px) |
| `CHAT_REGION_HEIGHT` | `130` | Height of chat box (px) |
| `N8N_WEBHOOK_URL` | — | Full URL of your n8n webhook |
| `N8N_SECRET_VALUE` | — | Secret passed in the auth header |
| `KEYWORD_FILTERS` | — | Comma-separated keywords to forward |
| `MIN_CONFIDENCE` | `0.75` | Minimum OCR confidence (0–1) |

---

## Usage

```bash
# Normal run
python main.py

# Use a custom .env file
python main.py --config /path/to/custom.env

# Enable verbose debug logging
python main.py --debug

# Launch the interactive calibration tool
python main.py --calibrate
```

---

## Calibration

Run `python main.py --calibrate` to open a live preview window:

```
=== OSRS Chat Region Calibration ===
  Arrow keys      : move region
  Shift+Arrow     : resize region
  S               : save & quit
  Q / ESC         : quit without saving
======================================
```

The region co-ordinates are saved back to your `.env` file automatically.

---

## n8n Webhook Payload

Each HTTP POST sends a JSON body:

```json
{
  "messages": [
    {
      "timestamp": "2024-01-15T14:32:10Z",
      "chat_type": "clan",
      "author": "Zezima",
      "message": "Nice kill!",
      "raw_ocr": "[Clan] [Zezima] Nice kill!",
      "confidence": 0.91,
      "game_window": {
        "resolution": "520x130",
        "client_mode": "resizable",
        "region_captured": {"left": 10, "top": 850, "width": 520, "height": 130}
      },
      "metadata": {
        "capture_time_ms": 12,
        "processing_time_ms": 95,
        "session_id": "550e8400-e29b-41d4-a716-446655440000"
      }
    }
  ]
}
```

---

## Project Structure

```
Osrsbotting/
├── main.py               # Entry point (--config / --calibrate / --debug)
├── capture_engine.py     # MSS screen capture thread
├── ocr_processor.py      # HSV masking + Tesseract OCR pipeline
├── chat_parser.py        # OSRS chat regex patterns + deduplication
├── n8n_dispatcher.py     # Webhook client, batching, SQLite offline queue
├── config_manager.py     # Pydantic settings models
├── calibration_tool.py   # Interactive OpenCV region selector
├── utils.py              # Logging, signal handlers, timers
├── requirements.txt
├── .env.example
└── tests/
    ├── test_chat_parser.py
    ├── test_config_manager.py
    ├── test_n8n_dispatcher.py
    └── test_ocr_processor.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Safety Notes

- The logger is **read-only** — it never sends keystrokes or mouse clicks to OSRS.
- Capture intervals are randomised with ±0.5 s jitter.
- No secrets are stored in source code; all sensitive values live in `.env`.
- The process can be stopped cleanly with **Ctrl+C** (SIGINT) or SIGTERM.
