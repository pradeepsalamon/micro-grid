"""
app/main.py  ─  Microgrid ESP32 Simulator (Data Generator)
────────────────────────────────────────────────────────────
Simulates an ESP32 sending real-time power telemetry to the
middleware backend's  POST /api/telemetry  endpoint.

Works in both environments:
  • Local dev  → BACKEND_URL=http://localhost:8000  (default)
  • Minikube   → BACKEND_URL=http://middleware-service.microgrid.svc.cluster.local:8000
                 (set via env var in the Kubernetes deployment)

Control endpoints
─────────────────
  GET /         health check
  GET /start    begin sending fake telemetry every INTERVAL_SECONDS
  GET /stop     stop sending
  GET /status   show current state + last response from backend
  GET /sample   return one fake payload (dry-run, no POST)
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()

BACKEND_URL: str       = os.getenv("BACKEND_URL", "http://localhost:8000")
INTERVAL_SECONDS: float = float(os.getenv("INTERVAL_SECONDS", "1"))   # send every N sec
TELEMETRY_PATH: str    = "/api/telemetry"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("esp32-sim")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Microgrid ESP32 Simulator",
    description=(
        "Generates realistic fake microgrid telemetry and POSTs it to the "
        "middleware /api/telemetry endpoint. Simulates an ESP32 controller."
    ),
    version="2.0.0",
)

# ── Global state ──────────────────────────────────────────────────────────────
is_running: bool          = False
_thread: Optional[threading.Thread] = None
_last_payload: Dict       = {}
_last_response: Dict      = {}
_send_count: int          = 0
_error_count: int         = 0


# ══════════════════════════════════════════════════════════════════════════════
# Fake data generator
# ══════════════════════════════════════════════════════════════════════════════

def _rand(lo: float, hi: float, decimals: int = 2) -> float:
    """Return a random float in [lo, hi] rounded to `decimals` places."""
    return round(random.uniform(lo, hi), decimals)


def generate_fake_telemetry() -> Dict[str, Any]:
    """
    Build a realistic fake ``PowerData`` payload that matches the Pydantic
    model expected by the backend's ``/api/telemetry`` endpoint.

    Simulated scenario
    ──────────────────
    * Solar  0–500 W  (daytime variability)
    * Wind   0–300 W  (gusty)
    * Battery SOC 0.1–1.0
    * Grid available ~80 % of the time
    * Inverter used when grid is down

    Returns:
        Dict matching the ``PowerData`` schema.
    """
    solar_power   = _rand(0,   500)
    wind_power    = _rand(0,   300)
    battery_soc   = _rand(0.1, 1.0, 3)
    battery_power = _rand(0,   200)

    grid_available = random.random() > 0.2          # True 80 % of the time
    using_inverter = not grid_available              # inverter kicks in when grid is down

    critical_load     = _rand(80,  300)
    non_critical_load = _rand(20,  150)
    grid_load         = critical_load + non_critical_load if grid_available else 0.0
    inverter_load     = critical_load if using_inverter else 0.0
    total_load        = critical_load + non_critical_load

    return {
        "sources": {
            "solar_power":   solar_power,
            "wind_power":    wind_power,
            "battery_power": battery_power,
            "battery_soc":   battery_soc,
        },
        "loads": {
            "critical":     critical_load,
            "non_critical": non_critical_load,
            "grid":         grid_load,
            "inverter":     inverter_load,
            "total":        total_load,
        },
        "info": {
            "grid_available":           grid_available,
            "using_inverter":           using_inverter,
            "critical_on_inverter":     using_inverter,
            "non_critical_on_inverter": False,   # non-critical shed during outage
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Background sender loop
# ══════════════════════════════════════════════════════════════════════════════

def _send_loop() -> None:
    """
    Background thread: generate and POST fake telemetry to the backend.

    Uses a fixed-rate scheduler so INTERVAL_SECONDS controls the time
    between the *start* of each request, not end→start.  If a request
    takes longer than INTERVAL_SECONDS the next one fires immediately
    (no negative sleep).

    A single httpx.Client is reused for the entire loop to avoid
    TCP connection overhead on every send.
    """
    global is_running, _last_payload, _last_response, _send_count, _error_count

    target_url = f"{BACKEND_URL}{TELEMETRY_PATH}"
    logger.info("▶  Sender loop started → %s  (interval=%.1fs)", target_url, INTERVAL_SECONDS)

    with httpx.Client(timeout=15.0) as client:          # reuse connection pool
        while is_running:
            tick_start = time.monotonic()               # ← mark interval start

            payload = generate_fake_telemetry()
            _last_payload = payload

            try:
                resp = client.post(target_url, json=payload)
                resp.raise_for_status()
                _last_response = resp.json()
                _send_count += 1

                elapsed = time.monotonic() - tick_start
                logger.info(
                    "✅ [%d] sent in %.2fs → action_id=%s  critical_src=%s  grid=%s",
                    _send_count,
                    elapsed,
                    _last_response.get("action_id"),
                    _last_response.get("critical_source"),
                    payload["info"]["grid_available"],
                )

            except httpx.HTTPStatusError as exc:
                _error_count += 1
                elapsed = time.monotonic() - tick_start
                logger.error(
                    "❌ [%.2fs] HTTP %s from backend: %s",
                    elapsed, exc.response.status_code, exc.response.text,
                )
            except httpx.RequestError as exc:
                _error_count += 1
                elapsed = time.monotonic() - tick_start
                logger.error("❌ [%.2fs] Cannot reach backend (%s): %s", elapsed, target_url, exc)
            except Exception as exc:
                _error_count += 1
                elapsed = time.monotonic() - tick_start
                logger.error("❌ [%.2fs] Unexpected error: %s", elapsed, exc, exc_info=True)

            # ── Fixed-rate sleep: only sleep what's left of the interval ───────
            elapsed_total = time.monotonic() - tick_start
            sleep_for = max(0.0, INTERVAL_SECONDS - elapsed_total)
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Backend is slower than INTERVAL_SECONDS — log a warning once
                logger.warning(
                    "⚠️  Backend took %.2fs > interval %.1fs — sending as fast as backend allows.",
                    elapsed_total, INTERVAL_SECONDS,
                )

    logger.info("⏹  Sender loop stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# API routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["health"])
def home():
    """Health check — returns service info and current target URL."""
    return {
        "service":      "ESP32 Simulator (Data Generator)",
        "backend_url":  BACKEND_URL,
        "target":       f"{BACKEND_URL}{TELEMETRY_PATH}",
        "interval_sec": INTERVAL_SECONDS,
        "is_running":   is_running,
    }


@app.get("/start", tags=["control"])
def start_stream():
    """
    Start the background loop that sends fake telemetry to the backend
    every ``INTERVAL_SECONDS``.
    """
    global is_running, _thread, _send_count, _error_count

    if is_running:
        return {
            "status":  "already_running",
            "message": "Telemetry stream is already active.",
            "sends":   _send_count,
            "errors":  _error_count,
        }

    # Reset counters on fresh start
    _send_count  = 0
    _error_count = 0

    is_running = True
    _thread = threading.Thread(target=_send_loop, daemon=True)
    _thread.start()

    return {
        "status":   "started",
        "message":  f"Sending telemetry to {BACKEND_URL}{TELEMETRY_PATH} every {INTERVAL_SECONDS}s",
        "target":   f"{BACKEND_URL}{TELEMETRY_PATH}",
    }


@app.get("/stop", tags=["control"])
def stop_stream():
    """Stop the telemetry sender loop."""
    global is_running

    if not is_running:
        return {"status": "not_running", "message": "No stream is active."}

    is_running = False
    return {
        "status":  "stopped",
        "message": "Telemetry stream stopped.",
        "sends":   _send_count,
        "errors":  _error_count,
    }


@app.get("/status", tags=["control"])
def get_status():
    """Return current streaming state, counters, and last backend response."""
    return {
        "is_running":    is_running,
        "sends":         _send_count,
        "errors":        _error_count,
        "backend_url":   BACKEND_URL,
        "target":        f"{BACKEND_URL}{TELEMETRY_PATH}",
        "interval_sec":  INTERVAL_SECONDS,
        "last_payload":  _last_payload,
        "last_response": _last_response,
    }


@app.get("/sample", tags=["debug"])
def get_sample():
    """
    Generate and return one fake telemetry payload without POSTing it.
    Useful for debugging the payload shape.
    """
    return {
        "note":    "dry-run — nothing was sent to the backend",
        "target":  f"{BACKEND_URL}{TELEMETRY_PATH}",
        "payload": generate_fake_telemetry(),
    }
