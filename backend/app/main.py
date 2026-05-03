"""
app/main.py
───────────
FastAPI middleware service for the Microgrid system.

Data-flow overview
──────────────────
  ESP32 → POST /api/telemetry
        → [InfluxDB] write_power_data          (raw telemetry)
        → fetch ML forecasts
        → build PredictionRequest
        → [InfluxDB] write_prediction_request  (RL observation vector)
        → POST Core AI /predict
        → [InfluxDB] write_prediction_response (AI decision)
        → return command to ESP32

InfluxDB writes are fire-and-forget: failures are logged but never
propagate to the caller.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import httpx

from models import PowerData, PredictionRequest, PredictionResponse
from utils import (
    process_telemetry_and_get_command,
    fetch_predictions,
    get_weather,
    extract_observation,
    send_observation_to_core_ai_and_get_decision,
    build_prediction_request_from_obs,   # new helper – see utils.py
)
from services.influx_service import (
    write_power_data,
    write_prediction_request,
    write_prediction_response,
    close as influx_close,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Lifespan: manage InfluxDB connection lifecycle
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open resources on startup; flush & close on shutdown."""
    logger.info("🚀 Middleware service starting up…")
    yield
    logger.info("🛑 Middleware service shutting down — closing InfluxDB client…")
    influx_close()


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI application
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Microgrid Middleware Service",
    description=(
        "Receives real-time telemetry from ESP32, fetches ML forecasts, "
        "forwards observations to the Core AI RL agent, and returns power-routing "
        "commands.  All three data flows are persisted to InfluxDB 2.x."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["health"])
def read_root():
    """Health-check — confirms the middleware service is running."""
    return {"service": "middleware service is running"}


@app.post("/api/telemetry", tags=["telemetry"])
async def receive_telemetry(data: PowerData):
    """
    Primary ingest endpoint called by the ESP32 at each control interval.

    Pipeline
    ────────
    1. Persist raw telemetry  → InfluxDB ``power_data``
    2. Fetch ML forecasts     → solar / wind / load / power-cut / theft
    3. Build observation      → ``PredictionRequest`` (normalised RL state)
    4. Persist observation    → InfluxDB ``prediction_request``
    5. Call Core AI           → ``PredictionResponse`` (RL action)
    6. Persist AI decision    → InfluxDB ``prediction_response``
    7. Return command         → ESP32

    InfluxDB failures at any step are swallowed so the control loop
    is never interrupted.
    """
    # ── Step 1: persist raw telemetry ─────────────────────────────────────────
    logger.info("📥 Telemetry received — writing to InfluxDB…")
    write_power_data(data)           # non-blocking: errors are only logged

    # ── Steps 2–7: existing pipeline ──────────────────────────────────────────
    decision = await process_telemetry_and_get_command(
        telemetry_data   = data.model_dump(),
        on_request_built = _on_prediction_request,   # callback for step 4
        on_response_got  = _on_prediction_response,  # callback for step 6
    )

    return {
        "status":             decision.get("status", "error"),
        "action_id":          decision.get("action_id"),
        "critical_source":    decision.get("critical_source"),
        "noncritical_source": decision.get("noncritical_source"),
        "inverter_total_w":   decision.get("inverter_total_w"),
        "overloaded":         decision.get("overloaded"),
        "constraint_applied": decision.get("constraint_applied"),
    }


# ── InfluxDB callbacks injected into the pipeline ────────────────────────────

def _on_prediction_request(req: PredictionRequest) -> None:
    """Persist the RL observation vector before it is sent to Core AI."""
    logger.info("🔬 Writing prediction_request to InfluxDB…")
    write_prediction_request(req)


def _on_prediction_response(resp: PredictionResponse) -> None:
    """Persist the Core AI decision after it is received."""
    logger.info("🤖 Writing prediction_response to InfluxDB…")
    write_prediction_response(resp)


# ── Secondary / diagnostic endpoints ─────────────────────────────────────────

@app.get("/api/telemetry/latest", tags=["telemetry"])
def get_latest_telemetry():
    """Return the most recently received telemetry payload."""
    from utils import latest_telemetry
    return latest_telemetry or {}

@app.get("/forecast", tags=["diagnostics"])
async def forecast():
    """Fetch raw predictions from all ML models and return them."""
    predictions = await fetch_predictions()
    return predictions


@app.get("/weather", tags=["diagnostics"])
def get_weather_data():
    """Fetch weather data from the ML service."""
    return get_weather()


@app.post("/predict", response_model=PredictionResponse, tags=["ai"])
async def predict(payload: PredictionRequest):
    """
    Direct proxy: forward a ``PredictionRequest`` to Core AI and return the
    ``PredictionResponse``.

    Also writes both the request and the response to InfluxDB.

    Response schema::

        {
          "action_id":          0,
          "critical_source":    "grid",
          "noncritical_source": "grid",
          "inverter_total_w":   0.0,
          "overloaded":         false,
          "constraint_applied": false
        }
    """
    # Persist the incoming request
    logger.info("📨 /predict called — writing prediction_request to InfluxDB…")
    write_prediction_request(payload)

    try:
        from utils import CORE_AI_URL   # import here to avoid circular at module level

        async with httpx.AsyncClient(timeout=10.0) as client:
            result = await client.post(
                f"{CORE_AI_URL}/predict",
                json=payload.model_dump(),
            )
            result.raise_for_status()
            response_data = result.json()

        response = PredictionResponse(**response_data)

        # Persist the AI decision
        logger.info("✅ /predict response received — writing prediction_response to InfluxDB…")
        write_prediction_response(response)

        return response

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Core AI error: {exc.response.text}",
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Core AI unreachable: {exc}",
        )