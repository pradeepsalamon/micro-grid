import logging
import os
from typing import Callable, Dict, Optional, Tuple

import httpx
import requests
from dotenv import load_dotenv
from models import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)

# ================================
# CONFIGURATION
# ================================
load_dotenv()

PORT    = int(os.getenv("PORT", 8000))
ML_URL      = os.getenv("ML_URL",      "http://localhost:8001")
CORE_AI_URL = os.getenv("CORE_AI_URL", "http://localhost:8002")

# ================================
# GLOBAL STATE
# ================================
latest_telemetry: Dict = {}


# ================================
# TELEMETRY PROCESSING
# ================================
def process_telemetry(telemetry_data: Dict) -> Dict:
    """
    Store telemetry data globally.
    
    Args:
        telemetry_data: Raw telemetry data from ESP32
    """
    global latest_telemetry
    latest_telemetry = telemetry_data
    print("\n📥 TELEMETRY RECEIVED:", latest_telemetry)


async def process_telemetry_and_get_command(
    telemetry_data: Dict,
    on_request_built: Optional[Callable[[PredictionRequest], None]] = None,
    on_response_got:  Optional[Callable[[PredictionResponse], None]] = None,
) -> Dict:
    """
    Complete flow: Store telemetry → Fetch forecasts → Extract observation →
    Send to Core AI → Return decision/command.

    Args:
        telemetry_data:   Raw telemetry data from ESP32.
        on_request_built: Optional callback fired just before the
                          ``PredictionRequest`` is sent to Core AI.
                          Receives the validated model instance.
        on_response_got:  Optional callback fired after a successful
                          ``PredictionResponse`` is received from Core AI.
                          Receives the validated model instance.

    Returns:
        Dictionary with command/decision from Core AI.
    """
    global latest_telemetry
    latest_telemetry = telemetry_data
    logger.info("📥 TELEMETRY RECEIVED: %s", latest_telemetry)

    # Step 1: Fetch forecasts from ML models
    logger.info("🔄 Fetching forecasts from ML models…")
    predictions = await fetch_predictions()

    if predictions.get("errors"):
        logger.warning("⚠️ Errors fetching predictions: %s", predictions["errors"])

    # Step 2: Extract observation
    obs_data = extract_observation(predictions)

    # Step 3: Send observation to Core AI and get decision
    logger.info("🤖 Sending observation to Core AI…")
    decision = await send_observation_to_core_ai_and_get_decision(
        obs_data,
        on_request_built=on_request_built,
        on_response_got=on_response_got,
    )

    logger.info("✅ Decision received: %s", decision)
    return decision


# ================================
# PREDICTION SERVICES
# ================================
async def fetch_predictions() -> Dict:
    """
    Fetch predictions from ML service for multiple models.
    
    Returns:
        Dictionary with predictions and errors
    """
    response = {"predictions": {}, "errors": []}
    
    # Build theft-prediction endpoint with grid load from latest telemetry
    theft_voltage = 230.0  # default nominal voltage
    theft_current = 1.0    # default
    if latest_telemetry and "loads" in latest_telemetry:
        grid_load_w = latest_telemetry["loads"].get("grid", 0)
        # Estimate current from power and nominal voltage (P = V * I)
        if grid_load_w > 200:  # Only calculate if load is significant to avoid noise
            theft_current = round(grid_load_w / theft_voltage, 4)
    theft_endpoint = f"theft-prediction?voltage={theft_voltage}&current={theft_current}"
    
    predictions = [
        ("solar-prediction", "solar_prediction"),
        ("wind-prediction", "wind_prediction"),
        ("load-prediction", "load_prediction"),
        ("powercut-prediction", "power_cut_prediction"),
        (theft_endpoint, "theft_prediction"),
    ]
    
    async with httpx.AsyncClient() as client:
        for endpoint, key in predictions:
            try:
                result = await client.get(f"{ML_URL}/{endpoint}")
            except httpx.RequestError as exc:
                response["errors"].append({
                    "service": key,
                    "message": str(exc)
                })
            else:
                if result.status_code == 200:
                    response["predictions"][key] = result.json()
                else:
                    response["errors"].append({
                        "service": key,
                        "status_code": result.status_code,
                        "message": f"Failed to get {key}"
                    })
    
    return response


# ================================
# EXTERNAL SERVICE CALLS
# ================================
def get_weather() -> Dict:
    """
    Fetch weather data from ML service.
    
    Returns:
        Weather data or error message
    """
    try:
        result = requests.get(f"{ML_URL}/weather-data")
        result.raise_for_status()
        return result.json()
    except requests.RequestException as exc:
        return {"error": str(exc)}



# ================================
# CORE AI INTERFACE
# ================================
def extract_observation(predictions: Dict) -> Dict:
    """
    Extract and normalize observation data from telemetry and predictions
    for Core AI controller.
    
    Args:
        predictions: Response from ML prediction service
        
    Returns:
        Dictionary with normalized observation array and metadata
    """
    pred = predictions.get("predictions", {})

    # ── Current measurements from telemetry (actual sensor readings in Watts) ──
    solar_power_w      = latest_telemetry.get("sources", {}).get("solar_power", 0)
    wind_power_w       = latest_telemetry.get("sources", {}).get("wind_power", 0)
    battery_soc        = latest_telemetry.get("sources", {}).get("battery_soc", 0)
    critical_load_w    = latest_telemetry.get("loads", {}).get("critical", 0)
    noncritical_load_w = latest_telemetry.get("loads", {}).get("non_critical", 0)
    grid_available     = int(latest_telemetry.get("info", {}).get("grid_available", False))

    # ── Forecasts from ML models ──
    # power_output is in kW (0–1 kW for 1kW panel/turbine) → convert to Watts (×1000)
    solar_forecast_w = pred.get("solar_prediction", {}).get("power_output", 0) * 100
    wind_forecast_w  = pred.get("wind_prediction",  {}).get("power_output", 0)
    # scaled_load is a 0-1 fraction of house_max_kw (1 kW) → stored as fraction, sent as W later
    load_forecast_frac = pred.get("load_prediction", {}).get("scaled_load", 0)

    # Power cut probability
    power_cut_prob = pred.get("power_cut_prediction", {}).get("prob_cut", 0)

    # ── Normalize (matches normalize_state() in RL agent env.py) ──
    obs = [
        solar_power_w      / 500,   # obs[0]: actual solar  W ÷ 500
        wind_power_w       / 500,   # obs[1]: actual wind   W ÷ 500
        battery_soc,                # obs[2]: SOC already in [0, 1]
        critical_load_w    / 300,   # obs[3]
        noncritical_load_w / 300,   # obs[4]
        float(grid_available),      # obs[5]
        solar_forecast_w   / 500,   # obs[6]: forecast W ÷ 500
        wind_forecast_w    / 500,   # obs[7]: forecast W ÷ 500
        load_forecast_frac,         # obs[8]: fraction (×1000 → W when sending to RL)
        power_cut_prob              # obs[9]
    ]
    
    return {
        "observation": obs,
        "metadata": {
            "solar_power_w": latest_telemetry.get("sources", {}).get("solar_power", 0),
            "wind_power_w": latest_telemetry.get("sources", {}).get("wind_power", 0),
            "battery_soc": battery_soc * 100,  # Return as percentage for readability
            "critical_load_w": latest_telemetry.get("loads", {}).get("critical", 0),
            "grid_available": grid_available,
            "power_cut_probability": power_cut_prob
        }
    }




def build_prediction_request_from_obs(obs: list) -> PredictionRequest:
    """
    De-normalise an observation vector and return a validated
    ``PredictionRequest`` Pydantic model.

    Observation index mapping
    ─────────────────────────
    obs[0]  solar_power   (normalised ÷ 500)
    obs[1]  wind_power    (normalised ÷ 500)
    obs[2]  battery_soc   (0–1, unchanged)
    obs[3]  critical_load (normalised ÷ 300)
    obs[4]  noncrit_load  (normalised ÷ 300)
    obs[5]  grid_avail    (0 or 1)
    obs[6]  solar_fcast   (normalised ÷ 500)
    obs[7]  wind_fcast    (normalised ÷ 500)
    obs[8]  load_fcast    (fraction × 1000 W house max)
    obs[9]  power_cut_prob(0–1)

    Args:
        obs: 10-element normalised observation list.

    Returns:
        Validated ``PredictionRequest`` instance.
    """
    return PredictionRequest(
        solar_power_w        = round(obs[0] * 500,  4),
        wind_power_w         = round(obs[1] * 500,  4),
        battery_soc          = max(0.0, min(1.0, obs[2])),
        critical_load_w      = round(obs[3] * 300,  4),
        noncritical_load_w   = round(obs[4] * 300,  4),
        grid_available       = int(obs[5]),
        solar_forecast_w     = round(obs[6] * 500,  4),
        wind_forecast_w      = round(obs[7] * 500,  4),
        load_forecast_w      = round(obs[8] * 1000, 4),
        power_cut_probability= max(0.0, min(1.0, obs[9])),
    )


async def send_observation_to_core_ai_and_get_decision(
    obs_data: Dict,
    on_request_built: Optional[Callable[[PredictionRequest], None]] = None,
    on_response_got:  Optional[Callable[[PredictionResponse], None]] = None,
) -> Dict:
    """
    Send observation to Core AI controller (/predict) and get decision/command.

    Builds a ``PredictionRequest`` from the observation array, optionally
    fires ``on_request_built`` before the HTTP call and ``on_response_got``
    after a successful response, then maps the ``PredictionResponse`` back
    to a plain command dict.

    Args:
        obs_data:         Extracted observation data (observation list + metadata).
        on_request_built: Optional hook called with the ``PredictionRequest``
                          before it is dispatched (e.g. for InfluxDB write).
        on_response_got:  Optional hook called with the ``PredictionResponse``
                          after it is received (e.g. for InfluxDB write).

    Returns:
        Decision/command dict from Core AI, or an error fallback dict.
    """
    obs = obs_data.get("observation", [0] * 10)

    # Build and validate the typed PredictionRequest
    prediction_request_model = build_prediction_request_from_obs(obs)

    # ── Hook: persist the request before sending ──────────────────────────────
    if on_request_built is not None:
        try:
            on_request_built(prediction_request_model)
        except Exception as exc:
            logger.error("on_request_built callback failed: %s", exc, exc_info=True)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            result = await client.post(
                f"{CORE_AI_URL}/predict",
                json=prediction_request_model.model_dump(),
            )
            result.raise_for_status()
            response_data = result.json()
            logger.info("📤 OBSERVATION SENT TO CORE AI (/predict)")

        # Validate the response into a typed model
        response_model = PredictionResponse(**response_data)

        # ── Hook: persist the response after receiving ────────────────────────
        if on_response_got is not None:
            try:
                on_response_got(response_model)
            except Exception as exc:
                logger.error("on_response_got callback failed: %s", exc, exc_info=True)

        return {
            "status":             "ok",
            "action_id":          response_model.action_id,
            "critical_source":    response_model.critical_source,
            "noncritical_source": response_model.noncritical_source,
            "inverter_total_w":   response_model.inverter_total_w,
            "overloaded":         response_model.overloaded,
            "constraint_applied": response_model.constraint_applied,
        }

    except Exception as exc:
        logger.error("❌ Error communicating with Core AI: %s", exc, exc_info=True)
        return {"status": "error", "error": str(exc)}
