import os
import requests
import httpx
from typing import Dict, Tuple
from dotenv import load_dotenv

# ================================
# CONFIGURATION
# ================================
load_dotenv()

HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT", 8000))
ML_PORT = int(os.getenv("ML-PORT", 8001))
RL_PORT = int(os.getenv("RL-PORT", 8002))

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


async def process_telemetry_and_get_command(telemetry_data: Dict) -> Dict:
    """
    Complete flow: Store telemetry → Fetch forecasts → Extract observation → 
    Send to Core AI → Return decision/command.
    
    Args:
        telemetry_data: Raw telemetry data from ESP32
        
    Returns:
        Dictionary with command/decision from Core AI
    """
    global latest_telemetry
    latest_telemetry = telemetry_data
    print("\n📥 TELEMETRY RECEIVED:", latest_telemetry)
    
    # Step 1: Fetch forecasts from ML models
    print("🔄 Fetching forecasts from ML models...")
    predictions = await fetch_predictions()
    
    if predictions.get("errors"):
        print("⚠️ Errors fetching predictions:", predictions["errors"])
    
    # Step 2: Extract observation
    obs_data = extract_observation(predictions)
    
    # Step 3: Send observation to Core AI and get decision
    print("🤖 Sending observation to Core AI...")
    decision = await send_observation_to_core_ai_and_get_decision(obs_data)
    
    print("✅ Decision received:", decision)
    
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
                result = await client.get(f"http://{HOST}:{ML_PORT}/{endpoint}")
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
        result = requests.get(f"http://{HOST}:{ML_PORT}/weather-data")
        result.raise_for_status()
        return result.json()
    except requests.RequestException as exc:
        return {"error": str(exc)}


def get_decision() -> Dict:
    """
    Fetch decision from RL service.
    
    Returns:
        Decision data or error message
    """
    try:
        result = requests.get(f"http://{HOST}:{RL_PORT}/decision")
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




async def send_observation_to_core_ai_and_get_decision(obs_data: Dict) -> Dict:
    """
    Send observation to Core AI controller (/predict) and get decision/command.

    Builds a PredictionRequest from the observation array and maps the
    PredictionResponse back to a command dict.

    Args:
        obs_data: Extracted observation data with observation array and metadata

    Returns:
        Decision/command from Core AI or error fallback
    """
    obs = obs_data.get("observation", [0] * 10)

    # Map obs array indices → PredictionRequest fields
    # obs: [solar_power, wind_power, battery_soc, critical_load,
    #        noncritical_load, grid_available, solar_forecast,
    #        wind_forecast, load_forecast, power_cut_prob]
    prediction_request = {
        "solar_power_w":        round(obs[0] * 500, 4),   # denormalise
        "wind_power_w":         round(obs[1] * 500, 4),
        "battery_soc":          max(0.0, min(1.0, obs[2])),
        "critical_load_w":      round(obs[3] * 300, 4),
        "noncritical_load_w":   round(obs[4] * 300, 4),
        "grid_available":       int(obs[5]),
        "solar_forecast_w":     round(obs[6] * 500, 4),
        "wind_forecast_w":      round(obs[7] * 500, 4),
        "load_forecast_w":      round(obs[8] * 1000, 4),   # scaled_load (0-1) × 1000W house max
        "power_cut_probability": max(0.0, min(1.0, obs[9])),
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            result = await client.post(
                f"http://{HOST}:{RL_PORT}/predict",
                json=prediction_request
            )
            result.raise_for_status()
            response = result.json()   # PredictionResponse
            print("📤 OBSERVATION SENT TO CORE AI (/predict)")

            return {
                "status": "ok",
                "action_id":          response.get("action_id"),
                "critical_source":    response.get("critical_source"),
                "noncritical_source": response.get("noncritical_source"),
                "inverter_total_w":   response.get("inverter_total_w"),
                "overloaded":         response.get("overloaded"),
                "constraint_applied": response.get("constraint_applied"),
            }

    except (httpx.RequestError, Exception) as exc:
        error_response = {
            "status": "error",
            "error": str(exc),
            # "command": {"inverter": [1, 0]}  # Safe fallback
        }
        print("❌ Error communicating with Core AI:", str(exc))
        return error_response
