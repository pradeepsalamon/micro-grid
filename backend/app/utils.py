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
    print("📊 Observation extracted:", obs_data["observation"])
    
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
    theft_endpoint = "theft-prediction?current=4"
    if latest_telemetry and "loads" in latest_telemetry and "grid" in latest_telemetry["loads"]:
        grid_load = latest_telemetry["loads"]["grid"]
        theft_endpoint = f"theft-prediction?current={grid_load}"
    
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
    
    # Extract values with defaults
    solar_power = pred.get("solar_prediction", {}).get("power_output", 0) / 500
    wind_power = pred.get("wind_prediction", {}).get("power_output", 0) / 500
    
    # From telemetry
    battery_soc = latest_telemetry.get("sources", {}).get("battery_soc", 0)
    critical_load = latest_telemetry.get("loads", {}).get("critical", 0) / 300
    noncritical_load = latest_telemetry.get("loads", {}).get("non_critical", 0) / 300
    grid_available = int(latest_telemetry.get("info", {}).get("grid_available", False))
    
    # Forecasts from predictions
    solar_forecast = pred.get("solar_prediction", {}).get("efficiency", 0) / 500
    wind_forecast = pred.get("wind_prediction", {}).get("efficiency", 0) / 500
    load_forecast = pred.get("load_prediction", {}).get("scaled_load", 0)  # Already normalized
    
    # Power cut probability
    power_cut_prob = pred.get("power_cut_prediction", {}).get("prob_cut", 0)
    
    # Build observation array
    obs = [
        solar_power,
        wind_power,
        battery_soc,
        critical_load,
        noncritical_load,
        grid_available,
        solar_forecast,
        wind_forecast,
        load_forecast,
        power_cut_prob
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


async def send_observation_to_core_ai(predictions: Dict) -> Dict:
    """
    Extract observation and send to Core AI controller.
    
    Args:
        predictions: Response from ML prediction service
        
    Returns:
        Response from Core AI or error
    """
    obs_data = extract_observation(predictions)
    
    try:
        result = await httpx.AsyncClient().post(
            f"http://{HOST}:{RL_PORT}/observe",
            json=obs_data
        )
        result.raise_for_status()
        print("\n📤 OBSERVATION SENT TO CORE AI:", obs_data["observation"])
        return result.json()
    except (httpx.RequestError, Exception) as exc:
        return {"error": str(exc), "observation": obs_data["observation"]}


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
        "load_forecast_w":      round(obs[8] * 300, 4),
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

            # Convert PredictionResponse → command dict
            inverter_on  = 1 if response.get("inverter_total_w", 0) > 0 else 0
            critical_inv = 1 if response.get("critical_source") == "inverter" else 0
            noncrit_inv  = 1 if response.get("noncritical_source") == "inverter" else 0

            return {
                "status": "ok",
                "action_id":          response.get("action_id"),
                "critical_source":    response.get("critical_source"),
                "noncritical_source": response.get("noncritical_source"),
                "inverter_total_w":   response.get("inverter_total_w"),
                "overloaded":         response.get("overloaded"),
                "constraint_applied": response.get("constraint_applied"),
                "observation":        obs,
                "command": {
                    "inverter": [inverter_on, critical_inv]
                }
            }

    except (httpx.RequestError, Exception) as exc:
        error_response = {
            "status": "error",
            "error": str(exc),
            "observation": obs,
            "command": {"inverter": [1, 0]}  # Safe fallback
        }
        print("❌ Error communicating with Core AI:", str(exc))
        return error_response
