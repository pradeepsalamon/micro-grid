from fastapi import FastAPI, HTTPException
import uvicorn
import httpx
from .models import PowerData, PredictionRequest, PredictionResponse
from .utils import process_telemetry_and_get_command, fetch_predictions, get_weather, get_decision, send_observation_to_core_ai, HOST, RL_PORT

app = FastAPI()

@app.get("/")
def read_root():
    return {"service": "middleware service is running"}

@app.post("/api/telemetry")
async def receive_telemetry(data: PowerData):
    """
    Receive telemetry data → Fetch forecasts → Send to Core AI → Return command.
    
    Flow:
    1. Store telemetry from ESP32
    2. Fetch ML forecast models
    3. Extract normalized observation
    4. Send to Core AI for decision
    5. Return decision/command to ESP32
    """
    decision = await process_telemetry_and_get_command(data.dict())
    
    return {
        "status": "ok",
        "command": decision.get("command", {"inverter": [1, 0]}),
        "decision": decision
    }



@app.get("/predict")
async def predict():
    """Fetch predictions from ML service and send observation to Core AI."""
    predictions = await fetch_predictions()
    
    # Send observation to Core AI
    if not predictions.get("errors"):
        await send_observation_to_core_ai(predictions)
    
    return predictions

@app.get("/weather")
def get_weather_data():
    """Fetch weather data from ML service."""
    return get_weather()

@app.post("/decision")
def get_decision_endpoint():
    """Fetch decision from RL service."""
    return get_decision()


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest):
    """
    Forward a structured PredictionRequest to the Core AI (/predict)
    and return a PredictionResponse.

    Response format:
    {
      "action_id": 0,
      "critical_source": "grid",
      "noncritical_source": "grid",
      "inverter_total_w": 0.0,
      "overloaded": false,
      "constraint_applied": false
    }
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            result = await client.post(
                f"http://{HOST}:{RL_PORT}/predict",
                json=payload.dict()
            )
            result.raise_for_status()
            return result.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Core AI error: {exc.response.text}"
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Core AI unreachable: {exc}"
        )

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)