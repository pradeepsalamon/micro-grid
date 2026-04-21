from fastapi import FastAPI, HTTPException

from app.utils import (
    PredictionRequest,
    PredictionResponse,
    load_model,
    predict_payload,
)

app = FastAPI()

try:
    model = load_model()
except FileNotFoundError:
    model = None


@app.get("/health")
def health():
    return {"status": "Good"}



@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    print(f"Received prediction request: {payload}")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Ensure microgrid_model.zip exists.")

    return predict_payload(payload, model)

