from fastapi import FastAPI
import uvicorn
import requests
import httpx
import os
from dotenv import load_dotenv

app = FastAPI()

HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", 8000))
ML_PORT = int(os.getenv("ML-PORT", 8001))

@app.get("/")
def read_root():
    return {"service": "middleware service is running"}

@app.get("/predict")
async def predict():
    # Make a request to the solar prediction service
    response = {}
    async with httpx.AsyncClient() as client:
        solar_prediction = await client.get(f"http://{HOST}:{ML_PORT}/solar-prediction")
        if solar_prediction.status_code == 200:
            response["solar_prediction"] = solar_prediction.json()
        else:
            return {"error": "Failed to get solar prediction"}
    
        wind_prediction = await client.get(f"http://{HOST}:{ML_PORT}/wind-prediction")
        if wind_prediction.status_code == 200:
            response["wind_prediction"] = wind_prediction.json()
        else:
            return {"error": "Failed to get wind prediction"}
        
        return response

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)