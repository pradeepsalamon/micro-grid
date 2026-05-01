import fastapi
import uvicorn
import os
from dotenv import load_dotenv
from . utils import  predict_solar, predict_wind, predict_powercut, predict_load, detect_anomaly, get_weather_info

load_dotenv()


app = fastapi.FastAPI()

@app.get("/")
def read_root():
    return {"service": "ML prediction service is running"}

@app.get("/solar-prediction")
def solar_predict():

    return predict_solar()

@app.get("/wind-prediction")
def wind_predict():
    
    return predict_wind()

@app.get("/powercut-prediction")
def powercut_predict():
    return predict_powercut()

@app.get("/load-prediction")
def load_predict():
    return predict_load()

@app.get("/theft-prediction")
def theft_predict(voltage: float = 230, current: float = 1.0):
    input_data = {
        "voltage": voltage,
        "current": current,
    }
    return detect_anomaly(input_data)

@app.get("/weather-data")
def get_weather():

    return get_weather_info()

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)