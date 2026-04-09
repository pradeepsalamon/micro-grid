import fastapi
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENWEATHER_API_KEY")

app = fastapi.FastAPI()

@app.get("/")
def read_root():
    return {"service": "ML prediction service is running"}

@app.get("/solar-prediction")
def solar_predict():
    
    return {"prediction": 10}

@app.get("/wind-prediction")
def wind_predict():
    
    return {"prediction": 70}


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)