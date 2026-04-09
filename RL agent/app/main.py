from fastapi import FastAPI

app = FastAPI()

from pydantic import BaseModel, Field
from typing import Literal

class TelemetryData(BaseModel):
    solar_power_w: float = Field(..., ge=0, description="Solar power in watts")
    wind_power_w: float = Field(..., ge=0, description="Wind power in watts")
    
    battery_soc: float = Field(
        ..., ge=0.0, le=1.0,
        description="Battery state of charge (0.0 to 1.0)"
    )

    critical_load_w: float = Field(..., ge=0, description="Critical load in watts")
    noncritical_load_w: float = Field(..., ge=0, description="Non-critical load in watts")

    grid_available: Literal[0, 1] = Field(
        ..., description="Grid status (1 = available, 0 = outage)"
    )

@app.post("/api/telemetry")
def receive_telemetry(data: TelemetryData):
    return {
        "status": "received",
        "total_load": data.critical_load_w + data.noncritical_load_w
    }
    
@app.get("/")
def home():
    return {"message": "Welcome to the Solar-Wind-Battery Telemetry API"}

