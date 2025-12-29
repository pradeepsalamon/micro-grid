from fastapi import FastAPI, Request
from typing import Optional, Dict
from .models import PowerData

app = FastAPI(
    title="ESP API",
    root_path="/esp"
)

# ================================
# STORAGE FOR TELEMETRY + COMMANDS
# ================================
latest_telemetry: Dict = {}
count = 0


# ================================
# TELEMETRY MODEL
# ================================


# ================================
# ESP32 SENDS DATA → SERVER
# POST /api/telemetry
# ================================
@app.post("/api/telemetry")
async def receive_telemetry(data: PowerData):
    global latest_telemetry
    global count
    count += 1
    latest_telemetry = data.dict()
    print("\n📥 TELEMETRY RECEIVED:", latest_telemetry)

    # Decide command BASED ON telemetry
    command = None

    if True:
        command = {
            "action": f"{count},3.0,500.0,30",
            "sockets": ["non_critical"]
        }

    # If no command → respond normally
    if command is None:
        return {
            "status": "ok",
            "command": None
        }

    # If command exists → send it NOW
    print("📤 COMMAND SENT:", command)
    return {
        "status": "ok",
        "command": command
    }




# ================================
# ROOT ENDPOINT
# ================================
@app.get("/")
async def root():
    return {"status": "FastAPI Microgrid Server Running"}


