from fastapi import FastAPI, Request
from typing import Optional, Dict
from .models import PowerData

app = FastAPI(
    # title="ESP API",
    # root_path="/esp"
)

# ================================
# STORAGE FOR TELEMETRY + COMMANDS
# ================================
latest_telemetry: Dict = {}
count = 0
cr = False
nc = False

@app.get("/cr")
def changeCr():
    global cr
    cr = not cr
    
@app.get("/nc")
def changeNc():
    global nc
    nc = not nc


# ================================
# ESP32 SENDS DATA → SERVER
# POST /api/telemetry
# ================================
@app.post("/api/telemetry")
async def receive_telemetry(data: PowerData):
    global latest_telemetry
    latest_telemetry = data.dict()
    print("\n📥 TELEMETRY RECEIVED:", latest_telemetry)

    # Decide command BASED ON telemetry
    command = None

    if True:
        command = {
            "inverter": [1 , 0]
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


