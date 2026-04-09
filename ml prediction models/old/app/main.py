from fastapi import FastAPI
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import random
import threading
import time

app = FastAPI(title="Microgrid FastAPI - InfluxDB Data Generator")

# ⚙️ Direct InfluxDB Configuration (No env vars)
INFLUX_URL = "http://influxdb-service.microgrid.svc.cluster.local:8086"
INFLUX_TOKEN = "3hPb_BB4ExjeATXlastgsVUdcuA3MMAw991Es19JkH1vZI56G6Dth2AGd8AB1ITbIiDAxNU-BqBSGupc7Mtd1g=="  # paste your real token here
INFLUX_ORG = "microgrid"
INFLUX_BUCKET = "energy_data"

# Initialize InfluxDB client
client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

# 🔁 Global control flags
is_running = False
thread = None


def send_fake_data():
    """Continuously send fake energy data to InfluxDB."""
    global is_running
    while is_running:
        voltage = round(random.uniform(220, 240), 2)
        current = round(random.uniform(2, 4), 2)
        power = round(voltage * current, 2)

        point = (
            Point("power")
            .tag("location", "microgrid_lab")
            .field("voltage", voltage)
            .field("current", current)
            .field("power", power)
            .time(time.time_ns(), WritePrecision.NS)
        )

        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        print(f"✅ Sent data → V={voltage}V | I={current}A | P={power}W")

        time.sleep(1)


@app.get("/")
def home():
    """Check API availability."""
    return {"message": "FastAPI is running! Use /start and /stop to control data streaming."}


@app.get("/start")
def start_stream():
    """Start generating fake data."""
    global is_running, thread

    if is_running:
        return {"status": "already_running", "message": "Fake data generation is already active."}

    is_running = True
    thread = threading.Thread(target=send_fake_data, daemon=True)
    thread.start()

    return {"status": "started", "message": "Fake data generation started!"}


@app.get("/stop")
def stop_stream():
    """Stop fake data generation."""
    global is_running

    if not is_running:
        return {"status": "not_running", "message": "No data generation in progress."}

    is_running = False
    return {"status": "stopped", "message": "Fake data generation stopped!"}


@app.get("/status")
def get_status():
    """Check current streaming status."""
    return {"is_running": is_running}
