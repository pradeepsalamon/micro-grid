from fastapi import FastAPI, Request
from influxdb_client import InfluxDBClient, Point
import uvicorn, datetime

app = FastAPI(title="ESP8266 → InfluxDB Bridge")

# --- InfluxDB connection ---
INFLUX_URL = "http://influxdb-service.microgrid.svc.cluster.local:8086"  # 👈 Replace with your Minikube IP + NodePort
INFLUX_TOKEN = "3hPb_BB4ExjeATXlastgsVUdcuA3MMAw991Es19JkH1vZI56G6Dth2AGd8AB1ITbIiDAxNU-BqBSGupc7Mtd1g=="      
INFLUX_BUCKET = "energy_data"

client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = client.write_api(write_options=None)

@app.post("/espdata")
async def receive_data(request: Request):
    data = await request.json()
    print("📡 Received:", data)

    point = (
        Point("sensor_data")
        .tag("device_id", data.get("device_id", "esp8266"))
        .field("voltage", float(data.get("voltage", 0)))
        .field("current", float(data.get("current", 0)))
        .field("power", float(data.get("power", 0)))
        .time(datetime.datetime.utcnow())
    )

    write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
    return {"status": "ok", "message": "Data written to InfluxDB"}

@app.get("/")
def home():
    return {"status": "running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
