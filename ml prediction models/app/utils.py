import joblib
import pandas as pd
from datetime import datetime
import requests
import os
import numpy as np


# ============================
# MODEL LOADING
# ============================
solar_model = joblib.load("models/solar.pkl")
wind_model = joblib.load("models/wind.pkl")
load_model = joblib.load("models/load.pkl")
power_cut_model = joblib.load("models/powercut.pkl")
theft_model = joblib.load("models/anomaly.pkl")


# ============================
# CONFIGURATION & CONSTANTS
# ============================
LOCATION = {"lat": 11.9224, "lon": 79.6067}
PANEL_CAPACITY_KW = 1.0
TURBINE_CAPACITY_KW = 2.0

IRR_MIN = 0.9022
IRR_MAX = 7.3234

SOLAR_CSV_FILE = os.path.join("data", "solar.csv")
WIND_CSV_FILE = os.path.join("data", "wind.csv")
LOAD_CSV_FILE = os.path.join("data", "load.csv")
WEATHER_CACHE_TTL = 60  # 1 minute

# ============================
# GLOBAL CACHE
# ============================
_weather_cache = {"data": None, "timestamp": None}
_historical_data_cache = None


# ============================
# CENTRALIZED API & DATA FETCHING
# ============================

def fetch_weather_data():
    
    global _weather_cache
    
    now = datetime.now()
    
    # Return cached data if still valid
    if (_weather_cache["data"] is not None and 
        _weather_cache["timestamp"] is not None):
        age = (now - _weather_cache["timestamp"]).total_seconds()
        if age < WEATHER_CACHE_TTL:
            return _weather_cache["data"]
    
    # Fetch new weather data
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY environment variable is not set")
    
    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "lat": LOCATION["lat"],
                "lon": LOCATION["lon"],
                "appid": api_key
            },
            timeout=10
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch weather data: {str(e)}")
    
    data = response.json()
    
    if "main" not in data:
        raise ValueError(f"Invalid API response: {data.get('message', 'Unknown error')}")
    
    # Update cache
    _weather_cache["data"] = data
    _weather_cache["timestamp"] = now
    
    return data


def fetch_historical_data(current_date, current_hour, csv_file, column_name, default_value=0.8):
    
    global _historical_data_cache
    
    # Load CSV once and cache it (cache key is the file path)
    if _historical_data_cache is None:
        _historical_data_cache = {}
    
    if csv_file not in _historical_data_cache:
        _historical_data_cache[csv_file] = pd.read_csv(csv_file)
    
    df = _historical_data_cache[csv_file]
    
    # Handle different CSV structures
    current_month = datetime.now().month
    
    if "month" in df.columns:
        # CSV has separate month column (like load.csv)
        row = df[(df["month"] == current_month) & (df["hour"] == current_hour)]
    elif "date" in df.columns:
        # CSV has date column in YYYYMMDD format (like solar.csv, wind.csv)
        # Extract month from date column
        df_copy = df.copy()
        df_copy["month"] = df_copy["date"].astype(str).str[4:6].astype(int)
        row = df_copy[(df_copy["month"] == current_month) & (df_copy["hour"] == current_hour)]
    else:
        # Fallback: no month/hour filtering possible
        row = df.head(1)  # Just take first row as fallback
    
    if not row.empty and column_name in row.columns:
        return float(row.iloc[0][column_name])
    
    return default_value  # default fallback


def fetch_historical_load(current_date, current_hour, csv_file=None):
    """
    Fetch historical load with caching to avoid repeated CSV reads.
    (Legacy function - now uses fetch_historical_data)
    """
    if csv_file is None:
        csv_file = LOAD_CSV_FILE
    return fetch_historical_data(current_date, current_hour, csv_file, "historical_load", 0.8)


def extract_common_features(weather_data, csv_file=None):
    """
    Extract common features from weather data.
    Returns a dictionary with normalized datetime and weather features.
    """
    temp_c = weather_data["main"]["temp"] - 273.15
    wind_speed = weather_data["wind"]["speed"]
    humidity = weather_data["main"]["humidity"]
    
    current_time = datetime.fromtimestamp(weather_data["dt"])
    hour = current_time.hour
    month = current_time.month
    
    current_date = int(datetime.now().strftime("%Y%m%d"))
    hist_eff = fetch_historical_data(current_date, hour, csv_file, "historical_efficiency", 0.0)
    
    return {
        "temperature": temp_c,
        "wind_speed": wind_speed,
        "humidity": humidity,
        "hour": hour,
        "month": month,
        "historical_efficiency": hist_eff,
        "weather_data": weather_data,
        "current_time": current_time
    }


def get_model_irradiance(weather_data):
    """
    Convert OpenWeather API → dataset-scale irradiance.
    """
    clouds = weather_data["clouds"]["all"]
    current_time = weather_data["dt"]
    sunrise = weather_data["sys"]["sunrise"]
    sunset = weather_data["sys"]["sunset"]

    current = datetime.fromtimestamp(current_time)
    rise = datetime.fromtimestamp(sunrise)
    set_ = datetime.fromtimestamp(sunset)

    # Night
    if current < rise or current > set_:
        return 0.0

    # Daylight
    daylight = (set_ - rise).total_seconds()
    elapsed = (current - rise).total_seconds()

    solar_factor = np.sin(np.pi * elapsed / daylight)
    cloud_factor = (100 - clouds) / 100

    # Normalized (0–1)
    norm_irr = solar_factor * cloud_factor

    # Scale to dataset range
    scaled_irr = IRR_MIN + norm_irr * (IRR_MAX - IRR_MIN)

    return round(max(0, scaled_irr), 4)


# ============================
# PREDICTION FUNCTIONS
# ============================

def predict_solar():
    """
    Predict solar power output using centralized API calls.
    """
    # Fetch weather data (uses cache if available)
    weather_data = fetch_weather_data()
    
    # Extract common features using solar.csv for historical efficiency
    features = extract_common_features(weather_data, SOLAR_CSV_FILE)
    
    # Calculate irradiance (solar-specific)
    irr = get_model_irradiance(weather_data)
    print(f"☀️ Irradiance (model scale): {irr}")

    # Prepare model input
    input_data = {
        "month": features["month"],
        "hour": features["hour"],
        "temperature": features["temperature"],
        "wind_speed": features["wind_speed"],
        "irradiance": irr,
        "historical_efficiency": features["historical_efficiency"]
    }

    df = pd.DataFrame([input_data])

    # Predict
    efficiency = solar_model.predict(df)[0]
    efficiency = max(0, min(efficiency, 100))
    power_output = (efficiency / 100) * PANEL_CAPACITY_KW

    # Output
    print(f"⚡ Efficiency: {round(efficiency, 2)} %")
    print(f"☀️ Solar Output: {round(power_output, 2)} kW")

    return {
        "efficiency": round(efficiency, 2),
        "power_output": round(power_output, 2),
    }


def predict_wind():
    """
    Predict wind power output using centralized API calls.
    """
    # Fetch weather data (uses cache if available)
    weather_data = fetch_weather_data()
    
    # Extract common features using wind.csv for historical efficiency
    features = extract_common_features(weather_data, WIND_CSV_FILE)

    print(f"🌬️ Wind Speed: {features['wind_speed']}")
    print(f"📊 Historical Efficiency: {features['historical_efficiency']}")

    # Prepare model input
    input_data = {
        "month": features["month"],
        "hour": features["hour"],
        "temperature": features["temperature"],
        "wind_speed": features["wind_speed"],
        "historical_efficiency": features["historical_efficiency"]
    }

    df = pd.DataFrame([input_data])

    # Predict
    efficiency = wind_model.predict(df)[0]
    efficiency = max(0, min(efficiency, 100))
    power_output = (efficiency / 100) * TURBINE_CAPACITY_KW

    # Output
    print(f"🌬️ Efficiency: {round(efficiency, 2)} %")
    print(f"⚡ Wind Output: {round(power_output, 2)} kW")

    return {
        "efficiency": round(efficiency, 2),
        "power_output": round(power_output, 2),
        "wind_speed": features["wind_speed"]
    }


# ============================
# ADDITIONAL MODEL FUNCTIONS
# ============================

def predict_powercut(threshold=0.5):
    """Predict probability of power cut from powercut model using current weather."""
    weather_data = fetch_weather_data()
    current_time = datetime.fromtimestamp(weather_data["dt"])

    input_data = {
        "month": current_time.month,
        "temperature": weather_data["main"]["temp"] - 273.15,
        "humidity": weather_data["main"]["humidity"],
        "wind_speed": weather_data["wind"]["speed"],
        "rainfall": float(weather_data.get("rain", {}).get("1h", weather_data.get("rain", {}).get("3h", 0.0))),
        "cloud": weather_data["clouds"]["all"],
    }

    feature_order = [
        "month",
        "temperature",
        "humidity",
        "wind_speed",
        "rainfall",
        "cloud",
    ]
    df = pd.DataFrame([input_data])[feature_order]
    proba = power_cut_model.predict_proba(df)[0]
    prob_cut_prob = float(proba[1])
    is_cut = int(prob_cut_prob > threshold)

    return {
        "prob_no_cut": 1 - is_cut,
        "prob_cut": is_cut,
        "threshold": threshold,
    }


def detect_anomaly(live_input):
    """Detect anomaly/theft using the anomaly model."""
    hour = datetime.now().hour
    input_data = {
        "hour": hour,
        "voltage": live_input.get("voltage"),
        "current": live_input.get("current"),
        "consumption": (live_input.get("voltage", 230.0) * live_input.get("current", 5.5)) / 1000,
    }
    df = pd.DataFrame([input_data])
    prediction = int(theft_model.predict(df)[0])
    anomaly = int(prediction == -1)

    return {
        "prediction": prediction,
        "anomaly": anomaly,
    }


def predict_load(house_max_kw=1.0, dataset_csv=LOAD_CSV_FILE):
    """Predict household load using current weather and historical load from CSV."""
    weather_data = fetch_weather_data()
    current_time = datetime.fromtimestamp(weather_data["dt"])
    
    # Fetch historical load from CSV using generic function
    current_date = int(datetime.now().strftime("%Y%m%d"))
    historical_load = fetch_historical_data(current_date, current_time.hour, dataset_csv, "historical_load", 0.8)

    input_data = {
        "hour": current_time.hour,
        "day": current_time.weekday(),
        "month": current_time.month,
        "temperature": weather_data["main"]["temp"] - 273.15,
        "humidity": weather_data["main"]["humidity"],
        "historical_load": historical_load,
    }

    df = pd.DataFrame([input_data])
    raw_load = float(load_model.predict(df)[0])
    data = pd.read_csv(dataset_csv)
    dataset_max_kw = float(data["load"].max()) if "load" in data else 1.0

    scaled_load = (raw_load / dataset_max_kw) * house_max_kw if dataset_max_kw > 0 else 0.0
    scaled_load = max(0.0, min(scaled_load, house_max_kw))

    return {
        "scaled_load": round(scaled_load, 2),
        "house_max_kw": house_max_kw,
    }
    
    

# ============================
# WEATHER FUNCTIONS
# ============================

def get_weather_info():
    """
    Get formatted weather information for API response.
    Returns current weather conditions with processed data.
    """
    weather_data = fetch_weather_data()
    current_time = datetime.fromtimestamp(weather_data["dt"])
    
    # Calculate additional weather metrics
    irradiance = get_model_irradiance(weather_data)
    
    return {
        "location": {
            "latitude": LOCATION["lat"],
            "longitude": LOCATION["lon"],
            "city": weather_data.get("name", "Unknown")
        },
        "timestamp": {
            "current_time": current_time.isoformat(),
            "timezone_offset": weather_data.get("timezone", 0)
        },
        "temperature": {
            "celsius": round(weather_data["main"]["temp"] - 273.15, 2),
            "fahrenheit": round((weather_data["main"]["temp"] - 273.15) * 9/5 + 32, 2),
            "feels_like_celsius": round(weather_data["main"]["feels_like"] - 273.15, 2)
        },
        "weather": {
            "main": weather_data["weather"][0]["main"],
            "description": weather_data["weather"][0]["description"],
            "icon": weather_data["weather"][0]["icon"]
        },
        "atmospheric": {
            "humidity_percent": weather_data["main"]["humidity"],
            "pressure_hpa": weather_data["main"]["pressure"],
            "visibility_meters": weather_data.get("visibility", "N/A")
        },
        "wind": {
            "speed_mps": weather_data["wind"]["speed"],
            "direction_degrees": weather_data["wind"].get("deg", "N/A"),
            "gust_mps": weather_data["wind"].get("gust", "N/A")
        },
        "clouds": {
            "coverage_percent": weather_data["clouds"]["all"]
        },
        "solar": {
            "irradiance": irradiance,
            "sunrise": datetime.fromtimestamp(weather_data["sys"]["sunrise"]).isoformat(),
            "sunset": datetime.fromtimestamp(weather_data["sys"]["sunset"]).isoformat()
        },
        "precipitation": {
            "rain_1h_mm": weather_data.get("rain", {}).get("1h", 0.0),
            "rain_3h_mm": weather_data.get("rain", {}).get("3h", 0.0),
            "snow_1h_mm": weather_data.get("snow", {}).get("1h", 0.0),
            "snow_3h_mm": weather_data.get("snow", {}).get("3h", 0.0)
        }
    }

