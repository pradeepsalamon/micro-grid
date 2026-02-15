
# 📘 ML DEVELOPMENT DOCUMENT

## Project: AI-Based Energy Management System for Microgrid

### ML Module Owner: Hari

***************please maintain a seperate venv and requirements.txt, use python 3.11.0*************

---

# 1️⃣ SYSTEM OVERVIEW

The Microgrid AI system contains 5 ML models:

| Model No | Model Name                  | Type              |
| -------- | --------------------------- | ----------------- |
| 1        | Solar Generation Prediction | Regression        |
| 2        | Wind Generation Prediction  | Regression        |
| 3        | Load Demand Prediction      | Regression        |
| 4        | Energy Theft Detection      | Anomaly Detection |
| 5        | Power Cut Prediction        | Classification    |

Weather data source: **OpenWeather API**, 

---

# 2️⃣ WEATHER DATA INTEGRATION (OpenWeather API)

## 🌦 Data Required from API

From OpenWeather API, extract:

* Temperature
* Humidity
* Wind speed
* Cloud coverage
* Pressure
* Rainfall
* Weather condition
* Timestamp

These will be used as input features for:

* Solar model
* Wind model
* Power cut model
* Load model (optional)

---

## 🧑‍💻 API Fetching Example (Hari Reference)

```python
import requests

API_KEY = "your_api_key"
city = "Chennai"
url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

response = requests.get(url)
data = response.json()

temperature = data["main"]["temp"]
humidity = data["main"]["humidity"]
wind_speed = data["wind"]["speed"]
clouds = data["clouds"]["all"]
```

Store data in database for training.

---

# 3️⃣ MODEL 1: Solar Generation Prediction

## 🎯 Objective

Predict next hour solar power output (kW).

---

## 📊 Input Features

| Feature                         | Source      |
| ------------------------------- | ----------- |
| Temperature                     | OpenWeather |
| Humidity                        | OpenWeather |
| Cloud Coverage                  | OpenWeather |
| Solar Irradiance (if available) | Sensor      |
| Time (hour)                     | System      |
| Historical Solar Output         | Sensor      |

---

## 🎯 Target

Solar power output (kW)

---

## 🤖 Recommended Model

Start with:

* RandomForestRegressor (scikit-learn)

Upgrade later:

* LSTM (if required)

---

## 📈 Evaluation Metrics

* MAE
* RMSE
* R² Score

---

# 4️⃣ MODEL 2: Wind Generation Prediction

## 🎯 Objective

Predict wind turbine power output.

---

## 📊 Input Features

| Feature                | Source      |
| ---------------------- | ----------- |
| Wind Speed             | OpenWeather |
| Temperature            | OpenWeather |
| Pressure               | OpenWeather |
| Historical Wind Output | Sensor      |

---

## 🎯 Target

Wind power output (kW)

---

## 🤖 Recommended Model

* RandomForestRegressor
* Gradient Boosting

Note: Wind power is nonlinear → Tree models work well.

---

# 5️⃣ MODEL 3: Load Demand Prediction (Core Model)

## 🎯 Objective

Predict electricity consumption for next hour/day.

---

## 📊 Input Features

| Feature         | Source      |
| --------------- | ----------- |
| Historical Load | Smart Meter |
| Hour of Day     | System      |
| Day of Week     | System      |
| Holiday Flag    | Manual      |
| Temperature     | OpenWeather |
| Humidity        | OpenWeather |

---

## 🎯 Target

Load demand (kW)

---

## 🤖 Recommended Model

Start:

* RandomForestRegressor

Upgrade:

* LSTM / GRU (Time Series)

---

## 📈 Metrics

* MAE
* RMSE
* R²

This model is most important.

---

# 6️⃣ MODEL 4: Energy Theft Detection (Anomaly Detection)

## 🎯 Objective

Detect abnormal energy usage patterns.

---

## 📊 Input Features

| Feature                  | Source      |
| ------------------------ | ----------- |
| Hourly Consumption       | Smart Meter |
| Voltage                  | Grid Sensor |
| Current                  | Grid Sensor |
| Historical Usage Pattern | Database    |

---

## 🤖 Recommended Model

### Easiest:

IsolationForest

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.05)
model.fit(X_train)
```

Output:

* 1 → Normal
* -1 → Anomaly

---

## 📈 Evaluation

* Confusion Matrix
* Precision
* Recall
* F1-score

---

# 7️⃣ MODEL 5: Power Cut Prediction (Grid Failure Prediction)

## 🎯 Objective

Predict possible main grid outage.

---

## 📊 Input Features

| Feature          | Source      |
| ---------------- | ----------- |
| Grid Voltage     | Sensor      |
| Wind Speed       | OpenWeather |
| Rainfall         | OpenWeather |
| Storm Condition  | OpenWeather |

---

## 🎯 Target

0 → Normal
1 → Power cut expected

---

## 🤖 Recommended Model

* RandomForestClassifier
* Logistic Regression

---

## 📈 Metrics

* Accuracy
* Confusion Matrix
* ROC-AUC

---

# 8️⃣ DATA PREPROCESSING (IMPORTANT)

Hari must:

1. Handle missing values
2. Convert timestamps → hour/day features
3. Normalize data (if required)
4. Remove outliers (for regression models)
5. Split dataset (80% train / 20% test)

---

# 9️⃣ PROJECT FOLDER STRUCTURE

```

ml_module/
│
├── data/
├── models/
│   ├── solar_model.pkl
│   ├── wind_model.pkl
│   ├── load_model.pkl
│   ├── anomaly_model.pkl
│   └── powercut_model.pkl
│
├── train_solar.py
├── train_wind.py
├── train_load.py
├── train_anomaly.py
├── train_powercut.py
│
└── utils.py
```

---

# 1️⃣1️⃣ Development Order (For Hari)

Step 1 → Solar Model
Step 2 → Load Model
Step 3 → Wind Model
Step 4 → Anomaly Model
Step 5 → Power Cut Model
Step 6 → API Integration

---

# 1️⃣2️⃣ Final Deliverables From Hari

✔ Trained models (.pkl files)
✔ Model accuracy report
✔ Graph: Actual vs Predicted
✔ Confusion matrix (for classification)
✔ Proper documentation

---

# 🎯 Final Advice to Hari

* Start simple (Random Forest for everything)
* Make models work first
* Then improve accuracy
* Don’t try deep learning unless time permits
* Focus on clean data and evaluation
