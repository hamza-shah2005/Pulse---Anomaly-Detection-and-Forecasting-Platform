# Pulse — Anomaly Detection & Forecasting Platform

> **AI-powered analytics platform for real-time anomaly detection, energy forecasting, and industrial IoT monitoring.**

---

## 🚀 Overview

**Pulse** is a full-stack AI analytics platform built as a Final Year Project (FYP). It ingests time-series data — from CSV uploads or live machine sensor feeds — and runs a complete ML pipeline: data preprocessing, anomaly detection, consumption forecasting, root cause analysis (RCA), and AI-driven reporting.

Designed to generalize across industries: **energy management**, **industrial IoT**, **predictive maintenance**, and **IT infrastructure monitoring**.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📂 **CSV & Live Feed Ingestion** | Upload historical datasets or stream live sensor readings from machines |
| 🧹 **Auto Preprocessing** | Removes duplicates, fills missing values, and normalizes time-series data |
| 🔍 **Anomaly Detection** | Isolation Forest & Local Outlier Factor (LOF) with configurable contamination |
| 📈 **Consumption Forecasting** | Prophet & ARIMA models with MAE, RMSE, MAPE evaluation metrics |
| 🏭 **Industrial IoT Monitoring** | Real-time machine sensor tracking (CPU, temp, RAM, disk, power, network) |
| ⚠️ **Downtime Detection** | Automatically flags anomaly windows as downtime events with severity levels |
| 🔬 **Root Cause Analysis (RCA)** | AI-generated RCA reports identifying top contributing sensors and recommendations |
| ⚡ **Energy Forecasting** | Predicts future power consumption per machine with confidence bounds |
| 🤖 **AI Optimization Advice** | Machine-specific optimization suggestions powered by LLM analysis |
| 📊 **Interactive Dashboards** | Per-machine and per-dataset dashboards with live charts |
| 🔐 **JWT Authentication** | Secure user accounts with token-based auth and profile management |

---

## 🧠 ML Models Used

- **Isolation Forest** — unsupervised anomaly detection, robust to high-dimensional sensor data
- **Local Outlier Factor (LOF)** — density-based anomaly detection
- **Prophet** — Facebook's time-series forecasting model, handles seasonality and holidays
- **ARIMA** — classical statistical forecasting for stationary time-series

---

## 🏗️ Tech Stack

### Backend
- **Django 6** + **Django REST Framework** — REST API & template rendering
- **Celery** + **Redis** — asynchronous ML task queue
- **PostgreSQL** / **SQLite** — relational data storage
- **scikit-learn**, **Prophet**, **statsmodels**, **pandas**, **numpy** — ML & data processing
- **JWT (SimpleJWT)** — authentication

### Infrastructure
- **Redis** — message broker for Celery workers
- **psutil** — live system/machine metric collection

---

## 📐 Architecture

```
User / Frontend
      │
      ▼
Django REST API  ──►  Celery Task Queue  ──►  ML Workers
      │                                         │
      ▼                                         ▼
  PostgreSQL                         Anomaly Detection
  (Datasets,                         Forecasting
   Machines,                         RCA Generation
   Readings,                         Energy Optimization
   Forecasts,
   Downtime Events)
```

---

## 📦 Installation

### Prerequisites
- Python 3.11+
- Redis server running locally
- PostgreSQL (or use SQLite for development)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/pulse-anomaly-forecasting.git
cd pulse-anomaly-forecasting/Backend

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp myproject/.env.example myproject/.env
# Edit .env with your DB credentials, secret key, and Redis URL

# 5. Apply database migrations
python myproject/manage.py migrate

# 6. Start the Django development server
python myproject/manage.py runserver

# 7. Start Celery worker (new terminal)
celery -A myproject worker --loglevel=info
```

---

## 🔌 Key API Endpoints

### Authentication
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/register/` | Register a new user |
| `POST` | `/api/token/` | Obtain JWT token pair |
| `POST` | `/api/token/refresh/` | Refresh access token |

### Dataset Pipeline
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload/` | Upload a CSV dataset |
| `GET` | `/dataset/{id}/status/` | Poll processing status |
| `POST` | `/dataset/{id}/preprocess/` | Run data preprocessing |
| `POST` | `/dataset/{id}/detect-anomalies/` | Run anomaly detection |
| `POST` | `/dataset/{id}/forecast/` | Run consumption forecasting |
| `GET` | `/dataset/{id}/anomalies/data/` | Get anomaly results |
| `GET` | `/dataset/{id}/forecast/data/` | Get forecast results |

### Industrial IoT / Machines
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/machines/` | List all registered machines |
| `POST` | `/api/machines/{id}/detect-anomalies/` | Detect sensor anomalies |
| `GET` | `/api/machines/{id}/downtimes/` | List downtime events |
| `GET` | `/api/machines/{id}/downtimes/{event_id}/rca/` | Get RCA report |
| `GET` | `/api/machines/{id}/energy-forecast/` | Get energy forecast |
| `GET` | `/api/machines/{id}/optimization/` | Get optimization advice |
| `GET` | `/api/machines/{id}/dashboard/` | Full dashboard data |

---

## 🏭 Industry Use Cases

- **Energy Utilities** — detect abnormal consumption spikes, forecast load demand
- **Manufacturing** — monitor CNC machines, compressors, motors for predictive maintenance
- **IT Operations** — track server/PC resource anomalies, prevent downtime
- **Facilities Management** — building energy monitoring and optimization

---

## 📊 Dataset Support

Pulse accepts any time-series CSV with:
- A **timestamp** column (auto-detected)
- A **value** column (configurable target column)

Examples: energy meter readings, server CPU logs, vibration sensor data, temperature streams.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

[MIT](LICENSE)

---

## 👨‍💻 Author

Built as a Final Year Project (FYP) — demonstrating production-grade ML integration in a full-stack web platform.

---

> *"From raw sensor data to actionable intelligence — in one platform."*
