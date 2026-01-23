import mlflow.sklearn
import pandas as pd
import json
import numpy as np
import joblib
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient
from datetime import datetime

from utils.preprocessing import build_features, preprocess_for_model

app = FastAPI(title="House Price Prediction API")

# =====================
# Environment & Paths
# =====================
LOGS_DIR = "/app/logs"
MODEL_NAME = "housing-price-model"
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")

PREDICTION_LOG_FILE = os.path.join(LOGS_DIR, "predictions.jsonl")
METRICS_FILE = os.path.join(LOGS_DIR, "latest_metrics.json")

os.makedirs(LOGS_DIR, exist_ok=True)

# =====================
# Load Model from MLflow
# =====================
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
client = MlflowClient()

try:
    model_version = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias="champion"
    )
except Exception:
    latest = client.get_latest_versions(MODEL_NAME)
    model_version = latest[0]

RUN_ID = model_version.run_id
MODEL_VERSION = model_version.version

model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.sklearn.load_model(model_uri)

config_path = mlflow.artifacts.download_artifacts(
    run_id=RUN_ID,
    artifact_path="config/preprocess_config.pkl"
)
PREPROCESS_CONFIG = joblib.load(config_path)

print("✅ Model & preprocessing config loaded")

# =====================
# Monitoring variables
# =====================
prediction_count = 0
prediction_log = []

# =====================
# Helpers
# =====================
def log_prediction_to_file(engineered_data, prediction_value):
    features_to_log = [
        "OverallQual", "TotalSF", "GrLivArea", "HouseAge",
        "GarageArea", "GarageCars", "KitchenQual", "BsmtQual",
        "YearRemodAdd", "LotArea", "TotalBsmtFinSF",
        "LotFrontage", "MasVnrArea", "TotalPorchSF"
    ]

    entry = {
        "timestamp": datetime.now().isoformat(),
        "prediction": float(prediction_value),
        "model_version": MODEL_VERSION,
        "features": {k: engineered_data.get(k) for k in features_to_log}
    }

    with open(PREDICTION_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# =====================
# Endpoints
# =====================
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "predictions_served": prediction_count,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global prediction_count

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV file required")

    df = pd.read_csv(file.file)

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty")

    # Feature engineering
    df_engineered = build_features(df)
    X = preprocess_for_model(df_engineered, PREPROCESS_CONFIG)

    if np.isnan(X).any() or np.isinf(X).any():
        raise HTTPException(status_code=400, detail="Invalid values after preprocessing")

    # Predict (log scale → real price)
    log_pred = model.predict(X)
    price = np.expm1(log_pred)

    prediction_count += len(price)

    logged = 0
    for i, p in enumerate(price):
        prediction_log.append({
            "timestamp": datetime.now().isoformat(),
            "prediction": float(p),
            "model_version": MODEL_VERSION
        })

        log_prediction_to_file(df_engineered.iloc[i].to_dict(), p)
        logged += 1

    return {
        "predictions": price.tolist(),
        "count": len(price),
        "predictions_served": prediction_count,
        "logged_for_monitoring": logged,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics")
def metrics():
    training_metrics = {}
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            training_metrics = json.load(f)

    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "training_metrics": training_metrics,
        "total_predictions": prediction_count,
        "recent_predictions": prediction_log[-10:],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/monitoring/stats")
def monitoring_stats():
    if not os.path.exists(PREDICTION_LOG_FILE):
        return {
            "total_predictions": 0,
            "message": "No predictions logged yet"
        }

    predictions = []
    with open(PREDICTION_LOG_FILE) as f:
        for line in f:
            predictions.append(json.loads(line))

    pred_values = [p["prediction"] for p in predictions]

    return {
        "total_predictions": len(predictions),
        "recent_predictions": predictions[-10:],
        "statistics": {
            "mean": float(np.mean(pred_values)),
            "std": float(np.std(pred_values)),
            "min": float(min(pred_values)),
            "max": float(max(pred_values)),
            "range": float(max(pred_values) - min(pred_values)),
        },
        "timestamp": datetime.now().isoformat()
    }
