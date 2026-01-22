import mlflow.sklearn
import pandas as pd
from utils.preprocessing import build_features, preprocess_for_model
import json
import numpy as np
import joblib
import os
from mlflow.tracking import MlflowClient
# from xgboost import XGBRegressor
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# Environment & Paths
LOGS_DIR = "/app/logs"

MODEL_NAME = "housing-price-model"
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")

# CONFIG_PATH = "/app/config/preprocess_config.pkl"
# PREPROCESS_CONFIG = joblib.load(CONFIG_PATH)

PREDICTION_LOG_FILE = os.path.join(LOGS_DIR, "predictions.jsonl")
METRICS_FILE = os.path.join(LOGS_DIR, "latest_metrics.json")

os.makedirs(LOGS_DIR, exist_ok=True)


print("===================================")
print(f"   Loading model from MLflow Registry")
print(f"   Model: {MODEL_NAME}")
print(f"   Version: {MODEL_VERSION}")
print("===================================")

# MLFLOW model Registry URI
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
client = MlflowClient()

# Resolve model version (alias -> fallback latest)
try:
    print("Trying to load model using alias: champion")
    model_version = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias="champion"
    )
except Exception:
    print("Alias 'champion' not found, fallback to latest version")
    latest = client.get_latest_versions(MODEL_NAME)
    model_version = latest[0]

RUN_ID = model_version.run_id
MODEL_VERSION = model_version.version

print("===================================")
print("   Loading model from MLflow Registry")
print(f"   Model: {MODEL_NAME}")
print(f"   Version: {MODEL_VERSION}")
print(f"   Run ID: {RUN_ID}")
print("===================================")

# Load model
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.sklearn.load_model(model_uri)

# Load preprocessing config from same run
config_path = mlflow.artifacts.download_artifacts(
    run_id=RUN_ID,
    artifact_path="config/preprocess_config.pkl"
)
PREPROCESS_CONFIG = joblib.load(config_path)

print("Model & preprocessing config loaded successfully.")
print("===================================")

# Prediction counter fo monitoring
prediction_count = 0
prediction_log = []

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
                        "status": "healthy",
                        "model_name": MODEL_NAME,
                        "model_version": MODEL_VERSION,
                        "predictions_served": prediction_count,
                        "timestamp": datetime.now().isoformat()
                    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    global prediction_count, prediction_log

    try:
        if "file" not in request.files:
            return jsonify({"error": "CSV file is required"}), 400

        file = request.files["file"]
        df = pd.read_csv(file)

        if df.empty:
            return jsonify({"error": "Uploaded CSV is empty"}), 400

        # Feature Engineering
        df = build_features(df)
        X = preprocess_for_model(df, PREPROCESS_CONFIG)

        if np.isnan(X).any() or np.isinf(X).any():
            return jsonify({"error": "Invalid values after preprocessing"}), 400

        # Predict
        log_pred = model.predict(X)

        if np.isnan(log_pred).any() or np.isinf(log_pred).any():
            return jsonify({"error": "Model produced invalid predictions"}), 400
        
        price = np.expm1(log_pred)
        prediction_count += len(price)

        # Logging
        for i, (lp, p) in enumerate(zip(log_pred, price)):
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "input": df.iloc[i].to_dict(),   # input data
                "prediction": float(p),
                "model_version": MODEL_VERSION
            }
            # /metrics
            prediction_log.append(log_entry)

            # Log for mmonitoring
            with open(PREDICTION_LOG_FILE, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        return jsonify({
            "predictions": price.tolist(),
            "predictions_served": prediction_count
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/metrics", methods=["GET"])
def metrics():
    training_metrics = {}

    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            training_metrics = json.load(f)

    return jsonify({
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "training_metrics": training_metrics,
        "total_predictions": prediction_count,
        "recent_predictions": prediction_log[-10:],
        "timestamp": datetime.now().isoformat()
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)


