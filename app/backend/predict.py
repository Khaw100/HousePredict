import mlflow.sklearn
import pandas as pd
from utils.preprocessing import build_features, preprocess_for_model
from utils.config import preprocess_config as PREPROCESS_CONFIG
import json
import numpy as np
import joblib
import os

from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# Environment & Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_NAME = "housing-price-model"
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")

LOGS_DIR = os.path.join(BASE_DIR, "logs")

METRICS_FILE = os.path.join(LOGS_DIR, "latest_metrics.json")
# Path absolut model
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_DIR = os.path.abspath(MODEL_DIR)

print("===================================")
print("Loading model from:", MODEL_DIR)
print("PATH MODEL")
print(MODEL_DIR)

if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Model not found at {MODEL_DIR}")
# load model
model = joblib.load(os.path.join(MODEL_DIR, "tuned_xgb_model_v1.pkl"))

print("Model loaded successfully.")
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
        # === 1. LOAD CSV (SAMA SEPERTI DEBUG) ===
        if "file" not in request.files:
            return jsonify({"error": "CSV file is required"}), 400

        file = request.files["file"]
        df = pd.read_csv(file)

        if df.empty:
            return jsonify({"error": "Uploaded CSV is empty"}), 400

        print("=== RAW DATA ===")
        print(df.head())
        print(df.isna().sum())

        # === 2. FEATURE ENGINEERING (SAMA) ===
        df = build_features(df)

        print("\n=== AFTER FEATURE ENGINEERING ===")
        print(df.head())
        # === 3. PREPROCESS (SAMA PERSIS) ===
        X = preprocess_for_model(df, PREPROCESS_CONFIG)
        

        print("\n=== AFTER PREPROCESS ===")
        print("Shape:", X.shape)
        print("dtype:", X.dtype)
        print("min:", np.min(X))
        print("max:", np.max(X))
        print("NaN:", np.isnan(X).any())
        print("Inf:", np.isinf(X).any())

        if np.isnan(X).any() or np.isinf(X).any():
            return jsonify({"error": "Invalid values after preprocessing"}), 400

        # === 4. MODEL PREDICT (SAMA) ===
        log_pred = model.predict(X)

        print("\n=== MODEL OUTPUT ===")
        print("Log prediction:", log_pred[:5])
        print("NaN:", np.isnan(log_pred).any())
        print("Inf:", np.isinf(log_pred).any())

        if np.isnan(log_pred).any() or np.isinf(log_pred).any():
            return jsonify({"error": "Model produced invalid predictions"}), 400

        # === 5. REVERSE LOG (SAMA) ===
        price = np.expm1(log_pred)

        print("\n=== FINAL PRICE ===")
        print(price[:5])

        prediction_count += len(price)

        for lp, p in zip(log_pred, price):
            prediction_log.append({
                "log_prediction": float(lp),
                "prediction": float(p),
                "timestamp": datetime.now().isoformat()
            })

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
    app.run(host="0.0.0.0", port=5000, debug=False)


