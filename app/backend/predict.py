import mlflow.sklearn
import pandas as pd
from utils.preprocessing import build_features, preprocess_for_model
import json
import numpy as np
import joblib
import os
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# Environment & Paths
LOGS_DIR = "/app/logs"
MODEL_NAME = "housing-price-model"
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")

PREDICTION_LOG_FILE = os.path.join(LOGS_DIR, "predictions.jsonl")
METRICS_FILE = os.path.join(LOGS_DIR, "latest_metrics.json")

os.makedirs(LOGS_DIR, exist_ok=True)

print("===================================")
print(f"   Loading model from MLflow Registry")
print(f"   Model: {MODEL_NAME}")
print(f"   Version: {MODEL_VERSION}")
print("===================================")

# MLFLOW model Registry URI
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

print("✅ Model & preprocessing config loaded successfully.")
print("===================================")

# Prediction counter for monitoring
prediction_count = 0
prediction_log = []


def log_prediction_to_file(engineered_data, prediction_value, model_version):
    try:
        features_to_log = ["OverallQual", "TotalSF", "GrLivArea", "HouseAge", "GarageArea", "GarageCars",
                           "KitchenQual", "BsmtQual", "YearRemodAdd", "LotArea", "TotalBsmtFinSF", "LotFrontage",
                           "MasVnrArea", "TotalPorchSF"]
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prediction": float(prediction_value),
            "model_version": str(model_version),
            "features": {k: engineered_data.get(k) for k in features_to_log}
        }
        
        # Append to JSONL file
        with open(PREDICTION_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        return True
    except Exception as e:
        print(f"Warning: Failed to log prediction: {e}")
        return False


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
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
        # Check if file is provided
        if "file" not in request.files:
            return jsonify({"error": "CSV file is required"}), 400

        file = request.files["file"]
        df = pd.read_csv(file)

        if df.empty:
            return jsonify({"error": "Uploaded CSV is empty"}), 400

        # Feature Engineering
        df_engineered = build_features(df)
        X = preprocess_for_model(df_engineered, PREPROCESS_CONFIG)

        # Validate preprocessed data
        if np.isnan(X).any() or np.isinf(X).any():
            return jsonify({"error": "Invalid values after preprocessing"}), 400

        # Predict (log scale)
        log_pred = model.predict(X)

        # Validate predictions
        if np.isnan(log_pred).any() or np.isinf(log_pred).any():
            return jsonify({"error": "Model produced invalid predictions"}), 400
        
        # Convert back to original scale
        price = np.expm1(log_pred)
        prediction_count += len(price)

        # Log each prediction individually
        logged_count = 0
        for i, (lp, p) in enumerate(zip(log_pred, price)):

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "prediction": float(p),
                "model_version": MODEL_VERSION
            }
            prediction_log.append(log_entry)

            engineered_row = df_engineered.iloc[i].to_dict()
            if log_prediction_to_file(engineered_row, p, MODEL_VERSION):
                logged_count += 1

        print(f"📝 Logged {logged_count}/{len(price)} predictions to {PREDICTION_LOG_FILE}")

        return jsonify({
            "predictions": price.tolist(),
            "count": len(price),
            "predictions_served": prediction_count,
            "logged_for_monitoring": logged_count,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    """Get model metrics and prediction statistics"""
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


@app.route("/monitoring/stats", methods=["GET"])
def monitoring_stats():
    try:
        predictions = []
        
        if not os.path.exists(PREDICTION_LOG_FILE):
            return jsonify({
                "total_predictions": 0,
                "message": "No predictions logged yet"
            })
        
        with open(PREDICTION_LOG_FILE, "r") as f:
            for line in f:
                predictions.append(json.loads(line))
        
        if not predictions:
            return jsonify({
                "total_predictions": 0,
                "message": "No predictions in file"
            })
        
        pred_values = [p["prediction"] for p in predictions]
        
        return jsonify({
            "total_predictions": len(predictions),
            "recent_predictions": predictions[-10:],  # Last 10
            "statistics": {
                "mean": float(np.mean(pred_values)),
                "std": float(np.std(pred_values)),
                "min": float(min(pred_values)),
                "max": float(max(pred_values)),
                "range": float(max(pred_values) - min(pred_values))
            },
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)