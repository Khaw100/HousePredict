import mlflow.sklearn
import pandas as pd
from utils.preprocessing import build_features, preprocess_for_model
from utils.config import preprocess_config
import json
import numpy as np
import joblib
import os

from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# Environment & Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_NAME = "housing-price-model"
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")

LOGS_DIR = os.path.join(BASE_DIR, "logs")

METRICS_FILE = os.path.join(LOGS_DIR, "latest_metrics.json")
# Path absolut model
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_DIR = os.path.abspath(MODEL_DIR)


# =============================
# Load Model
# =============================
print("===================================")
print("Loading model from:", MODEL_DIR)

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
        data = request.get_json()

        # Validate input
        if not data or "inputs" not in data:
            return jsonify({"error": "Missing 'inputs'"}), 400

        if not isinstance(data["inputs"], list) or len(data["inputs"]) == 0:
            return jsonify({"error": "'inputs' must be non-empty list"}), 400

        input_df = pd.DataFrame(data["inputs"])

        # Feature engineering
        input_df = build_features(input_df)
        
        # Preprocess data (mapping and log transform + select features)
        x_processed = preprocess_for_model(input_df, preprocess_config)

        # Make predictions
        log_predictions = model.predict(x_processed)

        # Reverse log1p transformation
        predictions = np.expm1(log_predictions)          

        # Monitoring
        prediction_count += len(predictions)
        
        # Log predictions with timestamp
        for i, pred in enumerate(predictions):
            prediction_log.append({
                "input": data["inputs"][i],
                "prediction": float(pred),
                "timestamp": datetime.now().isoformat()
            })
        
        return jsonify({
            "predictions": predictions.tolist(),
            "predictions_served": prediction_count
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

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


