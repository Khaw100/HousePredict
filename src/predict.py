import mlflow.sklearn
import pandas as pd
import json
import os

from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# Load Model
MODEL_NAME = "housing-price-model"
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")

print(f"Loading model: {MODEL_NAME}, version: {MODEL_VERSION}")
model_url = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.sklearn.load_model(model_url)
print("Model loaded successfully.")

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
