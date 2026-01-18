import pandas as pd
import joblib
import numpy as np

from utils.preprocessing import build_features, preprocess_for_model
from utils.config import preprocess_config

# Load test data
df = pd.read_csv("../data/test_housing.csv")

print("=== RAW DATA ===")
print(df.head())
print(df.isna().sum())

# Feature engineering
df = build_features(df)

print("\n=== AFTER FEATURE ENGINEERING ===")
print(df.head())

# Preprocess
X = preprocess_for_model(df, preprocess_config)

print("\n=== AFTER PREPROCESS ===")
print("Shape:", X.shape)
print("dtype:", X.dtype)
print("min:", np.min(X))
print("max:", np.max(X))
print("NaN:", np.isnan(X).any())
print("Inf:", np.isinf(X).any())

# Load model
model = joblib.load("models/tuned_xgb_model_v1.pkl")

# Predict
log_pred = model.predict(X)

print("\n=== MODEL OUTPUT ===")
print("Log prediction:", log_pred[:5])
print("NaN:", np.isnan(log_pred).any())
print("Inf:", np.isinf(log_pred).any())

# Reverse log
price = np.expm1(log_pred)

print("\n=== FINAL PRICE ===")
print(price[:5])
print("NaN:", np.isnan(price).any())
print("Inf:", np.isinf(price).any())
