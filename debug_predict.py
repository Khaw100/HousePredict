import pandas as pd
import joblib
import numpy as np

import pandas as pd
import numpy as np

# utils/config.py

preprocess_config = {
    "selected_features": [
        "OverallQual", "TotalSF", "GrLivArea", "HouseAge",
        "GarageArea", "GarageCars", "KitchenQual", "BsmtQual",
        "YearRemodAdd", "LotArea", "TotalBsmtFinSF",
        "LotFrontage", "MasVnrArea", "TotalPorchSF"
    ],
    "log_transform_cols": [
        "LotFrontage", "LotArea", "MasVnrArea",
        "GrLivArea", "TotalSF", "TotalPorchSF", "TotalBsmtFinSF"
    ],
    "ordinal_maps": {
        "KitchenQual": {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        "BsmtQual": {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    },
}



def build_features(df):
    df = df.copy()
    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"] +
        df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"]
    )
    df["TotalBsmtFinSF"] = df["BsmtFinSF1"] + df["BsmtFinSF2"]
    return df

def preprocess_for_model(df, preprocess_config):
    X = df.copy()
    for col, mapping in preprocess_config["ordinal_maps"].items():
        X[col] = X[col].astype(str).map(mapping)
    X = X[preprocess_config["selected_features"]]
    print("FEATURES:", X.columns.tolist())
    for col in preprocess_config["log_transform_cols"]:
        X[col] = np.log1p(X[col])

    return X.values


# Load test data
df = pd.read_csv("data/test_housing.csv")

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
