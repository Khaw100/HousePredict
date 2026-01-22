import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import json
import joblib
import sys
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from tempfile import TemporaryDirectory

def build_features(df):
    df = df.copy()
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"]
        + df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"]
    )
    df["TotalBsmtFinSF"] = df["BsmtFinSF1"] + df["BsmtFinSF2"]
    return df


def preprocess_for_model(df, preprocess_config):
    X = df.copy()

    for col, mapping in preprocess_config["ordinal_maps"].items():
        X[col] = X[col].astype(str).map(mapping)

    X = X[preprocess_config["selected_features"]]

    for col in preprocess_config["log_transform_cols"]:
        X[col] = np.log1p(X[col])

    return X.values


def train_model(
    data_path,
    preprocess_config_path="/app/config/preprocess_config.pkl",
    experiment_name="HousePricePrediction",
):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(experiment_name)

    data = pd.read_csv(data_path)
    preprocess_config = joblib.load(preprocess_config_path)
    print(f"Features: {preprocess_config['selected_features']}")

    data = build_features(data)

    X = data[preprocess_config["selected_features"]]
    y = np.log1p(data["SalePrice"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model_params = {'subsample': 0.9, 'min_child_weight': 3, 'max_depth': 4, 'learning_rate': 0.03, 'gamma': 0, 'colsample_bytree': 0.7}
    X_train_p = preprocess_for_model(X_train, preprocess_config)
    X_val_p = preprocess_for_model(X_val, preprocess_config)

    with mlflow.start_run(run_name=f"xgb_{datetime.now():%Y%m%d_%H%M%S}"):

        model = XGBRegressor(
            objective="reg:squarederror",
            **model_params,
        )
        model.fit(X_train_p, y_train)

        y_pred = model.predict(X_val_p)

        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_val, y_pred)),
            "mae": mean_absolute_error(y_val, y_pred),
            "r2": r2_score(y_val, y_pred),
        }

        mlflow.log_metrics(metrics)
        mlflow.log_params(preprocess_config.get("model_params", {}))

        # ✅ 1. LOG MODEL
        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        # ✅ 2. LOG PREPROCESS CONFIG (TEMP SAFE)
        with TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "preprocess_config.pkl")
            joblib.dump(preprocess_config, cfg_path)
            mlflow.log_artifact(cfg_path, artifact_path="config")

        # ✅ 3. REGISTER MODEL
        run_id = mlflow.active_run().info.run_id
        mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name="housing-price-model",
        )

        print("✅ Training & registration completed")
        print(metrics)

        return run_id


if __name__ == "__main__":
    train_model(sys.argv[1] if len(sys.argv) > 1 else "data/train_cleaned.csv")
