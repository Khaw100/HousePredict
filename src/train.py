import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import json 
import joblib
import sys
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

def build_features(df):
    df = df.copy()

    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"] +
        df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"]
    )
    df["TotalBsmtFinSF"] = df["BsmtFinSF1"] + df["BsmtFinSF2"]

    return df


# preprocess_config = {
#     "selected_features": [
#         "OverallQual", "TotalSF", "GrLivArea", "HouseAge",
#         "GarageArea", "GarageCars", "KitchenQual", "BsmtQual",
#         "YearRemodAdd", "LotArea", "TotalBsmtFinSF",
#         "LotFrontage", "MasVnrArea", "TotalPorchSF"
#     ],
#     "log_transform_cols": [
#         "LotFrontage", "LotArea", "MasVnrArea",
#         "GrLivArea", "TotalSF", "TotalPorchSF", "TotalBsmtFinSF"
#     ],
#     "ordinal_maps": {
#         "KitchenQual": {'NA':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5},
#         "BsmtQual": {'NA':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
#     }
# }


def preprocess_for_model(
    df,
    preprocess_config,
    scaler=None,
    fit=False,
    use_scaler=True
):
    X = df.copy()

    selected_features = preprocess_config["selected_features"]
    log_transform_cols = preprocess_config["log_transform_cols"]
    ordinal_maps = preprocess_config["ordinal_maps"]

    # Ordinal encoding
    for col, mapping in ordinal_maps.items():
        X[col] = X[col].astype(str).map(mapping)

    # Select features
    X = X[selected_features]

    # Log transform
    for col in log_transform_cols:
        X[col] = np.log1p(X[col])

    # Scaling (optional)
    if not use_scaler:
        return X.values, None

    if scaler is None:
        scaler = RobustScaler()

    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler

# This function manually write by me 
def train_model(data_path,
                preprocess_config_path="models/preprocess_config.pkl",
                experiment_name="HousePricePrediction"):
    
    # Set up MLflow experiment
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)

    # Load data
    print("Loading data...")
    data = pd.read_csv(data_path)

    # Prepare Configurations
    preprocess_config = joblib.load(preprocess_config_path)
    
    # Feature Engineering
    print("Building features...")
    data = build_features(data)

    # Safety check
    missing_cols = set(preprocess_config["selected_features"]) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing columns after build_features: {missing_cols}")

    # Target and Features
    X = data[preprocess_config["selected_features"]]
    y = np.log1p(data["SalePrice"])

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess data
    print("Preprocessing data...")

    # Not using scaler for XGBoost because it's tree-based
    X_train_processed, _ = preprocess_for_model(X_train, preprocess_config, use_scaler=False)
    X_val_processed, _ = preprocess_for_model(X_val, preprocess_config, use_scaler=False)

    # Train model
    print("Training model...")
    with mlflow.start_run(
        run_name=f"xgb_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        # Prepare parameters
        params = preprocess_config.get("model_params", {})

        # Save Logged parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "XGBRegressor")
        mlflow.log_param("use_scaler", False)
        mlflow.log_param("Selected Features", preprocess_config["selected_features"])

        # Initialize and train model
        print("Initializing and training XGBRegressor...")
        model = XGBRegressor(
            objective="reg:squarederror",
            **params
        )

        model.fit(X_train_processed, y_train)
        print("Model training completed.")

        # Validate model
        print("Validating model...")
        y_val_pred = model.predict(X_val_processed)

        # Calculate metrics
        mse = mean_squared_error(y_val, y_val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        r2 = r2_score(y_val, y_val_pred)
        mae = mean_absolute_error(y_val, y_val_pred)
        metrics = {"mse": mse, "rmse": rmse, "mae": mae, "r2_score": r2}
        
        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(model, "model", registered_model_name="housing-price-model")
        print("Model validation completed.")

        # Log additional info
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("validation_samples", len(X_val))

        # Save metrics to file for monitoring
        with open("logs/latest_metrics.json", "w") as f:
            json.dump({
                **metrics,
                "timestamp": datetime.now().isoformat(),
                "run_id": mlflow.active_run().info.run_id
            }, f, indent=4)
        
        print("Training process completed.")
        print(f"Metrics:")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"MAE: ${mae:,.2f}")
        print(f"R²: {r2:.4f}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

        return mlflow.active_run().info.run_id
    

if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/train_cleaned.csv"
    train_model(data_path)


