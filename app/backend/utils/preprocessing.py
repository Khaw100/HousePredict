import pandas as pd
import numpy as np

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
        if col in X.columns:
            X[col] = X[col].astype(str).map(mapping).fillna(0).astype(int)

    missing = set(preprocess_config["selected_features"]) - set(X.columns)
    if missing:
        raise ValueError(f"Missing features after preprocessing: {missing}")
    
    X = X[preprocess_config["selected_features"]]
    for col in preprocess_config["log_transform_cols"]:
        if col in X.columns:
            X[col] = np.log1p(X[col].clip(lower=0))
    
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].clip(lower=0, upper=1e6)

    return X
