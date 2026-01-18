import pandas as pd
import numpy as np
from utils.config import preprocess_config as PREPROCESS_CONFIG



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

def preprocess_for_model(df, preprocess_config):
    X = df.copy()

    for col, mapping in preprocess_config["ordinal_maps"].items():
        X[col] = X[col].astype(str).map(mapping).fillna(0).astype(float) 

    X = X[preprocess_config["selected_features"]]
    print("FEATURES:", X.columns.tolist())
    for col in preprocess_config["log_transform_cols"]:
        X[col] = np.log1p(X[col])

    X = X.astype(float)
    return X.values.astype(np.float32)

