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
