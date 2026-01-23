import pandas as pd
import numpy as np
from faker import Faker
import os
import json

fake = Faker()

columns = [
    "Id","MSSubClass","MSZoning","LotFrontage","LotArea","Street","Alley","LotShape","LandContour",
    "Utilities","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType",
    "HouseStyle","OverallQual","OverallCond","YearBuilt","YearRemodAdd","RoofStyle","RoofMatl",
    "Exterior1st","Exterior2nd","MasVnrType","MasVnrArea","ExterQual","ExterCond","Foundation",
    "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinSF1","BsmtFinType2","BsmtFinSF2",
    "BsmtUnfSF","TotalBsmtSF","Heating","HeatingQC","CentralAir","Electrical","1stFlrSF",
    "2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath",
    "BedroomAbvGr","KitchenAbvGr","KitchenQual","TotRmsAbvGrd","Functional","Fireplaces",
    "FireplaceQu","GarageType","GarageYrBlt","GarageFinish","GarageCars","GarageArea","GarageQual",
    "GarageCond","PavedDrive","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch",
    "PoolArea","PoolQC","Fence","MiscFeature","MiscVal","MoSold","YrSold","SaleType","SaleCondition"
]

def generate_dummy_and_alert(n_samples_csv=10, file_name_csv="dummy_house_data",
                             threshold_samples=20):
    data = []
    for i in range(n_samples_csv):
        # Untuk kolom numerik utama, kita buat range besar supaya bisa trigger drift
        total_bsmt = np.random.randint(0, 3000)
        gr_liv = np.random.randint(0, 4000)
        first_flr = np.random.randint(400, 2000)

        row = [
            i + 1,
            np.random.choice([20, 30, 40, 60, 120, 160]),  # MSSubClass
            np.random.choice(["RL", "RH", "RM", "FV"]),
            np.random.randint(20, 100),  # LotFrontage
            np.random.randint(5000, 20000),  # LotArea
            np.random.choice(["Pave", "Grvl"]),
            np.random.choice([np.nan, "Grvl", "Pave"]),
            np.random.choice(["Reg", "IR1", "IR2", "IR3"]),
            np.random.choice(["Lvl", "Bnk", "HLS", "Low"]),
            "AllPub",
            np.random.choice(["Inside", "Corner", "CulDSac"]),
            np.random.choice(["Gtl", "Mod", "Sev"]),
            fake.city(),
            np.random.choice(["Norm", "Feedr", "PosN", "Artery"]),
            np.random.choice(["Norm", "Feedr", "PosN", "Artery"]),
            np.random.choice(["1Fam","2FmCon","Duplex","TwnhsE","Twnhs"]),
            np.random.choice(["1Story","2Story","1.5Fin","SLvl"]),
            np.random.randint(1,10),  # OverallQual
            np.random.randint(1,10),  # OverallCond
            np.random.randint(1900,2022),  # YearBuilt
            np.random.randint(1900,2022),  # YearRemodAdd
            np.random.choice(["Gable","Hip","Flat","Mansard"]),
            np.random.choice(["CompShg","Metal","Tar&Grv"]),
            np.random.choice(["VinylSd","MetalSd","HdBoard","CemntBd","BrkFace"]),
            np.random.choice(["VinylSd","MetalSd","HdBoard","CemntBd","BrkFace"]),
            np.random.choice(["None","BrkFace","Stone","CBlock"]),
            np.random.randint(0, 500),  # MasVnrArea
            np.random.choice(["Ex","Gd","TA","Fa"]),
            np.random.choice(["Ex","Gd","TA","Fa"]),
            np.random.choice(["PConc","CBlock","BrkTil","Slab"]),
            np.random.choice(["Ex","Gd","TA","Fa","No"]),
            np.random.choice(["Ex","Gd","TA","Fa","No"]),
            np.random.choice(["Gd","Av","Mn","No"]),
            np.random.choice(["GLQ","ALQ","Unf","Rec","LwQ","BLQ","No"]),
            np.random.randint(0, 1500),
            np.random.choice(["GLQ","ALQ","Unf","Rec","LwQ","BLQ","No"]),
            np.random.randint(0, 1500),
            np.random.randint(0, 1500),
            total_bsmt,
            np.random.choice(["GasA","GasW","Grav","Wall"]),
            np.random.choice(["Ex","Gd","TA","Fa"]),
            np.random.choice(["Y","N"]),
            np.random.choice(["SBrkr","FuseA","FuseF","FuseP"]),
            first_flr,
            np.random.randint(0, 1500),
            np.random.randint(0, 200),
            gr_liv,
            np.random.randint(0,3),
            np.random.randint(0,2),
            np.random.randint(0,4),
            np.random.randint(0,2),
            np.random.randint(1,5),
            np.random.randint(1,2),
            np.random.choice(["Ex","Gd","TA","Fa"]),
            np.random.randint(3,12),
            np.random.choice(["Typ","Min1","Min2","Mod","Maj1","Maj2","Sev","Sal"]),
            np.random.randint(0,3),
            np.random.choice(["Ex","Gd","TA","Fa","No"]),
            np.random.choice(["Attchd","Detchd","BuiltIn","CarPort","Basment","2Types","No"]),
            np.random.randint(1900,2022),
            np.random.choice(["Fin","RFn","Unf"]),
            np.random.randint(0,4),
            np.random.randint(0,1000),
            np.random.choice(["Ex","Gd","TA","Fa","No"]),
            np.random.choice(["Ex","Gd","TA","Fa","No"]),
            np.random.choice(["Y","P","N"]),
            np.random.randint(0,500),
            np.random.randint(0,500),
            np.random.randint(0,500),
            np.random.randint(0,200),
            np.random.randint(0,200),
            np.random.randint(0,800),
            np.random.choice([np.nan,"Ex","Gd","TA","Fa"]),
            np.random.choice([np.nan,"GdPrv","MnPrv","GdWo","MnWw"]),
            np.random.choice([np.nan,"Shed","Gar2","Othr"]),
            np.random.randint(0,5000),
            np.random.randint(1,13),
            np.random.randint(2006,2021),
            np.random.choice(["WD","New","COD","Con","ConLw"]),
            np.random.choice(["Normal","Abnorml","Partial","AdjLand","Family","Alloca"])
        ]
        data.append(row)
    
    df = pd.DataFrame(data, columns=columns)
    os.makedirs("data", exist_ok=True)
    file_path_csv = f"data/{file_name_csv}.csv"
    df.to_csv(file_path_csv, index=False)
    print(f"✅ Dummy CSV data saved to {file_path_csv}")