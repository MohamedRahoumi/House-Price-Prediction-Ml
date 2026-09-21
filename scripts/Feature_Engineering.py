from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "outputs" / "X_prepared.csv"
OUTPUT_PATH = ROOT / "outputs" / "House_Prices_engineered.csv"


df = pd.read_csv(DATA_PATH)

print("Donnees avant :", df.shape)


df["TotalSF"] = (
    df["TotalBsmtSF"].fillna(0)
    + df["1stFlrSF"].fillna(0)
    + df["2ndFlrSF"].fillna(0)
)


df["TotalBathrooms"] = (
    df["FullBath"].fillna(0)
    + 0.5 * df["HalfBath"].fillna(0)
    + df["BsmtFullBath"].fillna(0)
    + 0.5 * df["BsmtHalfBath"].fillna(0)
)


df["HouseAgeAtSale"] = (
    df["YrSold"] - df["YearBuilt"]
).clip(lower=0)


df["YearsSinceRemodel"] = (
    df["YrSold"] - df["YearRemodAdd"]
).clip(lower=0)


df["TotalPorchSF"] = (
    df["WoodDeckSF"].fillna(0)
    + df["OpenPorchSF"].fillna(0)
    + df["EnclosedPorch"].fillna(0)
    + df["3SsnPorch"].fillna(0)
    + df["ScreenPorch"].fillna(0)
)


df["HasGarage"] = (
    df["GarageArea"].fillna(0) > 0
).astype(int)


df["HasBasement"] = (
    df["TotalBsmtSF"].fillna(0) > 0
).astype(int)


df["HasPool"] = (
    df["PoolArea"].fillna(0) > 0
).astype(int)


OUTPUT_PATH.parent.mkdir(exist_ok=True)


df.to_csv(OUTPUT_PATH, index=False)


print("Données après :", df.shape)

print("Nouvelles variables :")
print([
    "TotalSF",
    "TotalBathrooms",
    "HouseAgeAtSale",
    "YearsSinceRemodel",
    "TotalPorchSF",
    "HasGarage",
    "HasBasement",
    "HasPool"
])

print("Fichier sauvegardé :", OUTPUT_PATH)