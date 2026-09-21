import pandas as pd

df = pd.read_csv("data_sauvage/House_Prices.csv")

print(df.head())

print("Shape :", df.shape)

print("Colonnes :")
print(df.columns.tolist())

print("Informations :")
print(df.info())

print("Valeurs manquantes :")
print(df.isnull().sum())

print("Nombre de doublons :", df.duplicated().sum())

df = df.drop_duplicates()

df = df[
    [
        "GrLivArea",
        "OverallQual",
        "YearBuilt",
        "FullBath",
        "BedroomAbvGr",
        "GarageCars",
        "SalePrice"
    ]
]

print("Valeurs manquantes :")
print(df.isnull().sum())

df = df.dropna()

print("Types :")
print(df.dtypes)

print("Statistiques :")
print(df.describe())

X = df[
    [
        "GrLivArea",
        "OverallQual",
        "YearBuilt",
        "FullBath",
        "BedroomAbvGr",
        "GarageCars"
    ]
]

y = df["SalePrice"]

print("X :")
print(X.head())

print("y :")
print(y.head())