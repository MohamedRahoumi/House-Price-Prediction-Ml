import pandas as pd
from sklearn.preprocessing import StandardScaler 
import os

df = pd.read_csv("data_sauvage/House_Prices.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum().sort_values(ascending=False).head(15))
print("Doublons :", df.duplicated().sum())
print(df["SalePrice"].describe())

df = df.drop_duplicates()
y = df["SalePrice"]
X = df.drop(columns=["SalePrice", "Id"])

colonnes_numeriques = X.select_dtypes(include="number").columns
colonnes_categorielles = X.select_dtypes(exclude="number").columns

X[colonnes_numeriques] = X[colonnes_numeriques].fillna(X[colonnes_numeriques].median())
X[colonnes_categorielles] = X[colonnes_categorielles].fillna("Missing")

scaler = StandardScaler()
X[colonnes_numeriques] = scaler.fit_transform(X[colonnes_numeriques])
X = pd.get_dummies(X, columns=colonnes_categorielles, dtype=int)

print("Valeurs manquantes dans X :", X.isnull().sum().sum())
print("Dimensions de X :", X.shape)
print("Dimensions de y :", y.shape)

os.makedirs("outputs", exist_ok=True)

X.to_csv("outputs/X_prepared.csv", index=False)
y.to_csv("outputs/y_prepared.csv", index=False)

print("Preparation terminee.")