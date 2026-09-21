"""Chargement et préparation simple des données House Prices."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ANNEE_REFERENCE = 2010
FEATURE_COLUMNS = [
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "GrLivArea",
    "TotalBsmtSF",
    "TotalSF",
    "TotalBathrooms",
    "HouseAge",
    "RemodAge",
    "GarageCars",
    "GarageArea",
    "LotArea",
    "Neighborhood",
    "ExterQual",
    "KitchenQual",
    "MSZoning",
    "FullBath",
    "BedroomAbvGr",
]


def ajouter_features(df, annee_reference = ANNEE_REFERENCE):
    donnees = df.copy()

    colonnes_necessaires = [
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "FullBath",
        "HalfBath",
        "BsmtFullBath",
        "BsmtHalfBath",
        "YearBuilt",
        "YearRemodAdd",
    ]
    colonnes_absentes = [
        colonne for colonne in colonnes_necessaires if colonne not in donnees.columns
    ]
    if colonnes_absentes:
        raise ValueError(f"Colonnes absentes pour les features : {colonnes_absentes}")

    for colonne in colonnes_necessaires:
        donnees[colonne] = pd.to_numeric(donnees[colonne], errors="coerce")

    donnees["TotalSF"] = (
        donnees["TotalBsmtSF"].fillna(0)
        + donnees["1stFlrSF"].fillna(0)
        + donnees["2ndFlrSF"].fillna(0)
    )
    donnees["TotalBathrooms"] = (
        donnees["FullBath"].fillna(0)
        + 0.5 * donnees["HalfBath"].fillna(0)
        + donnees["BsmtFullBath"].fillna(0)
        + 0.5 * donnees["BsmtHalfBath"].fillna(0)
    )
    donnees["HouseAge"] = annee_reference - donnees["YearBuilt"]
    donnees["RemodAge"] = annee_reference - donnees["YearRemodAdd"]

    return donnees


def charger_donnees(chemin_csv) :
    donnees = pd.read_csv(chemin_csv)
    if "SalePrice" not in donnees.columns:
        raise ValueError("La colonne cible SalePrice est absente du fichier CSV.")

    donnees = ajouter_features(donnees)
    donnees = donnees.drop(columns=["SalePrice"])
    colonnes_absentes = [
        colonne for colonne in FEATURE_COLUMNS if colonne not in donnees.columns
    ]
    if colonnes_absentes:
        raise ValueError(f"Colonnes de modèle absentes : {colonnes_absentes}")

    X = donnees[FEATURE_COLUMNS].copy()
    y = pd.read_csv(chemin_csv)["SalePrice"]
    return X, y


def creer_preprocesseur(X) :
    """Crée le preprocessing qui sera appris uniquement dans chaque pipeline."""
    colonnes_numeriques = X.select_dtypes(include="number").columns.tolist()
    colonnes_categorielles = X.select_dtypes(exclude="number").columns.tolist()

    pipeline_numerique = Pipeline(
        steps=[
            ("imputation", SimpleImputer(strategy="median")),
            ("standardisation", StandardScaler()),
        ]
    )
    pipeline_categorielle = Pipeline(
        steps=[
            ("imputation", SimpleImputer(strategy="most_frequent")),
            (
                "encodage",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerique", pipeline_numerique, colonnes_numeriques),
            ("categorielle", pipeline_categorielle, colonnes_categorielles),
        ]
    )


if __name__ == "__main__":
    chemin_csv = Path(__file__).resolve().parents[1] / "House_Prices.csv"
    X, y = charger_donnees(chemin_csv)
    preprocesseur = creer_preprocesseur(X)
    print(f"Dimensions de X : {X.shape}")
    print(f"Nombre de prix : {len(y)}")
    print(f"Variables numériques : {len(X.select_dtypes(include='number').columns)}")
    print(f"Variables catégorielles : {len(X.select_dtypes(exclude='number').columns)}")
    print(f"Preprocessing créé : {type(preprocesseur).__name__}")
