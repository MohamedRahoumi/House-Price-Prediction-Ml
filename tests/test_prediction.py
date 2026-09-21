import importlib
from pathlib import Path

import joblib
import pandas as pd

preparation = importlib.import_module("scripts.01_preparation")

RACINE = Path(__file__).resolve().parents[1]


def test_chargement_donnees_et_cible():
    donnees = pd.read_csv(RACINE / "House_Prices.csv")
    assert len(donnees) > 0
    assert "SalePrice" in donnees.columns


def test_creation_des_features():
    donnees = pd.read_csv(RACINE / "House_Prices.csv").head(1)
    resultat = preparation.ajouter_features(donnees)
    for colonne in ["TotalSF", "TotalBathrooms", "HouseAge", "RemodAge"]:
        assert colonne in resultat.columns


def test_chargement_du_modele_et_prediction():
    modele = joblib.load(RACINE / "models" / "final_model.joblib")
    X, _ = preparation.charger_donnees(RACINE / "House_Prices.csv")
    prediction = modele.predict(X.head(1))
    assert len(prediction) == 1
    assert prediction[0] > 0
