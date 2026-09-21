"""Application Streamlit de prédiction du prix d'un logement."""

import importlib
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
preparation = importlib.import_module("scripts.01_preparation")

CHEMIN_MODELE = RACINE / "models" / "final_model.joblib"
CHEMIN_DONNEES = RACINE / "House_Prices.csv"


@st.cache_resource
def charger_modele():
    """Charge le pipeline déjà entraîné une seule fois."""
    import joblib

    return joblib.load(CHEMIN_MODELE)


def charger_categories(colonne: str) -> list[str]:
    donnees = pd.read_csv(CHEMIN_DONNEES)
    return sorted(donnees[colonne].dropna().astype(str).unique().tolist())


st.set_page_config(page_title="Prix des logements", page_icon="🏠")
st.title("Prédiction du prix d'un logement")
st.write("Renseignez les caractéristiques du logement puis cliquez sur Prédire.")

try:
    with st.form("formulaire_prediction"):
        st.subheader("Caractéristiques principales")
        overall_qual = st.number_input("Qualité générale (1 à 10)", 1, 10, 5)
        overall_cond = st.number_input("État général (1 à 10)", 1, 10, 5)
        year_built = st.number_input("Année de construction", 1800, 2026, 2000)
        year_remod_add = st.number_input("Année de rénovation", 1800, 2026, 2000)
        gr_liv_area = st.number_input("Surface habitable", 0, 10000, 1500)
        total_bsmt_sf = st.number_input("Surface de sous-sol", 0, 5000, 800)
        first_flr_sf = st.number_input("Surface du premier étage", 0, 5000, 800)
        second_flr_sf = st.number_input("Surface du deuxième étage", 0, 5000, 400)
        lot_area = st.number_input("Surface du terrain", 0, 100000, 8000)
        bedroom_abv_gr = st.number_input("Nombre de chambres", 0, 20, 3)

        st.subheader("Équipements")
        full_bath = st.number_input("Salles de bain complètes", 0, 10, 2)
        half_bath = st.number_input("Demi-salles de bain", 0, 10, 1)
        bsmt_full_bath = st.number_input("Salles de bain complètes au sous-sol", 0, 5, 0)
        bsmt_half_bath = st.number_input("Demi-salles de bain au sous-sol", 0, 5, 0)
        garage_cars = st.number_input("Places de garage", 0, 10, 2)
        garage_area = st.number_input("Surface du garage", 0, 2000, 400)

        st.subheader("Catégories")
        neighborhood = st.selectbox("Quartier", charger_categories("Neighborhood"))
        exter_qual = st.selectbox("Qualité extérieure", charger_categories("ExterQual"))
        kitchen_qual = st.selectbox("Qualité de la cuisine", charger_categories("KitchenQual"))
        ms_zoning = st.selectbox("Zone immobilière", charger_categories("MSZoning"))

        bouton_prediction = st.form_submit_button("Prédire")

    if bouton_prediction:
        logement = pd.DataFrame(
            [
                {
                    "OverallQual": overall_qual,
                    "OverallCond": overall_cond,
                    "YearBuilt": year_built,
                    "YearRemodAdd": year_remod_add,
                    "GrLivArea": gr_liv_area,
                    "TotalBsmtSF": total_bsmt_sf,
                    "1stFlrSF": first_flr_sf,
                    "2ndFlrSF": second_flr_sf,
                    "FullBath": full_bath,
                    "HalfBath": half_bath,
                    "BsmtFullBath": bsmt_full_bath,
                    "BsmtHalfBath": bsmt_half_bath,
                    "GarageCars": garage_cars,
                    "GarageArea": garage_area,
                    "LotArea": lot_area,
                    "BedroomAbvGr": bedroom_abv_gr,
                    "Neighborhood": neighborhood,
                    "ExterQual": exter_qual,
                    "KitchenQual": kitchen_qual,
                    "MSZoning": ms_zoning,
                }
            ]
        )
        logement = preparation.ajouter_features(logement)
        logement = logement[preparation.FEATURE_COLUMNS]
        prix_estime = charger_modele().predict(logement)[0]
        st.success(f"Prix estimé : {prix_estime:,.0f} $".replace(",", " "))
except FileNotFoundError:
    st.error("Le modèle final est introuvable. Lancez d'abord scripts/02_modeling.py.")
except Exception as erreur:
    st.error(f"La prédiction n'a pas pu être réalisée : {erreur}")
