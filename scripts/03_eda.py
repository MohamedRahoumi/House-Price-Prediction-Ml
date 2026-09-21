"""Exploration visuelle simple du dataset House Prices."""

import importlib
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
preparation = importlib.import_module("scripts.01_preparation")


def creer_graphiques(chemin_csv, dossier_sortie) :
    donnees = pd.read_csv(chemin_csv)
    donnees = preparation.ajouter_features(donnees)
    dossier_sortie = Path(dossier_sortie)
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.histplot(donnees["SalePrice"], kde=True)
    plt.title("Distribution du prix de vente")
    plt.xlabel("SalePrice")
    plt.ylabel("Nombre de logements")
    plt.tight_layout()
    plt.savefig(dossier_sortie / "01_distribution_saleprice.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=donnees, x="TotalSF", y="SalePrice", alpha=0.6)
    plt.title("Surface totale et prix de vente")
    plt.xlabel("Surface totale (pieds carrés)")
    plt.ylabel("Prix de vente")
    plt.tight_layout()
    plt.savefig(dossier_sortie / "02_surface_prix.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=donnees, x="OverallQual", y="SalePrice")
    plt.title("Prix selon la qualité générale")
    plt.xlabel("Qualité générale")
    plt.ylabel("Prix de vente")
    plt.tight_layout()
    plt.savefig(dossier_sortie / "03_qualite_prix.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    ordre_quartiers = donnees.groupby("Neighborhood")["SalePrice"].median().sort_values().index
    sns.boxplot(data=donnees, x="Neighborhood", y="SalePrice", order=ordre_quartiers)
    plt.title("Prix selon le quartier")
    plt.xlabel("Quartier")
    plt.ylabel("Prix de vente")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig(dossier_sortie / "04_quartier_prix.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=donnees, x="YearBuilt", y="SalePrice", alpha=0.6)
    plt.title("Année de construction et prix")
    plt.xlabel("Année de construction")
    plt.ylabel("Prix de vente")
    plt.tight_layout()
    plt.savefig(dossier_sortie / "05_annee_prix.png")
    plt.close()

    variables_numeriques = donnees.select_dtypes(include="number")
    correlation = variables_numeriques.corr(numeric_only=True)
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation, cmap="coolwarm", center=0, xticklabels=False, yticklabels=False)
    plt.title("Matrice de corrélation des variables numériques")
    plt.tight_layout()
    plt.savefig(dossier_sortie / "06_matrice_correlation.png")
    plt.close()


if __name__ == "__main__":
    racine = Path(__file__).resolve().parents[1]
    creer_graphiques(racine / "House_Prices.csv", racine / "outputs" / "figures")
