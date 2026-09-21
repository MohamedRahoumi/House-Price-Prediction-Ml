"""Évaluation détaillée du pipeline final sur le test set."""

import importlib
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
preparation = importlib.import_module("scripts.01_preparation")


RANDOM_STATE = 42


def main() :
    racine = Path(__file__).resolve().parents[1]
    X, y = preparation.charger_donnees(racine / "House_Prices.csv")
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    modele = joblib.load(racine / "models" / "final_model.joblib")
    predictions = modele.predict(X_test)
    erreurs = y_test - predictions

    sorties = racine / "outputs"
    figures = sorties / "figures"
    predictions_dir = sorties / "predictions"
    figures.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    tableau_erreurs = X_test.copy()
    tableau_erreurs["Prix_reel"] = y_test
    tableau_erreurs["Prix_predit"] = predictions
    tableau_erreurs["Erreur_absolue"] = erreurs.abs()
    tableau_erreurs.sort_values("Erreur_absolue", ascending=False).head(10).to_csv(
        predictions_dir / "plus_grandes_erreurs.csv"
    )
    tableau_erreurs.to_csv(predictions_dir / "predictions_test.csv")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.7)
    minimum = min(y_test.min(), predictions.min())
    maximum = max(y_test.max(), predictions.max())
    plt.plot([minimum, maximum], [minimum, maximum], color="red")
    plt.title("Prix réels et prix prédits")
    plt.xlabel("Prix réel")
    plt.ylabel("Prix prédit")
    plt.tight_layout()
    plt.savefig(figures / "07_prix_reel_vs_predit.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(erreurs, kde=True)
    plt.title("Distribution des erreurs de prédiction")
    plt.xlabel("Erreur : prix réel - prix prédit")
    plt.ylabel("Nombre de logements")
    plt.tight_layout()
    plt.savefig(figures / "08_distribution_erreurs.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=predictions, y=erreurs, alpha=0.7)
    plt.axhline(0, color="red")
    plt.title("Analyse des résidus")
    plt.xlabel("Prix prédit")
    plt.ylabel("Résidu")
    plt.tight_layout()
    plt.savefig(figures / "09_analyse_residus.png")
    plt.close()

    comparaison = pd.read_csv(sorties / "metrics" / "model_comparison.csv")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=comparaison, x="Model", y="RMSE")
    plt.title("Comparaison des modèles selon le RMSE")
    plt.xlabel("Modèle")
    plt.ylabel("RMSE")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(figures / "10_comparaison_modeles.png")
    plt.close()

    importances = pd.read_csv(sorties / "metrics" / "feature_importance_final.csv")
    principales = importances.head(20).sort_values("Importance")
    plt.figure(figsize=(9, 7))
    sns.barplot(data=principales, x="Importance", y="Feature")
    plt.title("Principales caractéristiques du modèle final")
    plt.xlabel("Importance")
    plt.ylabel("Variable après preprocessing")
    plt.tight_layout()
    plt.savefig(figures / "11_features_importantes.png")
    plt.close()

    print("Évaluation terminée.")
    print("Les 10 plus grandes erreurs sont dans outputs/predictions/plus_grandes_erreurs.csv")
    print(tableau_erreurs[["Prix_reel", "Prix_predit", "Erreur_absolue"]].head(10).to_string())


if __name__ == "__main__":
    main()
