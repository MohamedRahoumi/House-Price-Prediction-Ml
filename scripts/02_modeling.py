"""Entraînement, validation et sauvegarde du modèle final."""

import importlib
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
preparation = importlib.import_module("scripts.01_preparation")


RANDOM_STATE = 42


def creer_pipelines(X) :
    modeles = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    pipelines = {}
    for nom, modele in modeles.items():
        pipelines[nom] = Pipeline(
            steps=[
                ("preprocessing", preparation.creer_preprocesseur(X)),
                ("model", modele),
            ]
        )
    return pipelines


def calculer_metriques(y_reel: pd.Series, y_predit) :
    return {
        "MAE": mean_absolute_error(y_reel, y_predit),
        "RMSE": mean_squared_error(y_reel, y_predit) ** 0.5,
        "R2": r2_score(y_reel, y_predit),
    }


def valider_pipelines(pipelines, X_train, y_train):
    """Compare les modèles avec les mêmes cinq folds."""
    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    resultats = []

    for nom, pipeline in pipelines.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=folds,
            scoring={
                "MAE": "neg_mean_absolute_error",
                "RMSE": "neg_root_mean_squared_error",
                "R2": "r2",
            },
        )
        resultats.append(
            {
                "Model": nom,
                "MAE": -scores["test_MAE"].mean(),
                "RMSE": -scores["test_RMSE"].mean(),
                "R2": scores["test_R2"].mean(),
            }
        )

    return pd.DataFrame(resultats).sort_values("RMSE")


def optimiser_random_forest(X_train, y_train) :
    pipeline = creer_pipelines(X_train)["RandomForest"]
    grille = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }
    recherche = GridSearchCV(
        pipeline,
        param_grid=grille,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    recherche.fit(X_train, y_train)
    return recherche


def obtenir_importances(pipeline) :
    preprocesseur = pipeline.named_steps["preprocessing"]
    modele = pipeline.named_steps["model"]
    noms = preprocesseur.get_feature_names_out()
    if hasattr(modele, "feature_importances_"):
        importances = modele.feature_importances_
    else:
        importances = modele.coef_
    importances = importances.ravel()
    return pd.DataFrame({"Feature": noms, "Importance": importances}).sort_values(
        "Importance", key=lambda valeurs: valeurs.abs(), ascending=False
    )


def main() :
    dossier_projet = Path(__file__).resolve().parents[1]
    chemin_csv = dossier_projet / "House_Prices.csv"
    dossier_modeles = dossier_projet / "models"
    dossier_metrics = dossier_projet / "outputs" / "metrics"
    dossier_modeles.mkdir(exist_ok=True)
    dossier_metrics.mkdir(parents=True, exist_ok=True)

    X, y = preparation.charger_donnees(chemin_csv)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipelines = creer_pipelines(X_train)
    validation = valider_pipelines(pipelines, X_train, y_train)
    print("Validation croisée :")
    print(validation.to_string(index=False))

    resultats_test = []
    modeles_entraines = {}
    for nom, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metriques = calculer_metriques(y_test, predictions)
        resultats_test.append({"Model": nom, **metriques})
        modeles_entraines[nom] = pipeline

    resultats_test = pd.DataFrame(resultats_test).sort_values("RMSE")
    resultats_test.to_csv(dossier_metrics / "model_comparison.csv", index=False)
    for nom, modele in modeles_entraines.items():
        if hasattr(modele.named_steps["model"], "feature_importances_") or hasattr(
            modele.named_steps["model"], "coef_"
        ):
            obtenir_importances(modele).to_csv(
                dossier_metrics / f"feature_importance_{nom}.csv", index=False
            )
    print("\nRésultats sur le test set :")
    print(resultats_test.to_string(index=False))

    recherche = optimiser_random_forest(X_train, y_train)
    modele_optimise = recherche.best_estimator_
    predictions_optimisees = modele_optimise.predict(X_test)
    metriques_optimisees = calculer_metriques(y_test, predictions_optimisees)
    print("\nMeilleurs paramètres Random Forest :")
    print(recherche.best_params_)
    print(f"Meilleur score CV RMSE : {-recherche.best_score_:.2f}")
    print("Résultats Random Forest optimisé :", metriques_optimisees)

    modeles_entraines["RandomForestOptimise"] = modele_optimise
    resultats_optimises = pd.DataFrame(
        [{"Model": "RandomForestOptimise", **metriques_optimisees}]
    )
    resultats_optimises.to_csv(
        dossier_metrics / "optimized_random_forest.csv", index=False
    )

    meilleur_nom = resultats_test.iloc[0]["Model"]
    meilleur_modele = modeles_entraines[meilleur_nom]
    joblib.dump(meilleur_modele, dossier_modeles / "final_model.joblib")
    print(f"\nModèle final sauvegardé : {meilleur_nom}")

    obtenir_importances(meilleur_modele).to_csv(
        dossier_metrics / "feature_importance_final.csv", index=False
    )


if __name__ == "__main__":
    main()
