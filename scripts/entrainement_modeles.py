from pathlib import Path

import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "outputs" / "House_Prices_engineered.csv"
OUTPUTS_DIR = ROOT / "outputs"

y_csv = ROOT / "outputs" / "y_prepared.csv"


X = pd.read_csv(INPUT_PATH)

y = pd.read_csv(y_csv)["SalePrice"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


modeles = {
    "Ridge (lineaire)": Ridge(alpha=10.0),

    "Random Forest (arbres)": RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting (non lineaire)": GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),
}


resultats = []

predictions = pd.DataFrame({
    "SalePrice_reel": y_test.reset_index(drop=True)
})


for nom, modele in modeles.items():

    # Entraînement
    modele.fit(X_train, y_train)

    # Prédictions
    predictions_modele = modele.predict(X_test)

    predictions[nom] = predictions_modele

    # Évaluation
    resultats.append({
        "Modele": nom,
        "MAE": mean_absolute_error(y_test, predictions_modele),
        "RMSE": mean_squared_error(y_test, predictions_modele) ** 0.5,
        "R2": r2_score(y_test, predictions_modele),
    })


# Résultats
resultats_df = pd.DataFrame(resultats).sort_values("RMSE")

resultats_formates = resultats_df.to_string(
    index=False,
    float_format=lambda valeur: f"{valeur:.2f}"
)


# Sauvegarde
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

resultats_df.to_csv(
    OUTPUTS_DIR / "resultats_etape4.csv",
    index=False
)

predictions.to_csv(
    OUTPUTS_DIR / "predictions_etape4.csv",
    index=False
)

print("Entraînement terminé.\n")
print(resultats_formates)

print(
    "\nRésultats sauvegardés dans :",
    OUTPUTS_DIR / "resultats_etape4.csv"
)