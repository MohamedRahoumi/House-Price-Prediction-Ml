# Prédiction du prix des logements

## Contexte

Ce projet étudiant estime le prix de vente d'un logement à partir de caractéristiques simples. Il utilise le dataset House Prices - Advanced Regression Techniques.

La variable cible est `SalePrice`.

## Objectif

Le projet permet de :

- préparer les données et traiter les valeurs manquantes ;
- créer des variables utiles ;
- comparer trois modèles de régression ;
- mesurer leurs performances ;
- sauvegarder un pipeline complet ;
- réaliser une prédiction avec Streamlit.

## Préparation des données

Le fichier contient 1 460 lignes et 81 colonnes. `SalePrice` est séparée de `X` et `y`.

Le modèle utilise 19 variables compréhensibles : qualité, état, années, surfaces, garage, terrain et quelques variables catégorielles comme le quartier.

Les valeurs numériques sont traitées par :

- `SimpleImputer(strategy="median")` ;
- `StandardScaler`.

Les variables catégorielles sont traitées par :

- `SimpleImputer(strategy="most_frequent")` ;
- `OneHotEncoder(handle_unknown="ignore")`.

Le preprocessing est dans un `ColumnTransformer`, lui-même dans un `Pipeline`. Il est donc appris uniquement sur les données d'entraînement. Cela évite le Data Leakage.

## Feature Engineering

Les variables suivantes sont créées dans `scripts/01_preparation.py` :

- `TotalSF` : `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` ;
- `TotalBathrooms` : salles de bain complètes et demi-salles de bain, avec un poids de 0,5 pour les demi-salles ;
- `HouseAge` : `2010 - YearBuilt` ;
- `RemodAge` : `2010 - YearRemodAdd`.

Les valeurs manquantes sont remplacées par zéro uniquement pour ces calculs simples. Les autres imputations sont réalisées dans le pipeline.

## EDA

Le script `scripts/03_eda.py` crée six graphiques dans `outputs/figures/` :

1. distribution de `SalePrice` ;
2. surface totale et prix ;
3. qualité générale et prix ;
4. prix selon le quartier ;
5. année de construction et prix ;
6. matrice de corrélation.

Les observations imprimées pendant l'exécution indiquent notamment que la surface, la qualité et le quartier peuvent aider à expliquer les différences de prix.

## Modèles

Trois modèles sont comparés :

- `LinearRegression` : modèle simple et facilement interprétable ;
- `RandomForestRegressor` : modèle basé sur plusieurs arbres ;
- `GradientBoostingRegressor` : modèle non linéaire construit progressivement.

### Résultats sur le test set

Résultats obtenus après exécution réelle de `scripts/02_modeling.py` :

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| GradientBoosting | 16494.15 | 25723.83 | 0.9137 |
| RandomForest | 17196.85 | 28261.92 | 0.8959 |
| LinearRegression | 18830.32 | 32235.88 | 0.8645 |

Le modèle final retenu est donc `GradientBoosting`, car il obtient le RMSE le plus faible et le R² le plus élevé sur le test set parmi les trois modèles comparés.

## Validation croisée

Une validation `KFold` à 5 folds est appliquée sur les données d'entraînement avec :

```python
KFold(n_splits=5, shuffle=True, random_state=42)
```

Chaque fold utilise une partie des données pour entraîner le pipeline et une autre partie pour l'évaluer. `shuffle=True` mélange les lignes et `random_state=42` rend le résultat reproductible.

### Résultats moyens de validation croisée

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| GradientBoosting | 16595.59 | 28950.06 | 0.8533 |
| RandomForest | 17843.23 | 29878.84 | 0.8445 |
| LinearRegression | 19667.80 | 33508.40 | 0.8088 |

## GridSearchCV

`GridSearchCV` est utilisé sur le pipeline Random Forest. Les paramètres testés contrôlent le nombre d'arbres, la profondeur et les conditions minimales de séparation des arbres.

Meilleurs paramètres obtenus :

```text
n_estimators=200
max_depth=20
min_samples_split=2
min_samples_leaf=1
```

Le score RMSE moyen de validation du Random Forest optimisé est de `29457.42`. Sur le test set, il obtient un RMSE de `28173.26` et un R² de `0.8965`.

## Évaluation et interprétation

Le script `scripts/04_evaluation.py` crée :

- prix réels contre prix prédits ;
- distribution des erreurs ;
- analyse des résidus ;
- comparaison des modèles ;
- graphique des variables importantes.

Les prédictions et les dix plus grandes erreurs sont enregistrées dans `outputs/predictions/`.

Les coefficients de la régression linéaire et les importances des modèles basés sur les arbres sont enregistrés dans `outputs/metrics/`. Les noms sont récupérés après le `ColumnTransformer` avec `get_feature_names_out()`.

## Modèle final

Le pipeline complet est sauvegardé ici :

```text
models/final_model.joblib
```

Il contient le preprocessing et le modèle `GradientBoostingRegressor`. Streamlit peut donc recevoir les données brutes sans refaire le preprocessing ni réentraîner le modèle.

## Streamlit

L'application se trouve dans `dashboard/app.py`.

Installation :

```bash
pip install -r requirements.txt
```

Lancer l'application :

```bash
streamlit run dashboard/app.py
```

Le formulaire utilise `number_input` pour les nombres et `selectbox` pour les catégories. Le modèle est chargé avec `@st.cache_resource`.

## Tests

Lancer les tests :

```bash
pytest
```

Les tests vérifient le chargement des données, la présence de `SalePrice`, la création des quatre features et le fonctionnement d'une prédiction.

## Docker

Construire l'image :

```bash
docker build -t house-prices-app .
```

Lancer le conteneur :

```bash
docker run -p 8501:8501 house-prices-app
```

Puis ouvrir `http://localhost:8501`.

## Limites

- Le dataset concerne un marché immobilier précis et ses prix ne sont pas universels.
- Les résultats dépendent du découpage train/test utilisé.
- Les grandes erreurs ne sont pas automatiquement expliquées par une cause certaine.
- Certaines variables du dataset original ne sont pas utilisées pour garder un projet compréhensible.

## Améliorations possibles

- Tester d'autres modèles simples ;
- ajouter une validation sur un nouveau dataset ;
- améliorer l'interface Streamlit ;
- analyser plus précisément les logements présentant les erreurs les plus importantes.
