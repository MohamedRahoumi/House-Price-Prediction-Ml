# Prédiction des prix immobiliers

Projet d'analyse et de prédiction des prix de maisons à partir de caractéristiques immobilières.

## Méthodologie

Le flux comprend l'exploration des données, le contrôle qualité, l'ingénierie des variables et la préparation des matrices d'apprentissage.

## Modèles

Trois approches sont comparées : Ridge, Random Forest et Gradient Boosting. Les prédictions et les résultats de comparaison sont conservés dans `outputs/`.

## Exécution

Installer les dépendances puis lancer les scripts depuis la racine du projet :

```bash
python scripts/exploration_preparation.py
python scripts/Feature_Engineering.py
python scripts/entrainement_modeles.py
```

## Résultats

Les fichiers `resultats_etape4.csv` et `predictions_etape4.csv` regroupent les performances et les prédictions finales. Les graphiques exploratoires se trouvent dans `outputs/graphiques_etape2/`.

## Organisation

- `data_sauvage/` contient les données sources.
- `scripts/` contient les étapes d'analyse et d'entraînement.
- `transformation/` regroupe les contrôles et le prétraitement.
- `outputs/` contient les données dérivées, résultats et graphiques.