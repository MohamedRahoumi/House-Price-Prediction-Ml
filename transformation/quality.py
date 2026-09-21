
import pandas as pd


path = "data_sauvage/House_Prices.csv"

df = pd.read_csv(path)
print("Nombre de lignes :", df.shape[0])
print("Nombre de colonnes :", df.shape[1])


print("\n valeurs manquants :")

missing = df.isnull().sum()

missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    print("Aucune valeur manquante.")
else:
    print(missing)

    print("\n les valeurs manquantes :")

    missing_percentage = (
        df.isnull().sum()
        
    )

    missing_percentage = (
        missing_percentage[missing_percentage > 0]
        .sort_values(ascending=False)
    )

    print(missing_percentage)

print("\n les doublants")

duplicates = df.duplicated().sum()

print("Nombre de doublons :", duplicates)

