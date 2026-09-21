from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


racine = Path(__file__).parent.parent

df = pd.read_csv(racine / "data_sauvage" / "House_Prices.csv")
df = df.drop_duplicates()

dossier = racine / "outputs" / "graphiques_etape2"
dossier.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")


df["Log_SalePrice"] = np.log(df["SalePrice"])

sns.histplot(df["Log_SalePrice"], bins=30)
plt.title("Distribution des prix (Après Log Transform)")
plt.xlabel("Log(Prix)")
plt.ylabel("Nombre de maisons")
plt.savefig(dossier / "01_distribution_prix_log.png")
plt.show()



outliers_index = df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index
df_clean = df.drop(outliers_index)
sns.scatterplot(data=df_clean, x="GrLivArea", y="SalePrice")
plt.title("Surface et prix (Bla les outliers)")
plt.xlabel("Surface habitable")
plt.ylabel("Prix")
plt.savefig(dossier / "02_surface_prix_clean.png")
plt.show()

sns.boxplot(data=df, x="OverallQual", y="SalePrice")
plt.title("Qualite et prix")
plt.xlabel("Qualite generale")
plt.ylabel("Prix")
plt.savefig(dossier / "03_qualite_prix.png")
plt.show()


plt.figure(figsize=(12, 8))

ordre_quartiers = df.groupby('Neighborhood')['SalePrice'].median().sort_values().index

sns.boxplot(
    data=df, 
    x="SalePrice", 
    y="Neighborhood", 
    order=ordre_quartiers, 
    color="b" 
)

plt.title("Prix selon le quartier (Trié par prix médian)")
plt.xlabel("Prix")
plt.ylabel("Quartier")

plt.tight_layout() 
plt.savefig(dossier / "04_quartier_prix_ameliore.png")
plt.show()

sns.scatterplot(data=df, x="YearBuilt", y="SalePrice")
plt.title("Annee de construction et prix")
plt.xlabel("Annee de construction")
plt.ylabel("Prix")
plt.savefig(dossier / "05_annee_prix.png")
plt.show()

colonnes = ["SalePrice", "GrLivArea", "OverallQual", "YearBuilt", "TotalBsmtSF"]
sns.heatmap(df[colonnes].corr(), annot=True, cmap="coolwarm")
plt.title("Matrice de correlation")
plt.savefig(dossier / "06_matrice_correlation.png")
plt.show()




print("Les graphiques sont dans le dossier :", dossier)