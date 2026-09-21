import pandas as pd

df=pd.read_csv("data_sauvage/House_Prices.csv")

print(df['OverallQual'].to_string())