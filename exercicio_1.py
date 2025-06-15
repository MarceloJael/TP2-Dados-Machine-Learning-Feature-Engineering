import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

print("Dados originais:")
print(df_iris.head())

scaler = StandardScaler()
dados_escalonados = scaler.fit_transform(df_iris)

df_escalonado = pd.DataFrame(dados_escalonados, columns=[f"{col}_zscore" for col in df_iris.columns])

medias = df_escalonado.mean()
desvios = df_escalonado.std()

print("\nDados após normalização:")
print(df_escalonado.head())

print("\nMédia das features normalizadas:")
print(medias)

print("\nDesvio padrão das features normalizadas:")
print(desvios)