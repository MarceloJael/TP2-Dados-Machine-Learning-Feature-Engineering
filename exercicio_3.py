import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer

iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

print("Dados originais (5 primeiras linhas):")
print(df_iris.head())

normalizador_l2 = Normalizer(norm='l2')
dados_l2 = normalizador_l2.transform(df_iris)

df_l2 = pd.DataFrame(dados_l2, columns=[f"{col}_l2" for col in df_iris.columns])

print("\nDados após regularização L2:")
print(df_l2.head())

normas = np.linalg.norm(dados_l2, axis=1)
print("\nNorma L2 das primeiras 5 amostras:")
print(normas[:5])
