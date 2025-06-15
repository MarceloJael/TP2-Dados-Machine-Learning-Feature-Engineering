import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer

iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

print("Dados originais (5 primeiras linhas):")
print(df_iris.head())

normalizador_l1 = Normalizer(norm='l1')
dados_l1 = normalizador_l1.transform(df_iris)

df_l1 = pd.DataFrame(dados_l1, columns=[f"{col}_l1" for col in df_iris.columns])

print("\nDados após normalização L1:")
print(df_l1.head())

somas_l1 = np.sum(dados_l1, axis=1)
print("\nSoma das features (norma L1) das primeiras 5 amostras (deve ser 1):")
print(somas_l1[:5])