import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

def zscore_manual(df):
    return (df - df.mean()) / df.std()

df_escalonado_manual = zscore_manual(df_iris)
print("Dados escalonados manualmente:")
print(df_escalonado_manual.head())

scaler = StandardScaler()
df_escalonado_sklearn = pd.DataFrame(scaler.fit_transform(df_iris), columns=iris.feature_names)

diferenca = abs(df_escalonado_manual - df_escalonado_sklearn).mean()
print("\nDiferença média entre implementação manual e sklearn:")
print(diferenca)