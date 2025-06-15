import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

modelo = LogisticRegression(max_iter=10000)

rfe = RFE(estimator=modelo, n_features_to_select=10)
rfe.fit(X, y)

features_rfe = X.columns[rfe.support_]

print("Features selecionadas com Wrapper:")
print(features_rfe.tolist())

from sklearn.feature_selection import SelectKBest, f_classif

filtro = SelectKBest(score_func=f_classif, k=10)
filtro.fit(X, y)
features_filtro = X.columns[filtro.get_support()]

print("\nComparação com seleção por filtro (Questão 5):")
print("Interseção:", set(features_rfe).intersection(set(features_filtro)))
print("Só no wrapper:", set(features_rfe) - set(features_filtro))
print("Só no filtro:", set(features_filtro) - set(features_rfe))