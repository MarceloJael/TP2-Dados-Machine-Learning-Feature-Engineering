import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

print("Dataset Breast Cancer - Shape:", X.shape)

filtro = SelectKBest(score_func=f_classif, k=10)
X_filtrado = filtro.fit_transform(X, y)

scores = filtro.scores_
features_escolhidas = X.columns[filtro.get_support()]

df_scores = pd.DataFrame({
    'feature': X.columns,
    'f_score': scores
}).sort_values(by='f_score', ascending=False)

print("\nTop 10 features mais significativas:")
print(df_scores.head(10))

plt.figure(figsize=(10, 5))
plt.barh(df_scores.head(10)['feature'], df_scores.head(10)['f_score'])
plt.xlabel("F-score")
plt.title("Top 10 Features - Seleção por Filtro")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()