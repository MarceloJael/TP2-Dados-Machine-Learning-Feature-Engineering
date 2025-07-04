from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

texto = """
A alma é, pois, imortal; renasceu repetidas vezes na existência e contemplou todas as coisas existentes e por isso não há nada que ela não conheça! 
Não é de espantar que ela seja capaz de evocar à memória a lembrança de objetos que viu anteriormente, 
e que se relacionam tanto com a virtude como com as outras coisas existentes. 
Toda a natureza, com efeito, é uma só, é um todo orgânico, 
e o espírito já viu todas as coisas; logo, nada impede que ao nos lembrarmos de uma coisa – 
o que nós, homens, chamamos de “saber” – todas as outras coisas acorram imediata e maquinalmente à nossa consciência.
"""

vectorizer = CountVectorizer()

X = vectorizer.fit_transform([texto])

df_bow = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print("Vetorização Bag-of-Words:")
print(df_bow)