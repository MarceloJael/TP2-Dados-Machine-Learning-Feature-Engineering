from sklearn.feature_extraction.text import CountVectorizer

texto = """
A alma é, pois, imortal; renasceu repetidas vezes na existência e contemplou todas as coisas existentes e por isso não há nada que ela não conheça! 
Não é de espantar que ela seja capaz de evocar à memória a lembrança de objetos que viu anteriormente, 
e que se relacionam tanto com a virtude como com as outras coisas existentes. 
Toda a natureza, com efeito, é uma só, é um todo orgânico, 
e o espírito já viu todas as coisas; logo, nada impede que ao nos lembrarmos de uma coisa – 
o que nós, homens, chamamos de “saber” – todas as outras coisas acorram imediata e maquinalmente à nossa consciência.
"""

vectorizer_uni = CountVectorizer(ngram_range=(1,1))
unigrams = vectorizer_uni.fit_transform([texto])
unigram_count = len(vectorizer_uni.get_feature_names_out())

vectorizer_bi = CountVectorizer(ngram_range=(2,2))
bigrams = vectorizer_bi.fit_transform([texto])
bigram_count = len(vectorizer_bi.get_feature_names_out())

vectorizer_tri = CountVectorizer(ngram_range=(3,3))
trigrams = vectorizer_tri.fit_transform([texto])
trigram_count = len(vectorizer_tri.get_feature_names_out())

print(f"Número de unigrams: {unigram_count}")
print(f"Número de bigrams: {bigram_count}")
print(f"Número de trigrams: {trigram_count}")

print("\nExemplos:")
print("Unigrams:", vectorizer_uni.get_feature_names_out()[:10])
print("Bigrams:", vectorizer_bi.get_feature_names_out()[:10])
print("Trigrams:", vectorizer_tri.get_feature_names_out()[:10])