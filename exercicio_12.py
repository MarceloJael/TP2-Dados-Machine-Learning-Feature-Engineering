import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

texto = """
A alma é, pois, imortal; renasceu repetidas vezes na existência e contemplou todas as coisas existentes e por isso não há nada que ela não conheça! 
Não é de espantar que ela seja capaz de evocar à memória a lembrança de objetos que viu anteriormente, 
e que se relacionam tanto com a virtude como com as outras coisas existentes. 
Toda a natureza, com efeito, é uma só, é um todo orgânico, 
e o espírito já viu todas as coisas; logo, nada impede que ao nos lembrarmos de uma coisa – 
o que nós, homens, chamamos de “saber” – todas as outras coisas acorram imediata e maquinalmente à nossa consciência.
"""

tokens = word_tokenize(texto, language='portuguese')
stop_words = set(stopwords.words('portuguese'))
tokens_filtrados = [palavra.lower() for palavra in tokens if palavra.lower() not in stop_words and palavra not in string.punctuation]

# Simulação com algumas palavras em inglês
exemplo = "men running sings going better knowing objects nature remembered memory"
tokens_exemplo = word_tokenize(exemplo)

lemmatizer = WordNetLemmatizer()
tokens_lemmatizados = [lemmatizer.lemmatize(palavra) for palavra in tokens_exemplo]

print("Texto (simulado) após lemmatization:")
print(' '.join(tokens_lemmatizados))
