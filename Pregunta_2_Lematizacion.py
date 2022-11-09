# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:01:23 2020

@author: franc
"""

# Importamos principales librerias que utilizaremos para el desarollo de la pregunta número 2.

import nltk; nltk.download('stopwords')
import os
import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import pyLDAvis
import pyLDAvis.gensim  
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
from  sklearn.feature_extraction.text import CountVectorizer


##############################################################################################
################################## LDA + LEMATIZACIÓN ########################################
##############################################################################################

# Función que crea corpus a partir de las noticias que extrajimos de diferentes medios.

def CrearCorpus(path):
  dirs = os.listdir(path)
  corpus=[]
  for fn  in dirs:
     FileName = PATH + fn
     textdata = open(FileName,'r').read()
     corpus = corpus + [textdata]
  return(corpus)


PATH =  "C:/Users/franc/Desktop/Archivos/Noticias_Tarea/"  

docs = CrearCorpus(PATH)


# Con estos comandos podemos ver cuales son las palabras más repetidas en en corpus donde principalmente  se ve 
# que se trata de stopwords o palabras que no aportarán mucho significado a nuestro análisis de tópico.

press_vect = CountVectorizer()

docs1 = pd.DataFrame(docs)
docs1.columns = ['Corpus_noticia']

press_vect.fit(docs1.Corpus_noticia)

# Transformamos el corpus en una representación matricial
X_press = press_vect.transform(docs1.Corpus_noticia)

# Creamos una tabla para visualizar la representación matricial
X_df=pd.DataFrame(X_press.toarray(), columns=press_vect.get_feature_names())
# Representamos la tabla
print(X_df.head())

# A continuación hacemos una consulta de las 20 palabras más frentes del corpus
df_s=X_df.sum(numeric_only=True)
df_2=df_s.sort_values(ascending=False)
print(df_2[:20:])

# Luego de las 20 palabras menos frecuentes del corpus
print(df_2.iloc[-20:])

# Con las librerias nltk extraemos las stopwords tradicionales del español. Por otro lado, a través de trigramas y bigramas del punto anterior,
# detectamos algunas inconsistencias en las palabras que podía causar ruido así que también las eliminaremos.
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import word_tokenize
nlp = spacy.load("es_core_news_sm")

custom_stopwords = [
                    'coronavirus','covid','covid-19','ProChile.También','ChileB2B','transfronterizo.Desde',
                    'ORyan.Además','digitalesAunque','directivo.Medidas','noticiarelacionadael','noticiarelacionadacrisis',
                    'noticiarelacionadacómo','noticiarelacionadadiputado','noticiarelacionadadenuncian','noticiarelacionadaefectos',
                    'noticiarelacionadacanciller','noticiarelacionadacoronavirus','noticiarelacionadaminsal','noticiarelacionadala',
                    'noticiarelacionadagobierno','noticiarelacionadabono','noticiarelacionadabancos','noticiarelacionadaaduanas',
                    'noticiarelacionada','noticiarelacionadaproyecto','noticiarelacionadaops','unestudio','notacoronaviruscoronavirus',
                    'aa','noticiarelacionadaricotti','\ufeff1','aair new'
                      ] 

nltk_stopwords = set(stopwords.words('spanish'))
spacy_stopwords = set(nlp.Defaults.stop_words)
spanish_stopwords = list(nltk_stopwords.union(spacy_stopwords))+custom_stopwords

# Vectorizamos y buscamos nuevamente las palabras más frecuentes y menos frecuentes.

press_vect = CountVectorizer(
                            #  max_features=100,
                             ngram_range=(2, 2),
                             max_df=0.85,
                             stop_words=spanish_stopwords,
                             token_pattern=r'\b[^\d\W][^\d\W]+\b'
                             )

# Aplicamos la transformación al coprus
press_vect.fit(docs1.Corpus_noticia)

X_press = press_vect.transform(docs1.Corpus_noticia)
X_df=pd.DataFrame(X_press.toarray(), columns=press_vect.get_feature_names())
print(X_df.head())


df_s=X_df.sum(numeric_only=True)
df_2=df_s.sort_values(ascending=False)
print(df_2[:20:])

print(df_2.iloc[-20:])

docs2 = docs1.Corpus_noticia.values.tolist()

docs2


# Se generan funciones que eliminar puntuación, tokenizan, elimina stopwords y hace la lematización.

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(docs2))

print(data_words[:1])

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in spanish_stopwords] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

data_words = remove_stopwords(data_words)

# Librería Spacy para lematizar.
nlp = spacy.load('es_core_news_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB','ADV'])


# CREAR DICCIONARIO Y CORPUS 

# Crea Diccionario
id2word = corpora.Dictionary(data_lemmatized)
len(id2word)
# Crea Corpus
texts = data_lemmatized
len(texts)
# Frecuencia de documentos BOW
corpus = [id2word.doc2bow(text) for text in texts]
len(corpus)
# imprime
print(corpus[:1])

# Crea Diccionario
dictionary = corpora.Dictionary(texts) 

# Función iterativa que determina la coherencia del modelo entre 1 y 40 para elegir cuántos tópico nos generan una mejor coherencia.
def compute_coherence_values(dictionary, corpus, texts, limit, start=1, step=1):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics,random_state=100,update_every=1,chunksize=100,passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# Muestra resultados de coherencia a partir de los parámetros anteriores.
model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=texts, start=1, limit=41, step=1)

# Muestra gráficos de coherencia

limit=41; start=1; step=1;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# Imprime coherencia
for i in range(0,40,1):
    print('El número de tópicos es:',[i+1],'con coherence de:',coherence_values[i])

# Modelo que contiene 13 tópicos pues fueron los que mejor coherencia nos dieron para el modelo en términos de eficiencia computacional y valor.
elegido = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=13, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# Imprime los 13 tópicos con sus palabras.
pprint(elegido.print_topics())


# Determina el perplexity del modelo
print('\nPerplexity: ', elegido.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

#  Determina la coherencia del modelo
coherence_model_lda = CoherenceModel(model=elegido, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Genere Mapa de tópicos 
pyLDAvis.enable_notebook()
#vis = pyLDAvis.gensim.prepare(elegido, corpus, id2word, mds='tsne')
vis = pyLDAvis.gensim.prepare(elegido, corpus, id2word, mds='tsne')
pyLDAvis.show(vis)

