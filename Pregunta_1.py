SPACY 
import os
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
from string import punctuation
from collections import Counter
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

for texto in corpus:  
    texto = EliminarStopwords(texto.lower())    
    print(texto)
# *****************************************************************

def CrearCorpus(path):
  dirs = os.listdir(path)
  corpus = []
  doc_ids = []  # lista con los nombres de los documentos
  for fn  in dirs:
     data = open(path+fn,'r',encoding='latin1').read()
     corpus.append(data)
     doc_ids.append(fn)
  return(corpus,doc_ids)

# *****************************************************************
   
def CrearVSM(corpus):
  textos = ProcesarLexico(corpus)
  tfidf = TfidfVectorizer()
  feature_matrix = tfidf.fit_transform(textos)
  tdm = feature_matrix.toarray().transpose()
  vocabulario = tfidf.vocabulary_
  return(tdm,vocabulario)   

 
# *****************************************************************

def ProcesarLexico(corpus):
  texts=[]
  for texto in corpus:  
    texto = EliminarStopwords(texto.lower())    
    texto = Stemming(texto)     
    texto = EliminaNumerosYPuntuacion(texto)      
    texts.append(texto)
  return(texts)

# *****************************************************************

def ProcesarLexicoSinStemming(corpus):
  texts=[]
  for texto in corpus:  
    texto = EliminarStopwords(texto.lower())       
    texto = EliminaNumerosYPuntuacion(texto)      
    texts.append(texto)
  return(texts)

# *****************************************************************
  
def EliminaNumerosYPuntuacion(s):
    strnum = re.sub(r'\d+', '', s)
    return ''.join(c for c in strnum if c not in punctuation)

# *****************************************************************

def Stemming(sentence): 
   words = word_tokenize(sentence,language="spanish")
   strw = ""
   for word in words:
      strw = strw + stemmer.stem(word) + " "
   return(strw.rstrip()) 

# *****************************************************************

def EliminarStopwords(sentence):
    word_tokens = word_tokenize(sentence,language="spanish")
    stop_words = set(stopwords.words('spanish'))
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = ""
    for w in word_tokens:
       if w not in stop_words:
           filtered_sentence = filtered_sentence + " "+w
    return(filtered_sentence)

# *****************************************************************

def crearQuery(terms,vocab):
    # Crear vector de query del mismo largo que el Vocabulario, con ceros
    query = np.zeros(len(vocab))
    # Stemming de terminos en la query
    listaTerms = word_tokenize(Stemming(terms),language="spanish")
    # Recorre la lista de terminos y cada vez que encuentra uno asigna un "1" en la celda
    # correspondiente a la query (vector binario)
    for t in listaTerms:      
       try:
           ind = vocab[t]
           query[ind] = 1
       except KeyError:
           ind = -1
    # Verificar que el vector query tenga valores diferentes de cero
    # Esto puede pasar cuando no se encuentra los terminos en el modelo
    # Si existe error retorna un vector vacio
    if (np.count_nonzero(query) == 0):
              return([])
    return(query)

# *****************************************************************
    
def RecuperarDocumentosRelevantes(query,model,doc_ids):
  DocList = []
  for idoc in range(len(doc_ids)):
    filename = doc_ids[idoc]  # Obtener el nombre de archivo con el que se compara
    # Calcular el coseno entre el vector de la query con el vector de cada documento
    similitud = 1 - cosine(query,model[:,idoc])
    # agregar a una lista con el par (similaridad, nombre archivo)
    DocList.append((similitud,filename))  
  # ordenar por similaridad la lista DocList
  return(sorted(DocList,reverse=True))

# *****************************************************************

def MostrarDocumentos(Docs):
    print("Lista de documentos relevantes a la query:\n")
    count_r = 0
    for (sim,d) in Docs:
        print("Doc: "+d+" ("+str(sim)+")\n")
        if str(sim) != "0.0":
            count_r = count_r + 1 
    print("Los documentos recuperados son "+str(count_r)+"\n")
    print("Ahora deberá leer manualmente los documentos para averiguar su relevancia\n")
    r_asterisco = int(input("Ingrese el numero de documentos relevantes: "))
    r_recall = int(input("Documentos que debiesen haber sido recuperados: "))
    precision = r_asterisco/count_r
    recall = r_asterisco/r_recall
    f1_score = 2 * precision * recall / (precision + recall)
    print("\nPrecision: "+str(precision))
    print("Recall: "+str(recall))
    print("F1_Score: "+str(f1_score))
    print("\n")

# **************************************************************
# Comienzo programa principal
# **************************************************************

#PATH = "/Users/diegorivera/GoogleDrive/UAI/1_MDS2019/8_AnaliticaTextual/Tarea/CorpusTarea/"
PATH = "C:/Users/Diego/Desktop/MDS/08 - Analitica Textual/CorpusTarea/"
PATH = "C:/Users/Cinthia Karina/Documents/cropus/"

stemmer = SnowballStemmer('spanish')  

corpus,docsID = CrearCorpus(PATH)

tfidf, vocabulario = CrearVSM(corpus)


corpus2 = ProcesarLexicoSinStemming(corpus)
cadena = " ".join(corpus2)

split_it = cadena.split()

Counter = Counter(split_it)

print("****************************")
print("200 palabras más frecuentes")
print("****************************")

most_occur = Counter.most_common(200)
print(most_occur)
print("\n")

print("*********************************************")
print("Features de pacientes afectados de Covid-19")
print("*********************************************")

terms = input("Ingrese query de términos a buscar: ")
query = crearQuery(terms,vocabulario)
if len(query)==0:
    print("ERROR en vector de consulta, no se pueden recuperar documentos!..")
else:
    DocsRelevantes = RecuperarDocumentosRelevantes(query,tfidf,docsID)
    MostrarDocumentos(DocsRelevantes)
