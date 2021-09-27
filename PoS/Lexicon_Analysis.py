import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import unicodedata2
import math
import string
import tokenize
import sklearn
import csv
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from sklearn import metrics
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.naive_bayes import MultinomialNB
from string import digits
from xml.dom import minidom
from unidecode import unidecode
from nltk.stem.snowball import SnowballStemmer
from string import punctuation
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC

"""
CARGA DE DATOS
Se especifica que se esta trabajando en Español, para carga de Stopword y Stemmización
y se incluyen otros terminos considerados Stop words.
:data: tweets previamente filtrados que hablan de seguridad.
:param: spanish . Se emplea para los stopwords y la stemmizacion.
:param: newStopWords.  Se incluyen un conjunto de palabras de unión no consideradas por default.
"""
stop_words = stopwords.words('spanish')
newStopWords = ['dr','dra','etc','bn','ud','u','ag','si','no','rt','q','m','bb','tan','aun','cr','tal','segun','w','lab','aca','wew','av','ah','cll','km','tm','ht','mk','xs','xxl','xl','xxx','reee','nls','kr']
stop_words.extend(newStopWords)
#data = pd.read_csv("predict_data_to_score.csv")
sbEsp = SnowballStemmer('spanish')

"""
PREPROCESAMIENTO
"""
def strip_links(text):
    ''' Eliminación de enlaces
    :param text: tweet a procesar
    :return text: tweet sin enlaces
    '''
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')
    return text


def strip_all_entities(text):
    ''' Eliminación de Hashtagsy menciones y creacion de lista de palabras
    :param text: tweet a procesar
    :return text: lista de las palabras dentro del tweet
    '''
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

# Eliminación de puntuacion, numeros y conversión del texto a minúsculas
def remove_punctuations(text):
    ''' Eliminación de puntuacion, numeros y conversión del texto a minúsculas
    :param text: tweet a procesar
    :return text: tweet en minuscula y sin puntuación
    '''
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    for digits in string.digits:
        text = text.replace(digits,'')
    text = text.lower()
    return text
#Remoción de puntuaciones tanto en letras como en numeros
def remove_punct(strin):
    ''' Eliminación de puntuaciones tanto en letras como en numeros
    :param strin: tweet a procesar
    :return strin: tweet en minuscula y sin puntuación
    '''
    strin = strin.translate(str.maketrans('','',string.punctuation));
    strin = strin.translate(str.maketrans('','',string.digits));
    return strin

#Normalizar: eliminar diéresis, acentos, y otros caracteres similares.
def normunicode_data(strin):
    #print(strin)
    ''' Eliminar diéresis, acentos, y otros caracteres similares
    :param strin: tweet a normalizar
    :return strin: tweet normalizado
    '''
    return unicodedata2.normalize('NFKD', strin).encode('ASCII', 'ignore').decode("utf-8").lower()

def proc_str(strin):
    '''implementación de funciones remove_punct y normunicode_data
    '''
    return remove_punct(normunicode_data(strin))

#Separación de textos en palabras y eliminación de stop_words
def tok_cln(text):
    ''' Separación de textos en palabras y eliminación de stopwords
    :param text: tweet a procesar
    :return set: conjunto de caracteristicas
    '''
    return set(nltk.wordpunct_tokenize(text)).difference(stop_words)

# Aplicación de los pasos de preprocesamiento
def preprocessing(text):
    '''implementación de funciones de preprocesamiento
    '''
    text= text.apply(strip_links)
    text= text.apply(remove_punct)
    text= text.apply(strip_all_entities)
    text = text.apply(normunicode_data)
    return text

#data.Text = preprocessing(data["Text"])

"""
Extracción de características
"""
def proc_string(strin,setData):
    ''' Stemmización de caracteristicase
    :param strin: tweet a procesar
    :return set: conjunto de caracteristicas
    '''
    resp = set([]);
    for data in tok_cln(proc_str(strin)):
        tm  = stemm_data(data)
        #tm = data
        resp.add(tm)
        if tm in setData:
            setData[tm].add(data)
        else:
            setData[tm] = set([data])
    return ', '.join(resp)

#separa cada unas de las palabras del lexicon en diccionarios positivo y negativo
def getVocab(fileName):
    '''separa cada unas de las palabras del lexicon en diccionarios positivo y negativo
    :param : fileName archivo de palabras con score (lexicon)
    return: positive_vocab y negative_vocab: diccionarios con las palabras positivas
    y negativas y su score
    '''
    positive_vocab = [];
    negative_vocab = [];
    xmldoc = minidom.parse(fileName)
    itemlist = xmldoc.getElementsByTagName('senticon')
    for s in itemlist[0].getElementsByTagName('layer'):
        for pl in s.getElementsByTagName('positive'):
            for pll in pl.getElementsByTagName('lemma'):
                positive_vocab.append([pll.firstChild.nodeValue.replace(" ", ""),
                                       float(pll.getAttribute('pol'))]);
        for pl in s.getElementsByTagName('negative'):
            for pll in pl.getElementsByTagName('lemma'):
                negative_vocab.append([pll.firstChild.nodeValue.replace(" ", ""),
                                       float(pll.getAttribute('pol'))]);
    return (positive_vocab,negative_vocab)

#(positive_vocab,negative_vocab) = getVocab('senticon.es.xml')

"""
Cuantificación del sentimiento
"""
##Cuenta las ocurrencias de las palabras negativas y positivas
def getScoreSentiment(words,posText,posScore,negText,negScore,optionScore):
    '''Cuenta las ocurrencias de las palabras negativas y positivas
    :param: words : palabras a clasificador
    :param posText: diccionario de palabras negativas
    :param posScore: score palabras negativas
    :param negText: diccionario palabras positivas
    :param negScore: score palabras positivas
    :param optionScore: escoge el tipo de analisis de sentimiento, simple o con polaridad
    :return finalvalue: PoS normalizado entre 1 y 5
    '''
    countTotalScore = 0;
    for word in words:
        if len(word)>0:
            indicesPos = [i for i, x in enumerate(posText) if word in x.split('_')]
            indicesNeg = [i for i, x in enumerate(negText) if word in x.split('_')]
            cvalP = 0;
            for j in indicesPos:
                if optionScore==0:
                    cvalP += 1;
                if optionScore==1:
                    cvalP += posScore[j];

            cvalN = 0;
            for k in indicesNeg:
                if optionScore==0:
                    cvalN -= 1;
                if optionScore==1:
                    cvalN += negScore[k];

            if (len(indicesNeg)+len(indicesPos))>0:
                countTotalScore+=(cvalP+cvalN)/(len(indicesNeg)+len(indicesPos))

            finalval=(countTotalScore-1)/4
            if finalval>1.0:
                finalval=1.0
            if finalval<-1.0:
                finalval = -1.0
            finalval= (finalval*2)+3
            finalval = int(round(finalval))
    return (finalval)

def computeSentimentScoresDictionary(strWords,posText,posScore,negText,negScore,optionScore):
    vecScoreSentimentDictionary = [];
    for idx in strWords:
        scv = getScoreSentiment(idx.split(' '),posText,posScore,negText,negScore,optionScore);
        vecScoreSentimentDictionary.append(scv)
    df= pd.DataFrame({'Terms':strWords, 'Sentiment Score':vecScoreSentimentDictionary});
    return (vecScoreSentimentDictionary,df);


#posText = [row[0] for row in positive_vocab]
#posScore = [row[1] for row in positive_vocab]
#negText = [row[0] for row in negative_vocab]
#negScore = [row[1] for row in negative_vocab]
#vecScoreSentimentSimple = []
#vecScoreSentimentPolarity = []

#dfpp = data[data['Sentiment'].isin(['1','2','3','4','5'])]

#for idx in dfpp.index:
#    wordsOpinion = [];
#    for data in tok_cln(dfpp['Text'][idx]):
#        tm = data
#        wordsOpinion.append(tm)
#    vecScoreSentimentSimple.append(getScoreSentiment(wordsOpinion,posText,
#                                                     posScore,negText,negScore,0))
#    vecScoreSentimentPolarity.append(getScoreSentiment(wordsOpinion,posText,posScore,negText,negScore,1))

#result = zip(dfpp.index,vecScoreSentimentSimple,vecScoreSentimentPolarity)

#with open('lexicon_f.csv', 'w') as f:
#    writer = csv.writer(f, delimiter =',')
#    writer.writerow(["ID","SentimentSimple","SentimentPolarity"])
#    writer.writerows(result)

#datas = pd.read_csv("predict_data_to_score.csv")
#results = pd.read_csv("lexicon_f.csv")
#results = results.dropna(axis = 1)
#merged = datas.merge(results, on ='ID')
#merged.to_csv("final_lexicon.csv", index = False)

"""
Obtención de metricas
"""
def get_metrics(true_labels, predicted_labels,clasificador):
    '''
    Obtención de metricas accuracy, f1, precision y recall
    '''
    acc=np.round(metrics.accuracy_score(true_labels, predicted_labels),4)
    print(clasificador+' Accuracy:',acc)
    pre=np.round(metrics.precision_score(true_labels,predicted_labels,average='weighted',zero_division=1),4)
    print(clasificador+' Precision:',pre)
    rec=np.round(metrics.recall_score(true_labels, predicted_labels,average='weighted'),4)
    print(clasificador+' Recall:',rec)
    f1=np.round(metrics.f1_score(true_labels, predicted_labels, average='weighted'),4)
    print(clasificador+' F1 score:',f1)
    print('\n')
    return acc,pre,rec,f1


#lexi = pd.read_csv("final_lexicon.csv")
#get_metrics(lexi.Sentiment, lexi.SentimentSimple,"Simple")
#get_metrics(lexi.Sentiment, lexi.SentimentPolarity,"Polarity")
