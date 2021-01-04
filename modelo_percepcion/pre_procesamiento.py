import pandas as pd
import numpy as np
import unicodedata
import re
import nltk.data
from gensim.corpora.dictionary import Dictionary
from emoji import UNICODE_EMOJI
from sklearn.metrics import confusion_matrix

from nltk.corpus import stopwords
spanish_stopwords = stopwords.words('spanish')
stop_words=[unicodedata.normalize('NFKD', str(i)).encode('ASCII', 'ignore').decode("utf-8") for i in spanish_stopwords]

#Eliminación indicador retweet
def remove_RT(text):
    text=unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode("utf-8")    
    if text[:2]=='RT':
        index=text.find(":")
        return text[index+2:]
    else:
        return text
#Eliminación urls
def Find(string): 
    url = re.findall('http[s]?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string) 
    twi_url=re.findall('pic.twitter.com(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)
    return url+twi_url 

def remove_url(text):
    urls=Find(text)
    for u in urls:
        text=text.replace(u,"")
    return text

# replace tags ej: # LeyDeFinanciamiento por Ley De Financiamiento 
def replace_tags(text):
    tags=re.findall('#(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for t in tags:
        if t.isupper():
            text=text.replace(t,t[1:])
        else:            
            text=text.replace(t," ".join(re.split('(?=[A-Z])', t[1:])[1:]))
    return text  

# remove @ symbol
def remove_arroba(text):
    tags=re.findall('@(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for t in tags:
        text=text.replace(t,t[1:])
    return text 
# remove emoji 


def text_has_emoji(text):
    for character in text:
        if character in UNICODE_EMOJI:
            return True
    return False

def remove_emoji(text):
    for character in text:
        if character in UNICODE_EMOJI:
            text=text.replace(character,"")
    text=text.replace("\n"," ")
    text=text.replace("!","")
    text=text.replace("¡","")
    text=text.replace("."," ")
    text=text.replace(":","")
    text=text.replace("?","")
    text=text.replace("¿","")
    text=text.replace("“","")
    text=text.replace("→"," ")
    text=text.replace("…"," ")
    text=text.replace(";","")
    text=text.replace(",","")
    text=text.replace("("," ")
    text=text.replace(")"," ")
    text=text.replace("`"," ")
    text=text.replace("'"," ")
    text=text.replace('"'," ")
    text=text.replace('%'," ")
    text=text.replace('|'," ")
    text=text.replace('-'," ")
    text=text.replace('['," ")
    text=text.replace(']'," ")
    text=text.replace('--'," ")
    text=text.replace('$'," ")
    text=text.replace('@'," ")
    text=text.replace('#'," ")
    text=text.replace('*'," ")
    text=text.replace('/'," ")
    text=text.replace('&'," ")
    text=text.replace('gt'," ")
     
    
    
     
    text=re.sub('\d', '', text)  
     
    
    return text.lower()


##### remove stop words

def remove_stopwords(text):
    from nltk.tokenize import word_tokenize 
    word_tokens = word_tokenize(text) 
    return tuple([w for w in word_tokens if not w in stop_words])

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')

def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed



def clean_text(i):
    
    
    functions=[remove_RT,remove_url,replace_tags,remove_arroba,remove_emoji,remove_stopwords,stem_tokens]
    text=i
    for j in functions:
        text=j(text)
    return text


def clean_text_join(i):
    return " ".join(clean_text(i))

def split_text(i):
    return str(i).split(" ")



def plot_confusion_matrix(cm,
                          target_names,
                          f1=0,
                          title='Matriz de Confusión',
                          path_save="",
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = f1
    

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
#     plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\nf1_score={:0.4f}'.format(accuracy))
    plt.savefig(path_save,bbox_inches='tight')
    plt.show()

