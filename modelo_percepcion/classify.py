import fasttext
from gensim.models import KeyedVectors
import pandas as pd
from joblib import dump, load

def predict(model,data_new,percent=True):
    '''
    Funcion de aplicacion de prediccion a un dataframe con textos que tiene la opcion
    de devolver solo los valores predichos o sus porcentajes respectivos a las clases.
    
    entradas: 
    
        model: modelo de clasificacion
        data_new: dataframe de pandas con los datos a clasificar
        percent: Flag de verificacion si devuelve los porcentajes o solo las
                 predicciones
        
    ########
    
    Salida:
        Dataframe con los datos clasificados
    
    
    '''
    A,B=model.predict(data_new,k=2)
    percents=[]
    for a,b in zip(A,B):
        Z = [x for _,x in sorted(zip(a,b))]
        percents.append(Z)
    percents=pd.DataFrame(percents)
    if percent==True:
        return percents
    else:
        return percents.idxmax(axis=1).values

def filter_keys(text):
    '''
    Funcion de aplicacion del filtrado por palabras clave a un solo texto
    
    entradas: 
    
        text: string que contiene el texto a filtrar
        
    ########
    
    Salida:
        valor de verdad si un texto continene las palabras clave
    
    
    '''
    return (True in [ True in [i in text for i in j] for j in keys])*1

def get_vectors(text):
    
    '''
    Funcion de aplicacion de conversion de textos a vectores numericos
    
    entradas: 
    
        text: string que contiene el texto a convertir
        
    ########
    
    Salida:
        vector de valores del embebimiento realizado
    
    
    '''
    
    R=np.zeros(100)
    Vecs = [model[i] for i in str(text).split() if i in model.vocab]
    if len(Vecs) > 0:
        R=np.array(Vecs).mean(axis=0)
    return R    
    
def predict_from_data(data):
    
    '''
    Funcion de clasificacion de tweets a partir del modelo pre-entrenado y los datos en un dataframe de pandas
    
    entradas: 
    
        data: dataframe de pandas con la información de los tweets
        
    ########
    
    Salida:
        subconjunto de los datos que pasan la clasificacion
    
    
    '''
    from pre_procesamiento import clean_text_join, clean_text
    ## carga modelo preentrenado
    clf=load('SVM_class.joblib')
    # keywords compuestas
    keys=pd.read_excel("entradas/19032020_Palabras_Filtro.xls")
    # preprocesamiento keywords
    keys=list(keys[keys.columns[0]].values)
    keys=[i.replace('*','').replace('+','') for i in keys]
    keys=list(pd.Series(keys).apply(clean_text))
    new_values=[str(i).split() for i in Train.clean.values]
    # Aplicacion filtrado por keywords
    filtro=[ True in [ True in [i in k for i in j] for j in keys] for k in new_values]
    new_values=[str(i).split() for i in Val.clean.values]
    filtro2=[ True in [ True in [i in k for i in j] for j in keys] for k in new_values]
    # modelo de embebimientos pre entrenados
    model = KeyedVectors.load_word2vec_format("Train_vectors/words_vectors.vec")
    # preprocesamiento
    data['clean']=data.Text.apply(clean_text_join)
    
    data['filter']=data['clean'].apply(filter_keys)
    
    data=data[data['filter'] == 1]
    # vectores de los textos
    data['vectors']=data.clean.apply(get_vectors)
    # prediccion con el modelo de clasificacion
    data['predict']=clf.predict(np.array(list(data['vectors'].values)))
    data=data[data['predict'] == 1]
    
   
    return data


