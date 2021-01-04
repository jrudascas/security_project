import fasttext
from gensim.models import KeyedVectors
import pandas as pd

def predict(model,data_new,percent=True):
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
    keys=pd.read_excel("entradas/19032020_Palabras_Filtro.xls")
    keys=list(keys[keys.columns[0]].values)
    keys=[i.replace('*','').replace('+','') for i in keys]
    keys=list(pd.Series(keys).apply(clean_text))
    
    model_C=fasttext.load_model("results/best_fasttext.bin")
    
    new_values=list(data.FullText.apply(clean_text))
    filtro=[ True in [ False not in [i in k for i in j] for j in keys] for k in new_values]
    data=data[filtro]
    clean_text=list(data.FullText.apply(clean_text_join))
    pred=predict(model_C,
                 clean_text,
                 percent=False)
    return data[pred > 0]


