from joblib import dump, load
from pre_procesamiento import clean_text_join, clean_text
from gensim.models import KeyedVectors
import pandas as pd

import pickle
import numpy as np
from sklearn.metrics import f1_score,accuracy_score,precision_recall_fscore_support,balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import logging


class modelClassification():
    """
    Clase del modelo de clasificación de textos relacionados con seguridad.
    
    :param keywords_path: direccion del excel que contiene las palabras clave compuestas
    :param vectors_path: direccion del archivo de transformacion de palabras a vectores
    :param text_column: nombre columna del dataframe que contine los textos
    :param model_path: direccion del modelo de clasificacion (leer y escribir)
    :param predict_column: nombre de la columna del dataframe que identidica la clase de cada texto
    :param data_to_train: dataframe de pandas utilizado para entrenar el modelo
    :param percent_val_data: porcentaje del dataframe de entrenamiento utilizado para la validacion
    """    
    def __init__(self,
                 keywords_path,
                 vectors_path,
                 text_column,
                 model_path,
                 predict_column=None,
                 data_to_train=None,
                 percent_val_data=0.3
                ):
        logging.debug('Empezo inicialización modelo de Clasificación.')
        try:
            self.text_column= text_column
            try:
                self.vectors_words= KeyedVectors.load_word2vec_format(vectors_path)
                logging.debug('Vectores de embebimiento de palabras cargado.')
            except OSError as error:
                msg_error="No se pudo cargar los vectores de palabras con la ruta especificada."
                logging.error(msg_error)
                raise Exception(msg_error)
            try:
                #preprocesamiento keywords
                keys=pd.read_excel(keywords_path)
                keys=list(keys[keys.columns[0]].values)
                keys=[i.replace('*','').replace('+','') for i in keys]
                keys=list(pd.Series(keys).apply(clean_text))
                self.keys=keys
                logging.debug('Keywords de seguridad cargado y procesado.')
            except:
                msg_error="No se pudo cargar las keywords con la ruta especificada"
                logging.error(msg_error)
                raise Exception(msg_error)
            
            self.model_path=model_path
            # si el modelo no se da se tienen que dar la base de datos de entranamiento, la columna de prediccion y el 
            # porcentaje de los datos de validacion
            if predict_column == None:
                try:
                    self.model=load(model_path)
                    logging.debug('Modelo de clasificación pre-entrenado cargado.')
                except:
                    msg_error="No se pudo cargar el modelo de classificación pre-entrenado con la ruta especificada."
                    logging.error(msg_error)
                    raise Exception(msg_error)
            else:
                try:
                    self.data_to_train=data_to_train
                    self.predict_column=predict_column
                    if self.data_to_train[self.predict_column].nunique() != 2:
                        msg_error="La columna a predecir es constante o aparecen más de 2 clases"
                        logging.error(msg_error)
                        raise Exception(msg_error)
                    self.percent_val_data=percent_val_data
                except:
                    msg_error ="La base de datos no tiene la columna especificada: "+str(self.predict_column)
                    logging.error(msg_error)
                    raise Exception(msg_error)
            logging.debug('Termino inicialización modelo de Clasificación.')
        except Exception as e:
            msg_error= "No se pudo completar con la inicialización del modelo de clasificación"
            logging.error(msg_error)
            raise Exception(msg_error + " / " +str(e))
        
    def filter_keys(self,text):
        """
        Filtrado de un texto para identificar si contiene las palabras clave

        :param text: cadena de strings
        :return: 1 si pasa el filtro y 0 en caso contrario
        """
        return (True in [ True in [i in text for i in j] for j in self.keys])*1

    def get_vectors(self,text):
        """
        Conversion de un texto a vector de entradas numericas

        :param text: cadena de strings
        :return: vector promedio de las representaciones de las palabras que contiene el texto
        """
        R=np.zeros(100)
        Vecs = [self.vectors_words[i] for i in str(text).split() if i in self.vectors_words.vocab]
        if len(Vecs) > 0:
            R=np.array(Vecs).mean(axis=0)
        return R
    
    def clasify_df(self,DataFrame):
        """
        Funcion para clasificar un dataframe
        :param DataFrame: Dataframe de pandas con los datos a clasificar
        :return: Dataframe con las filas que pasaron la clasificacion
        """
        logging.debug('Empezo clasificación base de datos.')
        try:
            df=DataFrame.copy()
            df['clean']=df[self.text_column].apply(clean_text_join)
            df['filter']=df['clean'].apply(self.filter_keys)
            df=df[df['filter'] == 1]
            df['vectors']=df.clean.apply(self.get_vectors)
            df['predict']=self.model.predict(np.array(list(df['vectors'].values)))
            df=df[df['predict'] == 1]
            logging.debug('Termino clasificación base de datos.')
            return df[DataFrame.columns]
        except:
            msg_error = "No se pudo terminar el proceso de clasificación"
            logging.error(msg_error)
            raise Exception(msg_error)
    
    def train_model(self):
        """
        Funcion para entrenar el modelo de clasificacion dado un dataframe etiquetado
        :return: scores de clasificacion obtenidos en el conjunto de validacion
        """
        logging.debug('Empezo entrenamiento modelo de clasificación.')
        try:
            train,val=train_test_split(self.data_to_train,test_size=self.percent_val_data)
            while (train[self.predict_column].nunique(),val[self.predict_column].nunique()) != (2,2):
                train,val=train_test_split(self.data_to_train,test_size=self.percent_val_data)
            for idx,df in enumerate([train,val]):
                df['clean']=df[self.text_column].apply(clean_text_join)
                df['filter']=df['clean'].apply(self.filter_keys)
                if idx == 0:
                    df.drop(df.loc[df['filter']==0].index, inplace=True)
                df['vectors']=df.clean.apply(self.get_vectors)
            self.model=SVC(**{'C': 10,'class_weight': 'balanced','degree': 2,'gamma': 0.1,'kernel': 'rbf'})
            self.model.fit(np.array(list(train['vectors'].values)), train[self.predict_column].values)
            y_pred = self.model.predict(np.array(list(val['vectors'].values)))
            y_pred[np.array(val['filter'].values == 0)]=0
            accuracy=accuracy_score(val[self.predict_column].values,y_pred)
            precision,recall,f1,_=precision_recall_fscore_support(val.seguridad.values,y_pred,average="binary")
            #print(accuracy,precision,recall,f1)
            self.scores={"Accuracy":accuracy,"Precision":precision,"Recall":recall,"F1-score":f1}
            dump(self.model, self.model_path)
            logging.debug('Termino entrenamiento modelo de clasificación.')
            return self.scores
        except:
            msg_error="No se pudo completar el entrenamiento del modelo de clasificación"
            logging.error(msg_error)
            raise Exception(msg_error)

    def save_model(self,path):
        """
        Funcion para guardar el objeto de la clase como archivo pkl
        """
        try:
            with open(path, 'wb') as outp:
                pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
            logging.debug('Modelo de clasificación de tweets guardado exitosamente.')
        except:
            msg_error="No se pudo guardar el modelo de clasificación con la direccion establecida"
            logging.error(msg_error)
            raise Exception(msg_error)