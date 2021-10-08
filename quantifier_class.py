import pandas as pd
import numpy as np
from lexicon_analysis import *
import pickle
import logging

class modelQuantification():
    """
    Clase del modelo de cuantificacion de textos relacionados con seguridad.
    
    :param text_column: nombre columna del dataframe que contine los textos
    :param model_path: direccion del modelo de cuantificacion (leer y escribir)
    :param predict_column: nombre de la columna del dataframe que identidica la clase de cada texto
    :param data_to_train: dataframe de pandas utilizado para entrenar el modelo
    """    
    def __init__(self,
                 text_column,                 
                 model_path,
                 score_column="score",
                 predict_column=None,
                 data_to_train=None
                ):
        logging.debug('Empezo inicialización modelo de Cuantificación.')
        try:
            self.text_column= text_column
            try:
                self.model_path=model_path
                (self.positive_vocab,self.negative_vocab)=getVocab(model_path)
                self.posText = [row[0] for row in self.positive_vocab]
                self.posScore = [row[1] for row in self.positive_vocab]
                self.negText = [row[0] for row in self.negative_vocab]
                self.negScore = [row[1] for row in self.negative_vocab]
                logging.debug('Conjuntos palabras positivas y negativas cargados.')
            except:
                msg_error = "No se pudo cargar el modelo de cuantificación con la ruta especificada"
                logging.error(msg_error)
                raise Exception(msg_error)
            
            self.score_column=score_column
            # si el modelo no se da se tienen que dar la base de datos de entranamiento, la columna de prediccion y el 
            # porcentaje de los datos de validacion
            if predict_column == None:
                pass
            else:
                try:
                    self.data_to_train=data_to_train
                    self.predict_column=predict_column
                    if self.data_to_train[self.predict_column].nunique() != 5:
                        msg_error="La columna a predecir es constante o aparecen mas de 5 clases"
                        logging.error(msg_error)
                        raise Exception(msg_error)                        
                except:
                    msg_error="La base de datos no tiene la columna especificada: "+str(self.predict_column)
                    logging.error(msg_error)
                    raise Exception(msg_error)
            logging.debug('Termino inicialización modelo de Cuantificación.')        
        except Exception as e:
            msg_error="No se pudo completar con la inicialización del modelo de cuantificación"
            logging.error(msg_error)
            raise Exception(msg_error + " / " +str(e))
    def quantify_df(self,DataFrame):
        """
        Funcion para cuantificar un dataframe
        :param DataFrame: Dataframe de pandas con los datos a clasificar
        :return: Dataframe con las filas que pasaron la clasificacion
        """
        logging.debug('Empezo proceso de cuantificación base de datos.')
        try:
            df=DataFrame.copy()
            get_score=lambda a: getScoreSentiment(a,self.posText,self.posScore,self.negText,self.negScore,1)
            df[self.score_column]=(preprocessing(df[self.text_column]).apply(tok_cln)).apply(get_score)
            return df
        except:
            msg_error="Error en la cuantificacion de textos"
            logging.error(msg_error)
            raise Exception(msg_error)

    
    def train_model(self):
        """
        Funcion para entrenar el modelo de clasificacion dado un dataframe etiquetado
        :return: scores de clasificacion obtenidos en el conjunto de validacion
        """
        logging.debug('Empezo proceso de entrenamiento modelo cauntificación.')
        try:
            df=self.quantify_df(self.data_to_train)
            accuracy,precision,recall,f1=get_metrics(df[self.predict_column],df[self.score_column],"Polarity")
            self.scores={"Accuracy":accuracy,"Precision":precision,"Recall":recall,"F1-score":f1}
            return self.scores
            
        except:
            msg_error="No se completo el entrenamiento del modelo de cuantificacion de textos"
            logging.error(msg_error)
            raise Exception(msg_error)

    def save_model(self,path):
        """
        Funcion para guardar el objeto de la clase como archivo pkl
        """
        try:
            with open(path, 'wb') as outp:
                pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
            logging.debug('Modelo de cuantificación de tweets guardado exitosamente.')
        except:
            msg_error="No se pudo guardar el modelo de cuantificación con la direccion establecida"
            logging.error(msg_error)
            raise Exception(msg_error)