import pandas as pd
import numpy as np
from Lexicon_Analysis import *

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
        self.text_column= text_column
        try:
            self.model_path=model_path
            (self.positive_vocab,self.negative_vocab)=getVocab(model_path)
            self.posText = [row[0] for row in self.positive_vocab]
            self.posScore = [row[1] for row in self.positive_vocab]
            self.negText = [row[0] for row in self.negative_vocab]
            self.negScore = [row[1] for row in self.negative_vocab]
        except:
            print("Error: No se pudo cargar el modelo con la ruta especificada")
        
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
                    print("Error: La columna a predecir esconstante o aparecen mas de 5 clases")
            except:
                print("Error: La base de datos no tiene la columna especificada: "+str(self.predict_column))
                
    def quantify_df(self,DataFrame):
        """
        Funcion para cuantificar un dataframe
        :param DataFrame: Dataframe de pandas con los datos a clasificar
        :return: Dataframe con las filas que pasaron la clasificacion
        """
        try:
            df=DataFrame.copy()
            get_score=lambda a: getScoreSentiment(a,self.posText,self.posScore,self.negText,self.negScore,1)
            df[self.score_column]=(preprocessing(df[self.text_column]).apply(tok_cln)).apply(get_score)
            return df
        except:
            print("Error en la cuantificacion de textos")
    
    def train_model(self):
        """
        Funcion para entrenar el modelo de clasificacion dado un dataframe etiquetado
        :return: scores de clasificacion obtenidos en el conjunto de validacion
        """
        try:
            df=quantify_df(self.data_to_train)
            accuracy,precision,recall,f1=get_metrics(df[self.predict_column],df[self.score_column],"Polarity")
            self.scores={"Accuracy":accuracy,"Precision":precision,"Recall":recall,"F1-score":f1}
            return self.scores
            
        except:
            print("Error en el entrenamiento del modelo")