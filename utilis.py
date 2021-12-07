import requests
import traceback
import json
import constants_manager as c
from datetime import datetime
from pyspark.sql import SparkSession, functions
import logging
import pandas as pd
from datetime import timedelta, datetime

def get_token_acces():
    """
    Obtiene un token para realizar operaciones en el API
    :return: string token
    """
    print('Obteniendo token ...')
    token_response = requests.post(c.API_HOST + ':' + c.API_PORT + c.API_RELATIVE_PATH_TOKEN_ACCESS, data={'username' : c.API_USER, 'password' : c.API_PASSWORD})
    #print(token_response.json())
    token = token_response.json()["auth_token"]
    return token

def get_tipos_proceso(token):
    """
    Obtiene la información de los tipos de proceso de la API

    :param token: string token 
    :return: dict con keys los tipos de procesos y valores los ids en el API
    """
    auth_header={'Authorization' : 'Token ' + token}
    response = requests.get(c.API_HOST + ':' + c.API_PORT + c.API_RELATIVE_PATH_GET_TIPOPROCESO, headers=auth_header)
    dict_response = response.json()
    tipos_proceso={c.NAME_ENTRENAMIENTO:0,c.NAME_PREDICCION:0,c.NAME_VALIDACION:0,c.NAME_PREPROCESAMIENTO:0}
    for i in dict_response:
        if i["nombre_tipo_proceso"] in tipos_proceso:
            tipos_proceso[i["nombre_tipo_proceso"]]=i['id_tipo_proceso']

    return tipos_proceso
    
def get_estados_ejecucion(token):
    """
    Obtiene la información de los estados de ejecución de la API

    :param token: string token 
    :return: dict con keys los estados de procesos y valores los ids en el AP
    """
    auth_header={'Authorization' : 'Token ' + token}
    response = requests.get(c.API_HOST + ':' + c.API_PORT + c.API_RELATIVE_PATH_GET_ESTADOEJECUCION, headers=auth_header)
    dict_response = response.json()

    estados={c.ESTADO_EXITO:0,c.ESTADO_ERROR:0,c.ESTADO_PROCESO:0,c.ESTADO_CANCELADO:0}
    for i in dict_response:
        if i["nombre_estado_ejecucion"] in estados:
            estados[i["nombre_estado_ejecucion"]]=i['id_estado_ejecucion']

    return estados


def update_process_state(id_tipo_proceso, id_estado_ejecucion, token):
    """
    Actualiza un estado de proceso

    :param id_tipo_proceso: id del tipo de proceso a actualizar
    :param id_estado_ejecucion: id del estado de ejecucion del proceso a actualizar
    :param token: string token
    :return: 
    """
    fecha_actual = datetime.now()
    time_stamp = fecha_actual.strftime('%Y-%m-%-d %H:%M:%S')
    auth_header={'Authorization' : 'Token ' + token}

    try:
        data = {'fecha_hora_proceso' : time_stamp, 'usuario_ejecucion' : c.USUARIO_EJECUCION, 'ip_ejecucion' : c.IP_EJECUCION, 'id_tipo_proceso' : id_tipo_proceso, 'id_estado_ejecucion' : id_estado_ejecucion}
        #print(data)
        response = requests.post(c.API_HOST + ':' + c.API_PORT + c.API_RELATIVE_PATH_UPDATE_PROCESS_STATE, headers=auth_header, data=data)
        response.raise_for_status()
        #print(response.json())
        logging.debug("Actualización estado de proceso en el API realizada")
    except requests.exceptions.HTTPError as error:
        print(error)
        traceback.print_exc()
        logging.debug("No se completo la actualización estado de proceso en el API")

def colum_rt(input_):
    """
    Función para establecer si un tweet es original o retweet

    :param input_: entrada de validación
    :return: valor 0 o 1 correspondiente a original y retweet 
    """
    if input_ == "0" :
        return 0
    else: 
        return 1


def get_data_from_postgress(database_url=c.URL_POSTGRES,
                            database=c.DATABASE_NAME,
                            table=c.TABLE_NAME,
                            user=c.USUARIO_EJECUCION,
                            password=c.PASSWORD,
                            postgres_jar=c.PATH_POSTGRES_JAR,
                            ):
    """
    Función para descargar los datos desde la base de datos postgress

    :param database_url: string dirección url de la base de datos
    :param database: string nombre base de datos
    :param table: string nombre de la tabla
    :param user: string nombre usuario
    :param password: string password
    :param postgres_jar: string dirección postgress jar
    :return: pyspark dataframe
    """
    try:
        logging.debug("Empezo descargue de datos desde base de datos postgress")
        spark=SparkSession.builder.appName("Python Spark SQL basic example").config("spark.jars",postgres_jar).config("spark.executor.memory", "70g").config("spark.driver.memory", "50g").config("spark.memory.offHeap.enabled",True).config("spark.memory.offHeap.size","16g") .getOrCreate()
        spark.sparkContext.setLogLevel('FATAL')
        df=spark.read.format("jdbc").option("url",database_url+database).option("dbtable",table).option("user", user).option("password", password).option("driver", "org.postgresql.Driver").load()
        logging.debug("Termino descargue de datos desde base de datos postgress")
        return df
    except:
        msg_error="No se completo la descarga de datos desde la base de datos"
        logging.debug(msg_error)
        raise Exception(msg_error)



def spark_to_pandas(df,
                    start_data=None,
                    tipe="tweets",
                    summary=None
                    ):
    """
    Función para convertir los datos descargados al formato necesario
    :param df: spark dataframe
    :param start_data: string fecha inicio de los datos
    :param summary: summary  file    
    :return: pandas dataframe con los datos descargados
    """
    try:
        logging.debug("Empezo proceso inicial de datos descargados desde base de datos postgress")
        date_c = c.DATE_COLUMN
        if tipe == "cov":
            date_c = "FECHA"


        df=df.withColumn(date_c,functions.to_timestamp(date_c,'yyyy-MM-dd HH:mm:ss.SSS'))
        logging.debug('Conversión fechas a formato datetime.')
        if tipe == "tweets":
            df = df.withColumn(date_c, functions.to_utc_timestamp(date_c,"+05:00"))
        if start_data != None:
            df=df.filter(date_c+" > date'"+start_data+"'")


        df=df.toPandas()
        if start_data != None:
            df=df[df[date_c]>start_data]
        if tipe == "tweets":
            duplicates = df.duplicated().sum()
            if summary != None:
                summary.write("Cantidad de tweets descargados de la base de datos: "+ str(len(df)) +"\n")
                summary.write("Cantidad de tweets originales: "+ str(sum(df[c.IDFROM_COLUMNN]=='0'))+"\n")
                summary.write("Cantidad de filas duplicadas: "+ str(duplicates)+"\n")   
            df.drop_duplicates(c.TWEETID_COLUMN,inplace=True)

            if duplicates !=0 and summary != None:
                summary.write("Cantidad de tweets: "+ str(len(df)) + "\n")
                summary.write("Cantidad de tweets originales: "+ str(sum(df[c.IDFROM_COLUMNN]=='0')) +"\n")

            logging.debug('Eliminación de posibles duplicados.')
            df[c.FOLLOWERS_COLUMN]=pd.to_numeric(df[c.FOLLOWERS_COLUMN],errors="coerce")
            df[c.FOLLOWERS_COLUMN].fillna(0,inplace=True)
            df.sort_values(c.DATE_COLUMN,inplace=True)
            df[c.RT_COLUMN]=df[c.IDFROM_COLUMNN].apply(colum_rt)
            
        logging.debug("Termino proceso incial de datos descargados desde base de datos postgress")
        return df
    except:
        msg_error="No se completo la conversion de los datos descargados"
        logging.debug(msg_error)
        raise Exception(msg_error)
