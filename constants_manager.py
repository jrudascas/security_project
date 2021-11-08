""" 
Variables relacionadas con la conexión API
"""
API_HOST='http://127.0.0.1' # Dirección url principal de acceso al API
API_PORT='8000' # Puerto donde funciona el API
API_RELATIVE_PATH_TOKEN_ACCESS='/analitica/api/auth/token/login' # Dirección relativa para acceder con un token
API_RELATIVE_PATH_UPDATE_PROCESS_STATE = '/analitica/api/v1/anpejecucionproceso/' #Dirección relativa para operaciones relacionadas a la ejecucion de procesos
API_RELATIVE_PATH_GET_TIPOPROCESO = '/analitica/api/v1/anptipoproceso/'
API_RELATIVE_PATH_GET_ESTADOEJECUCION = '/analitica/api/v1/anpestadoejecucion/'
API_USER = 'hadoop'
API_PASSWORD = 'Hadoop$'
IP_EJECUCION = '192.168.3.1'
USUARIO_EJECUCION = 'unal'
PASSWORD = "Unal123"

NAME_ENTRENAMIENTO="ENTRENAMIENTO"
NAME_PREDICCION="PREDICCIÓN"
NAME_VALIDACION="VALIDACIÓN"
NAME_PREPROCESAMIENTO="PREPROCESAMIENTO"

ESTADO_EXITO="EXITO"
ESTADO_ERROR="ERROR"
ESTADO_PROCESO="EN PROCESO"
ESTADO_CANCELADO="CANCELADO"

PATH_POSTGRES_JAR="/home/unal/postgresql-42.2.23.jar"
URL_POSTGRES="jdbc:postgresql://localhost:5432/"
DATABASE_NAME = "analiticades"
TABLE_NAME = "data_percepcion"
TWEETID_COLUMN = "tweetid"
DATE_COLUMN = "fecha_origen"
TEXT_COLUMN = "texto_tweet"
RT_COLUMN = "IsRetweet"
IDFROM_COLUMNN = "num_retweet"
FOLLOWERS_COLUMN = "num_followers"
SCORE_COLUMN = "score"
DATABASE_COVARIADOS = 'data_covariados_percepcion'
DATE_COLUMN_COV = 'FECHA'
COVARIATE_COLUMN = 'covariado'
PARTICULAR_COV_COLUMN = 'particular'