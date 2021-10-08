import sys
from general_class import *
import logging
import argparse
import constants_manager as cm
from utilis import get_data_from_postgress

###### variables fijas temporalmente
path_classify_model_="/home/unal/percepcion/Resultados/1Unal/ClassificationModel.pkl"
path_quantify_model_="/home/unal/percepcion/Resultados/1Unal/QuantificationModel.pkl"

data_cov=['/home/unal/percepcion/security_project/entradas/Partidos.csv',
          '/home/unal/percepcion/security_project/entradas/Manifestaciones.csv',
          '/home/unal/percepcion/security_project/entradas/FechasEspeciales.csv']

partidos=pd.read_csv(data_cov[0])
manifestaciones=pd.read_csv(data_cov[1])
f_especiales=pd.read_csv(data_cov[2])

partidos_M=pd.to_datetime(partidos['Fecha'].dropna())
partidos_S=pd.to_datetime(partidos['Fecha.1'].dropna())
manifestaciones=pd.to_datetime(manifestaciones['Fecha'].dropna())
f_especiales=pd.to_datetime(f_especiales[['2019','2020']].values.flatten())
f_especiales=f_especiales[~f_especiales.isna()]

def TC(t):
    return np.array([
                        t.weekday()/6.0,
                        (t.hour > 12)*1,
                        (pd.Timestamp(t.date()) == partidos_M).sum(),
                        (pd.Timestamp(t.date()) == partidos_S).sum(),
                        (pd.Timestamp(t.date()) == manifestaciones).sum(),
                        #(pd.Timestamp(t.date()) == f_especiales).sum(),
                        1
                    ])

def info_data(data,column_date):
    data_=data.copy()
    data_[column_date]=pd.to_datetime(data_[column_date])-timedelta(hours=6)
    min_,max_=data_[column_date].min(),data_[column_date].max()
    return str(min_),str(max_)



def process(log_file,
            save_model,
            subprocess,
            f_limite=None,
            predict_period=None,
            save_result_predict=None,
            exist_model_path=None,
            valid_period=None,
            path_classify_model=path_classify_model_,
            keywords_path=None,
            vectors_path=None,
            cmodel_path=None,
            predict_column=None,
            data_to_train_clasify=None,
            percent_val_data=0.3,
            path_quantify_model=path_quantify_model_,
            qmodel_path=None,
            score_column="score",
            text_column="Text",
            TweetId_column="TweetId",
            RT_column="IsRetweet",
            IdFrom_column="tweet_from",
            date_column="CreatedAt",              
            followers_column="followers",
            win_size_for_partition_cov=1,
            followers_rate=2,
            win_size_infectious_rate = 3,
            win_size_train_period = 1,
            win_size_pred_period = 1,
            method_pred = 'integral'
            ):
    logging.basicConfig(filename=log_file,level=logging.DEBUG)
    logging.debug('Empezo función process.')

    try:
   
        if exist_model_path == None:
            logging.debug('Verificación archivos para crear modelos.')
            ## modelo clasificación de tweets relacionados con seguridad
            if not os.path.isfile(path_classify_model):
                logging.debug('No se encontro modelo clasificación tweets existente con la dirección establecida.')
                try:
                    data_to_train_clasify=pd.read_csv(data_to_train_clasify)
                except:
                    msg_error="No se pudo cargar archivo csv para entrenar modelo clasificación de tweets"
                    logging.debug(msg_error)
                classify_model=modelClassification(keywords_path,
                                                   vectors_path,
                                                   text_column,
                                                   cmodel_path,
                                                   predict_column,
                                                   data_to_train_clasify,
                                                   percent_val_data
                                                   )
                classify_model.save_model(path_classify_model)
            else:
                try:
                    with open(path_classify_model, "rb") as input_file:
                        classify_model = pickle.load(input_file)
                    logging.debug('Modelo clasificación tweets existente cargado exitosamente.')
                except:
                    msg_error="No se pudo cargar el modelo de clasificación de tweets con la dirección establecida"
                    logging.error(msg_error)
                    raise Exception(msg_error)
            ## modelo de cuantificación de tweets relacionados con seguridad
            if not os.path.isfile(path_quantify_model):
                logging.debug('No se encontro modelo cuantificación tweets existente con la dirección establecida.')
                quantify_model=modelQuantification(text_column,
                                                   qmodel_path,
                                                   score_column
                                                   )
                quantify_model.save_model(path_quantify_model)
            else:
                try:
                    with open(path_quantify_model, "rb") as input_file:
                        quantify_model = pickle.load(input_file)
                    logging.debug('Modelo cuantificación tweets existente cargado exitosamente.')
                except:
                    msg_error="No se pudo cargar el modelo de cuantificación de tweets con la dirección establecida"
                    logging.error(msg_error)
                    raise Exception(msg_error)
            ## modelo tweets

            ## lectura datos
            data=get_data_from_postgress()
            train_period=info_data(data,date_column)
            print(train_period)
            if f_limite == None:
                f_limite = train_period[0]

            generalmodel=modelPercepcion(data,
                                         classify_model,
                                         quantify_model,
                                         train_period,
                                         valid_period,
                                         f_limite,
                                         TweetId_column,
                                         RT_column,
                                         IdFrom_column,
                                         date_column,
                                         followers_column,
                                         score_column,
                                         f_covariates=(TC,restore_date),
                                         win_size_for_partition_cov=win_size_for_partition_cov,
                                         followers_rate=followers_rate,
                                         win_size_infectious_rate = win_size_infectious_rate,
                                         win_size_train_period = win_size_train_period,
                                         win_size_pred_period = win_size_pred_period,
                                         method_pred = method_pred
                                        )
            _,info=generalmodel.prepare_data()
            print(len(info['Tweets']))
            generalmodel.create_model_tweets(info)

        else:
            try:
                with open(exist_model_path, "rb") as input_file:
                        generalmodel = dill.load(input_file)
                logging.debug('Modelo percepción de seguridad existente cargado exitosamente.')
            except Exception:
                msg_error="No se encontro modelo existente con la dirección establecida."
                logging.error(msg_error)
                raise Exception(msg_error)

            last_train_period=generalmodel.tweets_model.train_period
            # nueva tanda de datos desde last_train_period[1]
            data=get_data_from_postgress(last_train_period[1])
            
            _,end=info_data(data,date_column)
            if f_limite == None:
                train_period=(last_train_period[0],end)
            else:
                generalmodel.f_limite = datetime.fromisoformat(str(f_limite))
                train_period=(f_limite,end)
                _,info=generalmodel.prepare_data(new_data=True)
                generalmodel.create_model_tweets(info)

        if subprocess == "train":
                generalmodel.train_model()

        if subprocess == "predict":
            if not (hasattr(generalmodel.tweets_model,"Beta") & hasattr(generalmodel.tweets_model,"param_infectious_fit")):
                logging.debug('El modelo debe ser entrenado primero.')
                generalmodel.train_model()
            predict_period=tuple(predict_period.split(","))
            generalmodel.predict_model(predict_period,save_result_predict)
        
        generalmodel.save_model(save_model)
        
        logging.debug('Termino función process.')

        return generalmodel

    except Exception as e:
        msg_error = "No se completo función process"
        logging.error(msg_error)
        raise Exception(msg_error + " / " +str(e))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Función ejecución proceso')
    parser.add_argument("--log_file",required=True,help="Dirección archivo de log")
    parser.add_argument("--save_model",required=True,help="Dirección donde se guardara el archivo del modelo resultante")
    parser.add_argument("--subprocess",required=True,help="Subproceso a ejecutar",default="clean",choices=["clean","train","predict"])
    parser.add_argument("--predict_period",default=None,required=False,help="Periodo de predicción")
    parser.add_argument("--save_result_predict",default=None,required=False,help="Dirección donde se guarda el resultado de predicción")
    
    parser.add_argument("--f_limite",default=None,required=False,help="Fecha limite")
    parser.add_argument("--exist_model_path",default=None,required=False,help="Dirección modelo previo existente")
    parser.add_argument("--valid_period",default=None,required=False,help="Perido de validación modelo")
    parser.add_argument("--path_classify_model",default=path_classify_model_,required=False,help="Dirección modelo clasificación de tweets")
    parser.add_argument("--keywords_path",default=None,required=False,help="Dirección keywords para modelo clasificación de tweets")
    parser.add_argument("--vectors_path",default=None,required=False,help="Dirección archivo vectores palabras para modelo clasificación de tweets")
    parser.add_argument("--cmodel_path",default=None,required=False,help="Dirección modelo clasificación base de tweets")
    parser.add_argument("--predict_column",default=None,required=False,help="Nombre columna objetivo entrenamiento modelo clasificación de tweets")
    parser.add_argument("--data_to_train_clasify",default=None,required=False,help="Dirección archivo csv para entrenar modelo clasificación de tweets")
    parser.add_argument("--percent_val_data",default=0.3,required=False,help="Porcentaje datos de validación para modelo clasificación de tweets")
    parser.add_argument("--path_quantify_model",default=path_quantify_model_,required=False,help="Dirección modelo cuantificación de tweets")    
    parser.add_argument("--qmodel_path",default=None,required=False,help="Dirección modelo cuantificación base de tweets")
    parser.add_argument("--score_column",default="score",required=False,help="Nombre nueva columna resultante proceso cuantificación de tweets")
    parser.add_argument("--text_column",default=cm.text_column,required=False,help="Nombre columna que contiene eltexto de los tweets")
    parser.add_argument("--date_column",default=cm.date_column,required=False,help="Nombre columna fechas en las que se publican los tweets")

    parser.add_argument("--TweetId_column",default=cm.TweetId_column,required=False,help="Nombre columna que contiene los identificadores unicos de los tweets")
    parser.add_argument("--RT_column",default=cm.RT_column,required=False,help="Nombre columna que identifica si un tweets es original o no")
    parser.add_argument("--IdFrom_column",default=cm.IdFrom_column,required=False,help="Nombre columna que contiene la información del id del tweet original en caso que sea retweet")
    parser.add_argument("--followers_column",default=cm.followers_column,required=False,help="Nombre columna con el numero de seguidores de la cuenta que publica un tweet")
    parser.add_argument("--win_size_for_partition_cov",default=1,required=False,help="Tamaño en horas de la ventana temporal evaluación covariados")
    parser.add_argument("--followers_rate",default=2,required=False,help="parametro estabilizador numero de seguidores")
    parser.add_argument("--win_size_infectious_rate",default=3,required=False,help="Tamaño en horas de la ventana temporal estimacion intensidades")
    parser.add_argument("--win_size_train_period",default=1,required=False,help="Tamaño en horas de las ventanas de tiempo en el perido de entrenamiento")
    parser.add_argument("--win_size_pred_period",default=1,required=False,help="Tamaño en horas de las ventanas de tiempo en el perido de predicción")
    parser.add_argument("--method_pred",default="integral",required=False,choices=["integral","thinning"],help="Nombre metodo de predicción")



    args = parser.parse_args()
    subprocess=args.subprocess
    log_file = args.log_file
    save_model = args.save_model
    predict_period = args.predict_period
    save_result_predict= args.save_result_predict
    f_limite = args.f_limite
    exist_model_path = args.exist_model_path
    valid_period = args.valid_period
    path_classify_model = args.path_classify_model
    keywords_path = args.keywords_path
    vectors_path = args.vectors_path
    cmodel_path = args.cmodel_path
    predict_column = args.predict_column
    data_to_train_clasify = args.data_to_train_clasify
    percent_val_data = args.percent_val_data
    path_quantify_model = args.path_quantify_model
    qmodel_path = args.qmodel_path
    score_column = args.score_column
    text_column = args.text_column
    date_column = args.date_column

    TweetId_column = args.TweetId_column
    RT_column = args.RT_column
    IdFrom_column = args.IdFrom_column
    followers_column = args.followers_column
    win_size_for_partition_cov = args.win_size_for_partition_cov
    followers_rate = args.followers_rate
    win_size_infectious_rate = args.win_size_infectious_rate
    win_size_train_period = args.win_size_train_period
    win_size_pred_period = args.win_size_pred_period
    method_pred = args.method_pred
    

    try:
        process(log_file,
                save_model,
                subprocess,
                f_limite,
                predict_period,
                save_result_predict,
                exist_model_path,
                valid_period,
                path_classify_model,
                keywords_path,
                vectors_path,
                cmodel_path,
                predict_column,
                data_to_train_clasify,
                percent_val_data,
                path_quantify_model,
                qmodel_path,
                score_column,
                text_column,
                TweetId_column,
                RT_column,
                IdFrom_column,
                date_column,              
                followers_column,
                win_size_for_partition_cov,
                followers_rate,
                win_size_infectious_rate,
                win_size_train_period,
                win_size_pred_period,
                method_pred
                )
    except Exception as e:
        print(e)

    
