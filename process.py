from operator import ge
import sys

from general_class import *
import logging
import argparse
import constants_manager as cm
from utilis import get_data_from_postgress, spark_to_pandas

def info_data(data,column_date):
    data_=data.copy()
    data_[column_date]=pd.to_datetime(data_[column_date])-timedelta(hours=6)
    min_,max_=data_[column_date].min(),data_[column_date].max()
    return str(min_),str(max_)

def TC(t,data_cov):
    return np.array([
                        t.weekday()/6.0,
                        (t.hour > 12)*1,
                        pd.Timestamp(t.date()) in data_cov[(data_cov[cm.COVARIATE_COLUMN] == 'Partido') & (data_cov[cm.PARTICULAR_COV_COLUMN] == 'Millonarios')][cm.DATE_COLUMN_COV].values,
                        pd.Timestamp(t.date()) in data_cov[(data_cov[cm.COVARIATE_COLUMN] == 'Partido') & (data_cov[cm.PARTICULAR_COV_COLUMN] == 'Santa Fé')][cm.DATE_COLUMN_COV].values,
                        pd.Timestamp(t.date()) in data_cov[data_cov[cm.COVARIATE_COLUMN] == 'Manifestaciones'][cm.DATE_COLUMN_COV].values,
                        #pd.Timestamp(t.date()) in data_cov[data_cov[cm.COVARIATE_COLUMN] == 'Festividad'][cm.DATE_COLUMN_COV].values,
                        1
                    ])


def update_covariados(data_cov):
    return lambda t:TC(t,data_cov)


def process(log_file,
            summary_file,
            save_model,
            subprocess,
            f_limite=None,
            save_freq_palabras=None,
            save_real_scores=None,
            predict_period=None,
            save_result_predict=None,
            exist_model_path=None,
            valid_period=None,
            path_classify_model=None,
            keywords_path=None,
            vectors_path=None,
            cmodel_path=None,
            predict_column=None,
            data_to_train_clasify=None,
            percent_val_data=0.3,
            path_quantify_model=None,
            qmodel_path=None,
            score_column=cm.SCORE_COLUMN,
            text_column=cm.TEXT_COLUMN,
            TweetId_column=cm.TWEETID_COLUMN,
            RT_column=cm.RT_COLUMN,
            IdFrom_column=cm.IDFROM_COLUMNN,
            date_column=cm.DATE_COLUMN,              
            followers_column=cm.FOLLOWERS_COLUMN,
            win_size_for_partition_cov=1,
            followers_rate=2,
            win_size_infectious_rate = 3,
            win_size_train_period = 1,
            win_size_pred_period = 1,
            method_pred = 'integral'
            ):
    """
    Función principal ejecución modelo percepción de seguridad
    """  
    logging.basicConfig(filename=log_file,level=logging.DEBUG)
    logging.debug('Empezo función process.')

    try:
        locals_=locals()
        summary = open(summary_file,"w")
        summary.write("Ejecución proceso modelo Percepción de Seguridad \n")
        summary.write("Fecha: "+str(datetime.now())+"\n")
        summary.write("Variables de entrada: \n")
        for i in locals_:
            if locals_[i] != None:
                summary.write(str(i)+": "+str(locals_[i])+"\n")
        
        firsttime=False



        if exist_model_path == None:
            firsttime=True
            logging.debug('Verificación archivos para crear modelos.')
            ## modelo clasificación de tweets relacionados con seguridad
            if not os.path.isfile(path_classify_model):
                logging.debug('No se encontro modelo clasificación tweets existente con la dirección establecida.')
                try:
                    data_to_train_clasify=pd.read_csv(data_to_train_clasify)
                    summary.write("Cantidad textos para entrenar modelo: "+str(len(data_to_train_clasify)) +"\n")
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
            data=spark_to_pandas(get_data_from_postgress(),summary=summary)
            train_period=info_data(data,date_column)
            summary.write("Periodo de entrenamiento: " +str(train_period) + "\n")
            if f_limite == None:
                f_limite = train_period[0]

            if valid_period != None:
                valid_period=tuple(valid_period.split(","))
            
            
            data_cov=spark_to_pandas(get_data_from_postgress(table=cm.DATABASE_COVARIADOS),start_data=f_limite,tipe='cov')
            

            generalmodel=modelPercepcion(classify_model,
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
                                         f_covariates=(update_covariados(data_cov),restore_date),
                                         win_size_for_partition_cov=win_size_for_partition_cov,
                                         followers_rate=followers_rate,
                                         win_size_infectious_rate = win_size_infectious_rate,
                                         win_size_train_period = win_size_train_period,
                                         win_size_pred_period = win_size_pred_period,
                                         method_pred = method_pred
                                        )
            tweets_score,info,palabras,real_info=generalmodel.prepare_data(data)
            try:
                palabras.to_csv(save_freq_palabras,index=False)
                logging.debug('Guardado exitosamente tabla frecuencia palabras.')
                tweets_score.to_csv(save_real_scores,index=False)
                base=os.path.basename(save_real_scores)
                real_info.to_csv(save_real_scores[:-len(base)]+'stats_'+base,index=False)             
                logging.debug('Guardado exitosamente tabla scores tweets reales.')
            except:
                logging.debug('No se pudo guardar tabla frecuencia palabras.')
                logging.debug('No se pudo guardar tabla scores tweets reales.')
            summary.write("Cantidad tweets originales que pasaron el clasificador: " +str(len(info['Tweets'])) + "\n")
            generalmodel.create_model_tweets(info)

        else:
            try:
                with open(exist_model_path, "rb") as input_file:
                        generalmodel = dill.load(input_file)

                #########################
                data_cov=spark_to_pandas(get_data_from_postgress(table=cm.DATABASE_COVARIADOS),start_data=f_limite,tipe='cov') 
                f_covariates=(update_covariados(data_cov),restore_date)
                generalmodel.f_covariates = f_covariates
                if hasattr(generalmodel,"tweets_model"):
                    generalmodel.tweets_model.f_covariates = lambda a : f_covariates[0](f_covariates[1](a,generalmodel.tweets_model.f_inicio))
                #######################
                logging.debug('Modelo percepción de seguridad existente cargado exitosamente.')
            except Exception:
                msg_error="No se encontro modelo existente con la dirección establecida."
                logging.error(msg_error)
                raise Exception(msg_error)

        if (subprocess == 'clean') & (firsttime==False) :
            last_train_period=generalmodel.tweets_model.train_period
            # nueva tanda de datos desde last_train_period[1]
            data=spark_to_pandas(get_data_from_postgress(),last_train_period[1],summary=summary)
            
            _,end=info_data(data,date_column)

            
            if f_limite == None:
                train_period=(last_train_period[0],end)
            else:
                generalmodel.f_limite = datetime.fromisoformat(str(f_limite))
                train_period=(f_limite,end)
            summary.write("Periodo de entrenamiento: " +str(train_period) + "\n")
            tweets_score,info,palabras,real_info=generalmodel.prepare_data(data,new_data=True)
            summary.write("Cantidad tweets originales que pasaron el clasificador: " +str(len(info['Tweets'])) + "\n")
            try:
                palabras.to_csv(save_freq_palabras,index=False)
                logging.debug('Guardado exitosamente tabla frecuencia palabras.')
                tweets_score.to_csv(save_real_scores,index=False)
                base=os.path.basename(save_real_scores)
                real_info.to_csv(save_real_scores[:-len(base)]+'stats_'+base,index=False) 
                logging.debug('Guardado exitosamente tabla scores tweets reales.')
            except:
                logging.debug('No se pudo guardar tabla frecuencia palabras.')
                logging.debug('No se pudo guardar tabla scores tweets reales.')
            generalmodel.create_model_tweets(info)

        if subprocess == "train":
            
            if generalmodel.win_size_for_partition_cov != win_size_for_partition_cov:
                generalmodel.win_size_for_partition_cov = win_size_for_partition_cov
                generalmodel.tweets_model.win_size_for_partition_cov = win_size_for_partition_cov

            if generalmodel.followers_rate != followers_rate:
                generalmodel.followers_rate = followers_rate
                generalmodel.tweets_model.followers_rate = followers_rate

            if generalmodel.win_size_infectious_rate != win_size_infectious_rate:
                generalmodel.win_size_infectious_rate = win_size_infectious_rate
                generalmodel.tweets_model.win_size_infectious_rate = win_size_infectious_rate

            if generalmodel.win_size_train_period != win_size_train_period:
                generalmodel.win_size_train_period = win_size_train_period
                generalmodel.tweets_model.win_size_train_period = win_size_train_period





            beta,error_Beta,param_infectious_fit,error_infectious=generalmodel.train_model()
            summary.write("Parametros modelo tweets: \nBeta:" +str(beta) +
                            "\nError en calculo Beta: " + str(round(error_Beta,3))+
                            "\nEcuación de intensidades: "+ str(param_infectious_fit)+ 
                            "\nError en calculo intensidades: "+str(round(error_infectious,3)) +"\n"
                            )

        if subprocess == "predict":

            if generalmodel.win_size_pred_period != win_size_pred_period:
                generalmodel.win_size_pred_period = win_size_pred_period
                generalmodel.tweets_model.win_size_pred_period = win_size_pred_period
            
            if generalmodel.method_pred != method_pred:
                generalmodel.method_pred = method_pred
                generalmodel.tweets_model.method_pred = method_pred
            
            

            if not (hasattr(generalmodel.tweets_model,"Beta") & hasattr(generalmodel.tweets_model,"param_infectious_fit")):
                logging.debug('El modelo debe ser entrenado primero.')
                generalmodel.train_model()
                summary.write("Parametros modelo tweets: \nBeta:" +str(beta) +
                              "\nError en calculo Beta: " + str(round(error_Beta,3))+
                              "\nEcuación de intensidades: "+ str(param_infectious_fit)+ 
                              "\nError en calculo intensidades: "+str(round(error_infectious,3)) +"\n"
                              )
            predict_period=tuple(predict_period.split(","))
            generalmodel.predict_model(predict_period,save_result_predict)

        if subprocess == "validate":
            if generalmodel.win_size_pred_period != win_size_pred_period:
                generalmodel.win_size_pred_period = win_size_pred_period
                generalmodel.tweets_model.win_size_pred_period = win_size_pred_period

            if generalmodel.method_pred != method_pred:
                generalmodel.method_pred = method_pred
                generalmodel.tweets_model.method_pred = method_pred

            if valid_period != None:
                valid_period=tuple(valid_period.split(","))
            if generalmodel.validate_period != valid_period:
                generalmodel.validate_period = valid_period
            if not (hasattr(generalmodel.tweets_model,"Beta") & hasattr(generalmodel.tweets_model,"param_infectious_fit")):
                logging.debug('El modelo debe ser entrenado primero.')
                generalmodel.train_model()
                summary.write("Parametros modelo tweets: \nBeta:" +str(beta) +
                              "\nError en calculo Beta: " + str(round(error_Beta,3))+
                              "\nEcuación de intensidades: "+ str(param_infectious_fit)+ 
                              "\nError en calculo intensidades: "+str(round(error_infectious,3)) +"\n"
                              )
            errors,errors_cum=generalmodel.validation_model()
            summary.write("Metricas de validación modelo tweets: \nMAE:" +str(errors["MAE"]) +"\nMAE acumulado:"+ str(errors_cum["MAE"])+ "\n")
        generalmodel.save_model(save_model)
        logging.debug('Termino función process.')
        summary.close()
        return generalmodel
    except Exception as e:
        msg_error = "No se completo función process"
        logging.error(msg_error)
        raise Exception(msg_error + " / " +str(e))
    


if __name__ == "__main__":

    def check_positive(value):
        try:
            value = float(value)
            if value <= 0:
                raise argparse.ArgumentTypeError("{} is not a positive number".format(value))
        except ValueError:
            raise argparse.ArgumentTypeError("{} is not an number".format(value))
        return value

    def check_max(value):
        try:
            value = check_positive(value)
            if value > 1:
                raise argparse.ArgumentTypeError("{} is greater than 1".format(value))
        except ValueError:
            raise argparse.ArgumentTypeError("{} is not an number".format(value))
        return value

    parser = argparse.ArgumentParser(description='Función ejecución proceso')
    parser.add_argument("--log_file",required=True,type=str,help="Dirección archivo de log")
    parser.add_argument("--summary_file",required=True,type=str,help="Dirección archivo resumen procesos")
    parser.add_argument("--save_model",required=True,type=str,help="Dirección donde se guardara el archivo del modelo resultante")
    parser.add_argument("--subprocess",required=True,type=str,help="Subproceso a ejecutar",default="clean",choices=["clean","train","predict","validate"])
    parser.add_argument("--save_freq_palabras",type=str,default=None,required=False,help="Dirección donde se guarda la tabla de frecuencias de las palabras de los tweets")
    parser.add_argument("--save_real_scores",type=str,default=None,required=False,help="Dirección donde se guarda la tabla de medida de percepción de seguridad de los tweets reales")
    parser.add_argument("--predict_period",type=str,default=None,required=False,help="Periodo de predicción")
    parser.add_argument("--save_result_predict",type=str,default=None,required=False,help="Dirección donde se guarda el resultado de predicción")
    
    parser.add_argument("--f_limite",type=str,default=None,required=False,help="Fecha limite")
    parser.add_argument("--exist_model_path",type=str,default=None,required=False,help="Dirección modelo previo existente")
    parser.add_argument("--valid_period",type=str,default=None,required=False,help="Perido de validación modelo")
    parser.add_argument("--path_classify_model",type=str,default=None,required=False,help="Dirección modelo clasificación de tweets")
    parser.add_argument("--keywords_path",type=str,default=None,required=False,help="Dirección keywords para modelo clasificación de tweets")
    parser.add_argument("--vectors_path",type=str,default=None,required=False,help="Dirección archivo vectores palabras para modelo clasificación de tweets")
    parser.add_argument("--cmodel_path",type=str,default=None,required=False,help="Dirección modelo clasificación base de tweets")
    parser.add_argument("--predict_column",type=str,default=None,required=False,help="Nombre columna objetivo entrenamiento modelo clasificación de tweets")
    parser.add_argument("--data_to_train_clasify",type=str,default=None,required=False,help="Dirección archivo csv para entrenar modelo clasificación de tweets")
    parser.add_argument("--percent_val_data",type=check_max,default=0.3,required=False,help="Porcentaje datos de validación para modelo clasificación de tweets")
    parser.add_argument("--path_quantify_model",type=str,default=None,required=False,help="Dirección modelo cuantificación de tweets")    
    parser.add_argument("--qmodel_path",type=str,default=None,required=False,help="Dirección modelo cuantificación base de tweets")
    parser.add_argument("--score_column",type=str,default="score",required=False,help="Nombre nueva columna resultante proceso cuantificación de tweets")
    parser.add_argument("--text_column",type=str,default=cm.TEXT_COLUMN,required=False,help="Nombre columna que contiene eltexto de los tweets")
    parser.add_argument("--date_column",type=str,default=cm.DATE_COLUMN,required=False,help="Nombre columna fechas en las que se publican los tweets")

    parser.add_argument("--TweetId_column",type=str,default=cm.TWEETID_COLUMN,required=False,help="Nombre columna que contiene los identificadores unicos de los tweets")
    parser.add_argument("--RT_column",type=str,default=cm.RT_COLUMN,required=False,help="Nombre columna que identifica si un tweets es original o no")
    parser.add_argument("--IdFrom_column",type=str,default=cm.IDFROM_COLUMNN,required=False,help="Nombre columna que contiene la información del id del tweet original en caso que sea retweet")
    parser.add_argument("--followers_column",type=str,default=cm.FOLLOWERS_COLUMN,required=False,help="Nombre columna con el numero de seguidores de la cuenta que publica un tweet")
    parser.add_argument("--win_size_for_partition_cov",type=check_positive,default=1,required=False,help="Tamaño en horas de la ventana temporal evaluación covariados")
    parser.add_argument("--followers_rate",type=check_positive,default=2,required=False,help="parametro estabilizador numero de seguidores")
    parser.add_argument("--win_size_infectious_rate",type=check_positive,default=3,required=False,help="Tamaño en horas de la ventana temporal estimacion intensidades")
    parser.add_argument("--win_size_train_period",type=check_positive,default=1,required=False,help="Tamaño en horas de las ventanas de tiempo en el perido de entrenamiento")
    parser.add_argument("--win_size_pred_period",type=check_positive,default=1,required=False,help="Tamaño en horas de las ventanas de tiempo en el perido de predicción")
    parser.add_argument("--method_pred",type=str,default="integral",required=False,choices=["integral","thinning"],help="Nombre metodo de predicción")



    args = parser.parse_args()
    subprocess=args.subprocess
    save_freq_palabras=args.save_freq_palabras
    save_real_scores=args.save_real_scores
    log_file = args.log_file
    summary_file = args.summary_file
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
                summary_file,
                save_model,
                subprocess,
                f_limite,
                save_freq_palabras,
                save_real_scores,
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
        raise Exception(e)
        #print(e)

    
