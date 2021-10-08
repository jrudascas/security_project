from operator import index
from modelo_tweets import *
from classifier_class import *
from quantifier_class import *
from lexicon_analysis import *
import os.path
from utilis import get_estados_ejecucion, get_tipos_proceso, get_token_acces, update_process_state
from constants_manager import ESTADO_EXITO, ESTADO_ERROR, ESTADO_PROCESO, ESTADO_CANCELADO, NAME_PREDICCION,NAME_ENTRENAMIENTO
import logging
import dill
class modeloBase:
    
    def train_model(self):
        pass
       
    def predict_model(self):
        pass
    
    def validation_model(self):
        pass
        
class modelPercepcion(modeloBase):
    """
    Clase del modelo de percepcion de seguridad general juntando los procesos de clasificacion de tweets relacionados
    con seguridad, cuantificacion de los tweets clasificados y prediccion de la serie de tiempo en tiempo futuro.
    """    
    def __init__(self,
                 original_data,
                 classify_model,
                 quantify_model,
                 train_period,
                 val_period=None,
                 f_limite=None,
                 TweetId_column="TweetId",
                 RT_column="IsRetweet",
                 IdFrom_column="tweet_from",
                 date_column="CreatedAt",              
                 followers_column="followers",
                 score_column="score",         
                 tweets_model_base=modelTweets,
                 f_covariates=(T_c,restore_date),
                 win_size_for_partition_cov=1,
                 followers_rate=2,
                 win_size_infectious_rate = 3,
                 win_size_train_period = 1,
                 win_size_pred_period = 1,
                 method_pred = 'integral'
                ):
        #original_data
        logging.debug('Empezo inicialización modelo general de percepción de seguridad.')
        self.tipos_proceso=get_tipos_proceso(get_token_acces())
        self.estados_ejecucion=get_estados_ejecucion(get_token_acces())

        try:
            self.original_data=original_data
            self.RT_column=RT_column
            self.TweetId_column=TweetId_column
            self.date_column=date_column
            self.IdFrom_column=IdFrom_column
            self.followers_column=followers_column
            self.score_column=score_column
            
            # function to classify tweets related to security
            self.classify_model=classify_model
            try:
                if not os.path.isfile(self.classify_model.model_path):
                    self.classify_model.train_model()
            except Exception as e:
                raise Exception(e)
            # function to quantify tweets related to security
            self.quantify_model=quantify_model
            try:
                if not os.path.isfile(self.quantify_model.model_path):
                    self.quantify_model.train_model()
            except Exception as e:
                raise Exception(e)
            
            # variables proceso de hawkes    
            self.train_period = train_period
            self.validate_period = val_period
            self.f_limite = datetime.fromisoformat(f_limite)
            self.tweets_model_base = tweets_model_base
            self.f_covariates = f_covariates
            self.win_size_for_partition_cov=win_size_for_partition_cov
            self.followers_rate = followers_rate
            self.win_size_infectious_rate  = win_size_infectious_rate
            self.win_size_train_period  = win_size_train_period
            self.win_size_pred_period  = win_size_pred_period
            self.method_pred  = method_pred
            logging.debug('Termino inicialización modelo general de percepción de seguridad.')
        except Exception as e:
            raise Exception(e)
    
    def prepare_data(self,new_data=False):
        """
        Preprocesamiento de los datos y formateo para que queden listos para la entrada del modelo final
        :return: 
        """
        logging.debug('Empezo proceso preparación de los datos.')
        try:
            df=self.original_data.copy()
            df=df[df[self.date_column] >= self.f_limite]
            
            #tweets originales
            data_to_filter=df[df[self.RT_column]==0]
            
            # Rt cuyos tweets originales no estan
            
            if new_data == True:
                rt_not_found=df[(df[self.RT_column]==1) & ~(df[self.IdFrom_column].isin(data_to_filter[self.TweetId_column].values))].drop_duplicates(self.IdFrom_column)
                data_to_filter=pd.concat([data_to_filter,rt_not_found])


            pass_id=self.classify_model.clasify_df(data_to_filter)[self.TweetId_column].values
            
            tweets_score=self.quantify_model.quantify_df(df[df[self.TweetId_column].isin(pass_id)])
            

            Orig_id=tweets_score[tweets_score[self.RT_column] == 0][self.TweetId_column].values
            RT_id=tweets_score[tweets_score[self.RT_column] == 1][self.TweetId_column].values

            Inicio=self.f_limite

            info={}
            info['Inicio']=Inicio
            info['Tweets']={}

            for i in Orig_id:
                try:
                    subdata=df[(df[self.TweetId_column] == i)  | (df[self.IdFrom_column] ==i)]
                    subdata['times']=(subdata[self.date_column]-Inicio).dt.total_seconds()/3600
                    S=int(tweets_score[tweets_score[self.TweetId_column] == i][self.score_column])
                    info['Tweets'][str(i)]={'sentiment':S,
                                            'times':np.array(list(subdata.times.values)),
                                            'followers':np.array(list(subdata[self.followers_column].values))}
                except:
                    continue

            for i in RT_id:
                try:
                    subdata=df[df[self.IdFrom_column].isin(df[df[self.TweetId_column] ==i][self.TweetId_column])]
                    subdata['times']=(subdata[self.date_column]-Inicio).dt.total_seconds()/3600
                    S=int(tweets_score[tweets_score[self.TweetId_column] == i][self.score_column])
                    info['Tweets'][int(df[df[self.TweetId_column] == i]["tweet_from"])]={'sentiment':S,
                                                                                        'times':np.array(list(subdata.times.values)),
                                                                                        'followers':np.array(list(subdata[self.followers_column].values))}
                except:
                    continue                                                                                        

            logging.debug('Conversión tablas a diccionario como entrada modelo predicción tweets.')
            logging.debug('Termino proceso preparación de los datos.')
            return tweets_score,info
        except Exception as e:
            raise Exception(e)

    def create_model_tweets(self,data):
        logging.debug('Empezo proceso creación modelo de tweets.')
        try:
            if hasattr(self,"dict_data"):
                
                last=self.dict_data
                if last['Inicio'] < self.f_limite:
                    diff=(self.f_limite-last['Inicio']).dt.total_seconds()/3600
                    last['Inicio']=self.f_limite
                    to_drop=[]
                    for t in last['Tweets']:
                        if last['Tweets'][t]['times'][0]<diff:
                            to_drop.append(t)
                        else:
                            last['Tweets'][t]['times']=last['Tweets'][t]['times']-diff
                    for i in to_drop:
                        last["Tweets"].pop(i)

                logging.debug('Depuración tweets anteriores a la fecha limite.')   

                new={}         
                new['Inicio'] = last['Inicio']
                new['Tweets']={}
                diff=(new['Inicio']-data['Inicio']).dt.total_seconds()/3600
                to_drop_new=[]
                for t in last['Tweets']:
                    new['Tweets'][t] = last['Tweets'][t]
                    if t in data['Tweets']:
                        new['Tweets'][t]["times"]=np.unique(np.concatenate((new['Tweets'][t]["times"],diff+data['Tweets'][t]["times"])))
                        (new['Tweets'][t]["times"]).sort()
                        to_drop_new.append(t)
                for i in to_drop_new:
                    data["Tweets"].pop(i)

                for t in data["Tweets"]:
                    new['Tweets'][t] = data['Tweets'][t]
                    new['Tweets'][t]["times"]=new['Tweets'][t]["times"]+diff
                logging.debug('Adición tweets nuevos a los datos existentes.')   
            else:
                new=data

            self.tweets_model = self.tweets_model_base(new,
                                                    self.train_period,
                                                    self.validate_period,
                                                    f_covariates=self.f_covariates,
                                                    win_size_for_partition_cov=self.win_size_for_partition_cov,
                                                    followers_rate=self.followers_rate,
                                                    win_size_infectious_rate = self.win_size_infectious_rate,
                                                    win_size_train_period = self.win_size_train_period,
                                                    win_size_pred_period = self.win_size_pred_period,
                                                    method_pred = self.method_pred)
            logging.debug('Objeto modelo tweets creado.')
            self.dict_data=new
            return self.tweets_model
        except Exception as e:
            msg_error = "No se completo la creación del objeto modelo tweets."
            logging.error(msg_error)
            raise Exception(msg_error + " / " +str(e))
        
    def train_model(self):
        """
        Calcula los parametros ajustados del modelo de series de tiempo
        :return: (Beta,param_infectious_fit)
        """
        logging.debug('Empezo entrenamiento modelo percepción de seguridad.')

        update_process_state(self.tipos_proceso[NAME_ENTRENAMIENTO], self.estados_ejecucion[ESTADO_PROCESO], get_token_acces())

        try:
            self.tweets_model.train()
            logging.debug('Termino entrenamiento modelo percepción de seguridad.')
            update_process_state(self.tipos_proceso[NAME_ENTRENAMIENTO], self.estados_ejecucion[ESTADO_EXITO], get_token_acces())
            return self.tweets_model.Beta, self.tweets_model.param_infectious_fit
            
        except Exception as e:
            update_process_state(self.tipos_proceso[NAME_ENTRENAMIENTO], self.estados_ejecucion[ESTADO_ERROR], get_token_acces())
            msg_error = "No se completo entrenamiento modelo percepción de seguridad."
            logging.error(msg_error)
            raise Exception(msg_error + " / " +str(e))
        

        
    def predict_model(self,dates,save_path=None):
        """
        Calcula los errores de prediccion de la serie de tiempo para el tiempo predecido
        :return: (Beta,param_infectious_fit)
        """
        logging.debug('Empezo predicción modelo percepción de seguridad.')
        update_process_state(self.tipos_proceso[NAME_PREDICCION], self.estados_ejecucion[ESTADO_PROCESO], get_token_acces())

        try:
            self.tweets_model.compute_lambda_predict(dates)
            logging.debug('Termino predicción modelo percepción de seguridad.')
            logging.debug('Guardando resultado de predicción.')
            if save_path != None:
                try:
                    self.tweets_model.Tweets_pred.to_csv(save_path,index=False)
                    logging.debug('Guardado exitosamente resultado de predicción.')
                    update_process_state(self.tipos_proceso[NAME_PREDICCION], self.estados_ejecucion[ESTADO_EXITO], get_token_acces())

                except:
                    logging.debug('No se pudo guardar el resultado de predicción.')
                    update_process_state(self.tipos_proceso[NAME_PREDICCION], self.estados_ejecucion[ESTADO_ERROR], get_token_acces())
  
            return self.tweets_model.Tweets_pred
        except Exception as e:
            update_process_state(self.tipos_proceso[NAME_PREDICCION], self.estados_ejecucion[ESTADO_ERROR], get_token_acces())
            msg_error = "No se completo la predicción del modelo percepción de seguridad."
            logging.error(msg_error)
            raise Exception(msg_error + " / " +str(e))
    
    def validation_model(self):
        """
        Calcula las metricas de error prediccion de serie de tiempo para el tiempo predecido
        :return: diccionario diferentes metricas calculadas
        """
        logging.debug('Empezo proceso de validación.') 
        try:
            if self.validate_period == None:
                raise Exception("No se ha establecido el periodo de validación")
            else:
                self.tweets_model.compute_errors()
                return self.tweets_model.errors_predict
        except Exception as e:
            msg_error="No se pudo terminar el proceso de validación de la predicción"
            logging.error(msg_error)
            raise Exception(msg_error + " / " +str(e))
        
    def save_model(self,path):
        """
        Funcion para guardar el objeto de la clase como archivo pkl
        """
        try:
            with open(path, 'wb') as outp:
                dill.dump(self, outp, pickle.HIGHEST_PROTOCOL)
            logging.debug('Modelo de percepción de seguridad guardado exitosamente.')
        except:
            msg_error="No se pudo guardar el modelo de percepcion de seguridad con la direccion establecida"
            logging.error(msg_error)
            raise Exception(msg_error)
        
