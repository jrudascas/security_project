from get_info_followers_and_times import *
from modelo_tweets import *
from classifier_class import *
from quantifier_class import *
from get_scores import *
import os.path

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
                 train_period,
                 val_period,
                 classify_model,
                 quantify_model,
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
        except:
            print("Error en el entrenamiento del modelo de clasificacion")
        # function to quantify tweets related to security
        self.quantify_model=quantify_model
        if self.quantify_model == None:
            pass
            # TODO: funcion para entrenar modelo de clasificacion textos relacionados seguridad
            
        # variables proceso de hawkes    
        self.train_period = train_period
        self.validate_period = val_period
        self.tweets_model_base = tweets_model_base
        self.f_covariates = f_covariates
        self.win_size_for_partition_cov=win_size_for_partition_cov
        self.followers_rate = followers_rate
        self.win_size_infectious_rate  = win_size_infectious_rate
        self.win_size_train_period  = win_size_train_period
        self.win_size_pred_period  = win_size_pred_period
        self.method_pred  = method_pred
    
    def prepare_data(self):
        """
        Preprocesamiento de los datos y formateo para que queden listos para la entrada del modelo final
        :return: 
        """
        self.original_data.drop_duplicates(self.TweetId_column,inplace=True)
        Orig_id=self.original_data[self.original_data[self.RT_column] == 0][self.TweetId_column].values        
        pass_id=self.classify_model.clasify_df(self.original_data[self.original_data[self.RT_column] == 0])[self.TweetId_column].values
        
        tweets_score=self.quantify_model.quantify_df(self.original_data[self.original_data[self.TweetId_column].isin(pass_id)])
        
        tweets_predict=pd.merge(self.original_data[self.original_data[self.TweetId_column].isin(pass_id) | self.original_data[self.IdFrom_column].isin(pass_id)],tweets_score,how='left')
                 
        tweets_predict[self.date_column]=pd.to_datetime(tweets_predict[self.date_column])-timedelta(hours=6)
        tweets_predict.sort_values(self.date_column,inplace=True)
        Inicio=tweets_predict[self.date_column].min()


        info={}
        info['Inicio']=Inicio
        info['Tweets']={}

        contador=0
        for idx,i in enumerate(tweets_predict[tweets_predict[self.RT_column] == 0][self.TweetId_column].values):
            subdata=pd.concat([tweets_predict[tweets_predict[self.TweetId_column] == i],tweets_predict[tweets_predict[self.IdFrom_column] == i]])[[self.date_column,self.followers_column]]
            subdata['times']=(subdata[self.date_column]-pd.to_datetime(Inicio)).dt.total_seconds()/(3600)
            try:
                S=int(tweets_predict[tweets_predict[self.TweetId_column] == i][self.score_column])
                info['Tweets'][str(i)]={'sentiment':S,
                                        'times':np.array(list(subdata.times.values)),
                                        'followers':np.array(list(subdata[self.followers_column].values))}
            except:
                contador+=1
        
        self.pre_process_data=tweets_score
        self.dict_events=info
        
        #sentiment_data=self.quantify_model(pd.read_csv("entradas/ScoreTotalData.csv"))#sentiment_data
        # base de datos usuarios y sus seguidores
        #users_data=pd.read_csv("results/users.csv")
        #base de datos reconocimiento de tweets y RT
        #rt_data=pd.read_csv("results/tweets_from.csv")
        #tweets=build_dict_info_tweets(sentiment_data,users_data,rt_data,tweets_predict)
        
        self.tweets_model = self.tweets_model_base(info,
                                                   self.train_period,
                                                   self.validate_period,
                                                   f_covariates=self.f_covariates,
                                                   win_size_for_partition_cov=self.win_size_for_partition_cov,
                                                   followers_rate=self.followers_rate,
                                                   win_size_infectious_rate = self.win_size_infectious_rate,
                                                   win_size_train_period = self.win_size_train_period,
                                                   win_size_pred_period = self.win_size_pred_period,
                                                   method_pred = self.method_pred)
        return self.tweets_model
        
    def train_model(self):
        """
        Calcula los parametros ajustados del modelo de series de tiempo
        :return: (Beta,param_infectious_fit)
        """
        self.tweets_model.train()
        return self.tweets_model.Beta, self.tweets_model.param_infectious_fit
        
    def predict_model(self):
        """
        Calcula los errores de prediccion de la serie de tiempo para el tiempo predecido
        :return: (Beta,param_infectious_fit)
        """
        self.tweets_model.compute_lambda_predict()
        return self.tweets_model.Tweets_pred
    
    def validation_model(self):
        """
        Calcula las metricas de error prediccion de serie de tiempo para el tiempo predecido
        :return: diccionario diferentes metricas calculadas
        """
        self.tweets_model.compute_errors()
        return self.tweets_model.errors_predict
        