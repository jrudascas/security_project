import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle

def build_dict_info_tweets(sentiment_data,users_data,rt_data,tweets_predict):
    '''
    Funcion que permite crear un diccionario con la información de los tweets en las bases
    de datos y almacenarlos en un solo documento.
    
    entradas: 
    
        sentiment_data: dataframe de pandas con la información del sentimiento de los tweets
        users_data: dataframe de pandas con la información del numero de seguidores de cada usuario
        rt_data: dataframe de pandas con la información de que retweet corresponde a que tweet original
        tweets_predict: dataframe de pandas con la información de los tweets pasados por el algoritmo de clasificación 
        
    ########
    
    Salida:
        info: diccionario con la informacion centralizada de los tweets.
    
    
    
    '''
    SEN=tweets_predict[['TweetId','FullText']].merge(sentiment_data,left_on='FullText',right_on='Tweet')[['TweetId','SentimentPolarity']]
    tweets_predict['CreatedAt']=pd.to_datetime(tweets_predict['CreatedAt'])-timedelta(hours=5)
    tweets_predict.sort_values('CreatedAt',inplace=True)
    tweets_predict=tweets_predict.merge(rt_data).merge(users_data,left_on='TweetId',right_on='id_tweet')
    tweets_predict=tweets_predict[tweets_predict.tweet_from != -1].reset_index(drop=True)
    Inicio=tweets_predict.CreatedAt.min()
    tweets_predict=tweets_predict.merge(SEN,how='left')
    
    info={}
    info['Inicio']=Inicio
    info['Tweets']={}
    
    contador=0
    for idx,i in enumerate(tweets_predict[tweets_predict.tweet_from == 0].TweetId.values):
        subdata=pd.concat([tweets_predict[tweets_predict.TweetId == i],tweets_predict[tweets_predict.tweet_from == i]])[['CreatedAt','followers']]
        subdata['times']=(subdata.CreatedAt-pd.to_datetime(Inicio)).dt.total_seconds()/(3600)
        try:
            S=int(tweets_predict[tweets_predict.TweetId == i].SentimentPolarity.values[0])
            info['Tweets'][str(i)]={'sentiment':S,
                                    'times':np.array(list(subdata.times.values)),
                                    'followers':np.array(list(subdata.followers.values))}
        except:
            contador+=1
#     with open('data_tweets.pickle', 'wb') as handle:
#         pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return info