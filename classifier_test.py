from classifier_class import *
from quantifier_class import *
from general_class import *

def main():
    import logging
    logging.basicConfig(filename='example.log',level=logging.DEBUG)
    # your program code
    try:

        data=pd.read_csv("../extra/predict_socialmedia_info.csv")[["TweetId","IsRetweet","CreatedAt","Text","user","followers","tweet_from"]].head(100)
        #train_period=('2019-04-01 00:00', '2019-05-01 00:00')
        #val_period=('2019-05-01 00:00', '2019-05-10 00:00')
        #tweets_model_base=modelTweets

        classify_model=modelClassification(keywords_path="entradas/19032020_Palabras_Filtro.xls",
                                           vectors_path="../extra/words_vectors.vec",
                                           text_column="Text",
                                           #data_to_train=Train,
                                           #predict_column="seguridad",                                            
                                           model_path="entradas/SVM_class.joblib")

        quantify_model=modelQuantification(text_column="Text",model_path="entradas/senticon.es.xml")

        data.drop_duplicates("TweetId",inplace=True)
        Orig_id=data[data["IsRetweet"] == 0]["TweetId"].values 
        pass_id=classify_model.clasify_df(data[data["TweetId"].isin(Orig_id)])["TweetId"].values
        tweets_score=quantify_model.quantify_df(data[data["TweetId"].isin(pass_id)])
        tweets_predict=pd.merge(data[data["TweetId"].isin(pass_id) | data["tweet_from"].isin(pass_id)],tweets_score,how='left')

        tweets_predict["CreatedAt"]=pd.to_datetime(tweets_predict["CreatedAt"])-timedelta(hours=6)
        tweets_predict.sort_values("CreatedAt",inplace=True)
        Inicio=tweets_predict["CreatedAt"].min()

        info={}
        info['Inicio']=Inicio
        info['Tweets']={}

        contador=0
        for idx,i in enumerate(tweets_predict[tweets_predict["IsRetweet"] == 0]["TweetId"].values):
            subdata=pd.concat([tweets_predict[tweets_predict["TweetId"] == i],tweets_predict[tweets_predict["tweet_from"] == i]])[["CreatedAt","followers"]]
            subdata['times']=(subdata["CreatedAt"]-pd.to_datetime(Inicio)).dt.total_seconds()/(3600)
            try:
                S=int(tweets_predict[tweets_predict["TweetId"] == i]["score"])
                info['Tweets'][str(i)]={'sentiment':S,
                                        'times':np.array(list(subdata.times.values)),
                                        'followers':np.array(list(subdata["followers"].values))}
            except:
                contador+=1
        print(contador)

        return info
    except Exception as e:
        raise Exception(e)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)