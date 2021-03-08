from modelo_tweets import *
import pickle

partidos=pd.read_csv('entradas/Partidos.csv')
manifestaciones=pd.read_csv('entradas/Manifestaciones.csv')
f_especiales=pd.read_csv('entradas/FechasEspeciales.csv')

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
                     (pd.Timestamp(t.date()) == f_especiales).sum(),                     
                     1
                    ])
                    
with open('entradas/data_tweets.pickle', 'rb') as handle:
    data = pickle.load(handle)                 
    
results_by_split={}
for i in range(4,12):
    train_period=('2019-'+str(i).zfill(2)+'-01 00:00',
                  '2019-'+str(i+1).zfill(2)+'-01 00:00')
    results={}
    results_cum={}    
    for j in range(2,30,2):
        validate_period=('2019-'+str(i+1).zfill(2)+'-01 00:00',
                         '2019-'+str(i+1).zfill(2)+'-'+str(j).zfill(2)+' 00:00')
        print(train_period,validate_period)
        model=modelTweets(data,
                          train_period,
                          validate_period,
                          f_covariates=(TC,restore_date),
                          followers_rate=4,
                          win_size_pred_period=1
                         )
        model.train()
        model.compute_lambda_predict()
        model.compute_errors()
        results[j]=model.errors_predict
        results_cum[j]=model.errors_predict_cum
    results_by_split[train_period]=[results,results_cum]
    
        
with open('errors_times.pickle', 'wb') as handle:
    pickle.dump(results_by_split, handle, protocol=pickle.HIGHEST_PROTOCOL)