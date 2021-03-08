from modelo_tweets import *
import pickle
import itertools

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
    

# time series split
splits=[
        [('2019-04-01 00:00','2019-05-01 00:00'),
         ('2019-05-01 00:00','2019-05-15 00:00')],
        [('2019-04-01 00:00','2019-05-15 00:00'),
         ('2019-05-15 00:00','2019-06-01 00:00')],
        [('2019-04-01 00:00','2019-06-01 00:00'),
         ('2019-06-01 00:00','2019-06-15 00:00')],    
        [('2019-04-01 00:00','2019-06-15 00:00'),
         ('2019-06-15 00:00','2019-07-01 00:00')],    
        [('2019-04-01 00:00','2019-07-01 00:00'),
         ('2019-07-01 00:00','2019-07-15 00:00')],   
        [('2019-04-01 00:00','2019-07-15 00:00'),
         ('2019-07-15 00:00','2019-08-01 00:00')],     
        [('2019-04-01 00:00','2019-08-01 00:00'),
         ('2019-08-01 00:00','2019-08-15 00:00')],    
        [('2019-04-01 00:00','2019-08-15 00:00'),
         ('2019-08-15 00:00','2019-09-01 00:00')],   
        [('2019-04-01 00:00','2019-09-01 00:00'),
         ('2019-09-01 00:00','2019-09-15 00:00')],  
        [('2019-04-01 00:00','2019-09-15 00:00'),
         ('2019-09-15 00:00','2019-10-01 00:00')],    
        [('2019-04-01 00:00','2019-10-01 00:00'),
         ('2019-10-01 00:00','2019-10-15 00:00')],   
        [('2019-04-01 00:00','2019-10-15 00:00'),
         ('2019-10-15 00:00','2019-11-01 00:00')],    
        [('2019-04-01 00:00','2019-11-01 00:00'),
         ('2019-11-01 00:00','2019-11-15 00:00')],    
        [('2019-04-01 00:00','2019-11-15 00:00'),
         ('2019-11-15 00:00','2019-12-01 00:00')], 
        [('2019-04-01 00:00','2019-12-01 00:00'),
         ('2019-12-01 00:00','2019-12-15 00:00')],     
        [('2019-04-01 00:00','2019-12-15 00:00'),
         ('2019-12-15 00:00','2020-01-01 00:00')],     
        ]



results_by_split={}
for split in splits:
    train_period,validate_period=split
    print(train_period,validate_period)
    results={}
    results_cum={}
    for j in range(1,7):
        for i in list(itertools.combinations(np.arange(6),j)):
            print(i,np.array(i+(6,)))
            try:
                func = lambda x: TC(x)[np.array(i+(6,))]

                model=modelTweets(data,
                          train_period,
                          validate_period,
                          f_covariates=(func,restore_date),
                          followers_rate=1,
                          win_size_pred_period=1
                         )
                model.train()
                model.compute_lambda_predict()
                model.compute_errors()
                results[i]=model.errors_predict
                results_cum[i]=model.errors_predict_cum
            except:
                continue
    results_by_split[(train_period,validate_period)]=[results,results_cum]        
    with open('errors_covariados.pickle', 'wb') as handle:
        pickle.dump(results_by_split, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

        