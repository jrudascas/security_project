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
    return np.array([(pd.Timestamp(t.date()) == partidos_M).sum(),
                     (pd.Timestamp(t.date()) == partidos_S).sum(),
                     (pd.Timestamp(t.date()) == manifestaciones).sum(),
                     (pd.Timestamp(t.date()) == f_especiales).sum(),
                     t.weekday()/6.0,(t.hour > 12)*1,1
                    ])
                    
with open('entradas/data_tweets.pickle', 'rb') as handle:
    data = pickle.load(handle)                 
    
train_period=('2019-05-01 00:00','2019-05-30 00:00')

times_to_val=[('2019-06-01 00:00','2019-06-02 00:00'),
              ('2019-06-01 00:00','2019-06-04 00:00'),
              ('2019-06-01 00:00','2019-06-06 00:00'),
              ('2019-06-01 00:00','2019-06-08 00:00'),
              ('2019-06-01 00:00','2019-06-10 00:00'),
              ('2019-06-01 00:00','2019-06-12 00:00'),
              ('2019-06-01 00:00','2019-06-14 00:00'),
              ('2019-06-01 00:00','2019-06-16 00:00'),
              ('2019-06-01 00:00','2019-06-18 00:00'),
              ('2019-06-01 00:00','2019-06-20 00:00'),
              ('2019-06-01 00:00','2019-06-22 00:00'),
              ('2019-06-01 00:00','2019-06-24 00:00'),
              ('2019-06-01 00:00','2019-06-26 00:00'),
              ('2019-06-01 00:00','2019-06-28 00:00')   
]

results={}
results_cum={}
for num,validate_period in enumerate(times_to_val):
    print((2*num+1))
    print(validate_period)
    model=modelTweets(data,
                      train_period,
                      validate_period,
                      f_covariates=(TC,restore_date),
                      followers_rate=7,
                      win_size_pred_period=1
                     )
    model.train()
    model.compute_lambda_predict()
    model.compute_errors()
    results[2*num+1]=model.errors_predict
    results_cum[2*num+1]=model.errors_predict_cum
    
        
with open('errors_times.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('errors_times_cov.pickle', 'wb') as handle:
    pickle.dump(results_cum, handle, protocol=pickle.HIGHEST_PROTOCOL)