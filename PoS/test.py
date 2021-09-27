import sys
from general_class import *

def test(data_file="entradas/data_tweets.pickle",
         train_period=('2019-04-01 00:00', '2019-05-01 00:00'),
         valid_period=('2019-05-01 00:00', '2019-05-10 00:00'),
         data_cov=['entradas/Partidos.csv','entradas/Manifestaciones.csv','entradas/FechasEspeciales.csv']
        ):
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
                         (pd.Timestamp(t.date()) == f_especiales).sum(),
                         1
                        ])

    M=modelPercepcion("data",
                      train_period,
                      valid_period,
                      tweets_model_base=modelTweets,
                      f_covariates=(TC,restore_date),
                      win_size_for_partition_cov=1,
                      followers_rate=2,
                      win_size_infectious_rate = 3,
                      win_size_train_period = 1,
                      win_size_pred_period = 1,
                      method_pred = 'integral')
    with open(data_file, 'rb') as handle:
        data = pickle.load(handle)                  

    M.tweets_model = M.tweets_model_base(data,
                                           M.train_period,
                                           M.validate_period,
                                           f_covariates=M.f_covariates,
                                           win_size_for_partition_cov=M.win_size_for_partition_cov,
                                           followers_rate=M.followers_rate,
                                           win_size_infectious_rate = M.win_size_infectious_rate,
                                           win_size_train_period = M.win_size_train_period,
                                           win_size_pred_period = M.win_size_pred_period,
                                           method_pred = M.method_pred)
    Beta, param_infectious_fit = M.train_model()
    print(Beta, param_infectious_fit)
    T_pred=M.predict_model()
    print(T_pred)
    Errors=M.validation_model()
    print(Errors["MAE"],Errors["Pearson"])

if __name__ == "__main__":
    if len(sys.argv) == 1:
        test()
    elif len(sys.argv) == 2:
        data_file=sys.argv[1]
        test(data_file)
    elif len(sys.argv) == 3:
        data_file=sys.argv[1]
        train_period = tuple(sys.argv[2].split(','))
        print(train_period)
        test(data_file,train_period)
    elif len(sys.argv) == 4:
        data_file=sys.argv[1]
        train_period = tuple(sys.argv[2].split(','))
        valid_period = tuple(sys.argv[3].split(','))
        print(train_period,valid_period)
        test(data_file,train_period,valid_period)
    
    
    
    
