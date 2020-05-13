import pandas as pd
from datetime import datetime, timedelta

siedco_dict = {
                'date': 'FECHA_HECHO',
                'latitude': 'LATITUD_Y',
                'longitude': 'LONGITUD_X',
                'time': 'HORA_HECHO',
                'time_stamp':'',
               }

nuse_dict = {
                'date': 'FECHA',
                'latitude': 'LATITUD',
                'longitude': 'LONGITUD',
                'time': 'HORA',
                'time_stamp':'',
               }

DATASET_DICT = {'SIEDCO':siedco_dict, 'NUSE':nuse_dict}

class predictionExperiment:

    def __init__(self):
        self.dataset = {'name':'SIEDCO','path':''}
        self.train_set = {'initial_date':'2018-01-01','final_date':'2018-01-07'}
        self.metrics = {'hit-rate':[0.1],'PAI':[0.1]}
        self.validation = {'nested cross-validation':'walk-forward chaining','time_unit':'days'}
        self.model = {'type':'KDE','parameters_estimation':'custom'}
        self.aggregation_data = "subsequent"

        self.dataset['data_dict'] = self.set_dictionary()

    def set_dictionary(self):
        return DATASET_DICT[self.dataset['name']].copy()

    def validate_exp_params(self):
        #check path exists
        #check initial date is previous to final date
        #check data available for validation/test (e.g., final training date is the last on the defined DB)
        pass

    def select_train_data(self):
        #db is already filtered with the type of event we want to model
        current_path = self.dataset['path']
        try:
            df = pd.read_csv(current_path)
        except FileNotFoundError:
            return "File not found, check path and file name"
        df = self.add_timestamp(df)
        initial_date = self.train_set['initial_date']
        final_date = self.train_set['final_date']
        #print(df.columns)
        df_filtered = self.filter_by_date(df, initial_date, final_date)

        return "Successful"

    def add_timestamp(self, df):
        current_dict = self.dataset['data_dict']
        if current_dict['time_stamp'] == '':
            df[current_dict['date']] = pd.to_datetime(df[current_dict['date']], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')
            df['TIME_STAMP']=pd.to_datetime(df[current_dict['date']]+ ' '+df[current_dict['time']])
            self.dataset['data_dict']['time_stamp'] = 'TIME_STAMP'
        return df

    def filter_by_date(self, df, initial_date, final_date):
        current_dict = self.dataset['data_dict']
        time_stamp_field = current_dict['time_stamp']
        real_final_date = datetime.strptime(final_date,'%Y-%m-%d')+timedelta(days=1)
        df_filtered = df[(df[time_stamp_field] > datetime.strptime(initial_date,'%Y-%m-%d')) & (df[time_stamp_field] < real_final_date)]
        return df_filtered
