import pandas as pd
from datetime import datetime, timedelta

from services.aggressive_model import KDEModel

siedco_dict = {
                'date': 'FECHA_HECHO',
                'latitude': 'LATITUD_Y',
                'longitude': 'LONGITUD_X',
                'time': 'HORA_HECHO',
                'time_stamp':''
               }

nuse_dict = {
                'date': 'FECHA',
                'latitude': 'LATITUD',
                'longitude': 'LONGITUD',
                'time': 'HORA',
                'time_stamp':''
            }

rnmc_dict = {
                'date': 'FECHA',
                'latitude': 'LATITUD',
                'longitude': 'LONGITUD',
                'time': 'HORA',
                'time_stamp':''
            }

DATASET_DICT = {'SIEDCO':siedco_dict, 'NUSE':nuse_dict}

class PredictionExperiment:

    def __init__(self):
        self.dataset = {'name':'SIEDCO','path':''}
        self.train_set = {'initial_date':'2018-01-01','final_date':'2018-01-07'}
        self.metrics = {'hit-rate':[0.1],'PAI':[0.1]}
        self.validation = {'nested cross-validation':'walk-forward chaining','time_unit':'days'}
        self.model = KDEModel()
        self.aggregation_data = "subsequent"

        self.dataset['data_dict'] = self.set_dictionary()

    def set_dictionary(self):
        return DATASET_DICT[self.dataset['name']].copy()

    def run_experiment(self):
        self.check_exp_params()
        df_train = self.select_train_data()
        #iterate over train+test (nested cross-validation)
        return performace_metrics

    def check_exp_params(self):
        #check path exists
        #check initial date is previous to final date
        #check data available for validation/test (e.g., final training date is the last on the defined DB)
        pass

    def select_train_data(self):
        #db is already filtered with the type of event we want to model
        current_path = self.dataset['path']
        current_dict = self.dataset['data_dict']
        try:
            df = pd.read_csv(current_path)
        except FileNotFoundError:
            return "File not found, check path and file name"
        df = self.add_timestamp(df)
        df_filtered = PredictionExperiment.filter_by_date(df, current_dict, self.train_set['initial_date'], self.train_set['final_date'])
        return df_filtered

    def add_timestamp(self, df):
        # TODO: check if format is applied before (time-stamp with an specific format should be required before start an experiment)
        current_dict = self.dataset['data_dict']
        if current_dict['time_stamp'] == '':
            df = self.format_date_fields(df)
            #df[current_dict['date']] = pd.to_datetime(df[current_dict['date']], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')
            df['TIME_STAMP']=pd.to_datetime(df[current_dict['date']]+ ' '+df[current_dict['time']])
            self.dataset['data_dict']['time_stamp'] = 'TIME_STAMP'
        return df

    def format_date_fields(self, df):
        # TODO: check if format is applied before (time-stamp with an specific format should be required before start an experiment)
        current_dict = self.dataset['data_dict']
        #format date field to standard
        df[current_dict['date']] = pd.to_datetime(df[current_dict['date']]).dt.strftime('%Y-%m-%d')
        #format time field to standard
        time_series = df[current_dict['time']]
        match_format_flag = False
        regex_standard_hour = r'^\d{2}:\d{2}:\d{2}$'
        regex_short_hour = r'^\d{2}:\d{2}$'
        regex_int_hour = r'^\d{1,4}$'
        if time_series.astype(str).str.match(regex_standard_hour).all():
            match_format_flag = True
            df.loc[df[current_dict['time']] == '00:00:00',current_dict['time']] = '00:00:01'
        if time_series.astype(str).str.match(regex_short_hour).all():
            match_format_flag = True
            df.loc[df[current_dict['time']] == '00:00',current_dict['time']] = '00:01'
        if time_series.astype(str).str.match(regex_int_hour).all():
            match_format_flag = True
            df[current_dict['time']] = pd.to_datetime(df[current_dict['time']].astype(str).str.rjust(4,'0'),format= '%H%M').dt.strftime('%H:%M')
            df.loc[df[current_dict['time']] == '00:00',current_dict['time']] = '00:01'

        if match_format_flag == False:
            return "Date/time format error, check type and structure on date/time columns"
        else:
            return df

    def filter_by_date(df, current_dict, initial_date, final_date):
        # TODO: check if we create a miscelanuos class to manipulate df
        time_stamp_field = current_dict['time_stamp']
        initial_date = datetime.strptime(initial_date,'%Y-%m-%d')
        real_final_date = datetime.strptime(final_date,'%Y-%m-%d')+timedelta(days=1)
        df_filtered = df[(df[time_stamp_field] > initial_date) & (df[time_stamp_field] < real_final_date)]
        return df_filtered
