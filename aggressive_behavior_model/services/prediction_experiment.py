import pandas as pd
from datetime import datetime, timedelta

from services.validate_model import ValidateModel

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

DATASET_DICT = {'SIEDCO':siedco_dict, 'NUSE':nuse_dict, 'RNMC':rnmc_dict}

class PredictionExperiment:

    def __init__(self, dataset, filter, train_dates, model, metrics, aggregation_data):
        """
            :dataset: dictionary with db name and file path location
                    (e.g. {'name':'SIEDCO','path':'//'})
            :filter: dictionary with column db field_name and value to filter
                    (e.g. {'field':'LOCALIDAD','value':'CIUDAD BOLIVAR'})
            :train_dates: dictionary with initial and final training dates (YYYY-mm-dd)
                    (e.g. {'initial_date':'2018-01-01','final_date':'2018-01-07'})
            :model: class name from aggressive_model module (e.g. NaiveCounting)
            :metrics: dictionary with metrics names as keys and list as values
                    (e.g. {'hit-rate':[0.1],'PAI':[0.1]})
            :aggregation_data: string. Allowed values: "a priori", "subsequent"
        """
        self.dataset = dataset
        self.filter = filter
        self.train_dates = train_dates
        self.model = model
        self.metrics = metrics
        self.aggregation_data = aggregation_data
        self.dataset['data_dict'] = self.set_dictionary()

    def set_dictionary(self):
        return DATASET_DICT[self.dataset['name']].copy()

    def run_ncv_experiment(self, time_unit, grid_size, outer_iterations):
        """run nested-cross validation"""
        self.check_exp_params()
        df_train = self.select_train_data()
        validation = ValidateModel(df_train, self.dataset['data_dict'], time_unit, outer_iterations)
        performance_results = validation.walk_fwd_chain(self.model, grid_size, self.validation_dates, self.metrics)
        return performance_results

    def check_exp_params(self):
        #check path exists
        #check initial date is previous to final date
        #check data available for validation/test (e.g., final training date is the last on the defined DB)
        pass

    def select_train_data(self):
        #db is already filtered with the type of event we want to model
        current_path = self.dataset['path']
        dataset_dict = self.dataset['data_dict']
        try:
            df = pd.read_csv(current_path)
        except FileNotFoundError:
            return "File not found, check path and file name"

        df = self.add_timestamp(df)
        df_filtered = PredictionExperiment.filter_by_field(df, self.filter['field'], self.filter['value'])
        df_train = PredictionExperiment.filter_by_date(df_filtered, dataset_dict, self.train_dates['initial_date'], self.train_dates['final_date'])
        return df_train

    def add_timestamp(self, df):
        # TODO: check if format is applied before (time-stamp with an specific format should be required before start an experiment)
        dataset_dict = self.dataset['data_dict']
        if dataset_dict['time_stamp'] == '':
            df = self.format_date_fields(df)
            df['TIME_STAMP']=pd.to_datetime(df[dataset_dict['date']]+ ' '+df[dataset_dict['time']])
            self.dataset['data_dict']['time_stamp'] = 'TIME_STAMP'
        return df

    def format_date_fields(self, df):
        # TODO: check if format is applied before (time-stamp with an specific format should be required before start an experiment)
        dataset_dict = self.dataset['data_dict']
        #format date field to standard
        df[dataset_dict['date']] = pd.to_datetime(df[dataset_dict['date']]).dt.strftime('%Y-%m-%d')
        #format time field to standard
        time_series = df[dataset_dict['time']]
        match_format_flag = False
        regex_standard_hour = r'^\d{2}:\d{2}:\d{2}$'
        regex_short_hour = r'^\d{2}:\d{2}$'
        regex_int_hour = r'^\d{1,4}$'
        if time_series.astype(str).str.match(regex_standard_hour).all():
            match_format_flag = True
            df.loc[df[dataset_dict['time']] == '00:00:00',dataset_dict['time']] = '00:00:01'
        if time_series.astype(str).str.match(regex_short_hour).all():
            match_format_flag = True
            df.loc[df[dataset_dict['time']] == '00:00',dataset_dict['time']] = '00:01'
        if time_series.astype(str).str.match(regex_int_hour).all():
            match_format_flag = True
            df[dataset_dict['time']] = pd.to_datetime(df[dataset_dict['time']].astype(str).str.rjust(4,'0'),format= '%H%M').dt.strftime('%H:%M')
            df.loc[df[dataset_dict['time']] == '00:00',dataset_dict['time']] = '00:01'

        if match_format_flag == False:
            return "Date/time format error, check type and structure on date/time columns"
        else:
            return df

    def filter_by_date(df, dataset_dict, initial_date, final_date):
        # TODO: check if we create a miscelanuos class to manipulate df
        time_stamp_field = dataset_dict['time_stamp']
        initial_date = datetime.strptime(initial_date,'%Y-%m-%d')
        real_final_date = datetime.strptime(final_date,'%Y-%m-%d')+timedelta(days=1)
        df_filtered = df[(df[time_stamp_field] > initial_date) & (df[time_stamp_field] < real_final_date)]
        return df_filtered

    def filter_by_field(df, name_field, value):
        # TODO: check if we create a miscelanuos class to manipulate df
        try:
            df_filtered = df[df[name_field] == value]
        except KeyError:
            return "Field doesn't exist"
        if len(df_filtered) == 0:
            return "Empty filter result, check filter value"
        return df_filtered
