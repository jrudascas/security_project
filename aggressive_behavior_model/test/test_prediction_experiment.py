import unittest
import pandas as pd


# Can't use from... import directly since the file is into another folder
import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from services.prediction_experiment import PredictionExperiment
from services.aggressive_model import KDEModel

class TestCase(unittest.TestCase):

    def setUp(self):
        self.my_experiment = PredictionExperiment()

    def test_set_up(self):
        siedco_dict = {
                        'date': 'FECHA_HECHO',
                        'latitude': 'LATITUD_Y',
                        'longitude': 'LONGITUD_X',
                        'time': 'HORA_HECHO',
                        'time_stamp':'',
                       }
        self.assertEqual(self.my_experiment.dataset,{'name':'SIEDCO','path':'','data_dict':siedco_dict})
        self.assertEqual(self.my_experiment.train_set, {'initial_date':'2018-01-01','final_date':'2018-01-07'})
        self.assertEqual(self.my_experiment.metrics, {'hit-rate':[0.1],'PAI':[0.1]})
        self.assertEqual(self.my_experiment.validation, {'nested cross-validation':'walk-forward chaining','time_unit':'days'})
        self.assertEqual(type(self.my_experiment.model), type(KDEModel()))
        self.assertEqual(self.my_experiment.aggregation_data, 'subsequent')

    def test_select_train_data_case1(self):
        #case 1: file not found
        head_path = '/Users/anamaria/Desktop/dev/security_project/datasets/'
        file = 'deduplicate_siedco_10032020'
        self.my_experiment.dataset['path'] = head_path + file
        response = self.my_experiment.select_train_data()
        self.assertEqual(response, "File not found, check path and file name")

    def test_select_train_data_case2(self):
        #case 2: file found
        head_path = '/Users/anamaria/Desktop/dev/security_project/datasets/'
        file = 'deduplicate_siedco_10032020.csv'
        self.my_experiment.dataset['path'] = head_path + file
        response = self.my_experiment.select_train_data()
        self.assertEqual(str(type(response)), "<class 'pandas.core.frame.DataFrame'>")

    def test_filter_by_date_case1(self):
        #case 1: date on interval, siedco
        head_path = '/Users/anamaria/Desktop/dev/security_project/datasets/'
        file = 'deduplicate_siedco_10032020.csv'
        df = pd.read_csv(head_path + file)
        df = self.my_experiment.add_timestamp(df)
        initial_date = '2018-01-01'
        final_date = '2018-01-01'
        current_dict = self.my_experiment.dataset['data_dict']
        df_filtered = PredictionExperiment.filter_by_date(df,current_dict,initial_date,final_date)
        df_expected = df.loc[df['FECHA_HECHO'] == "2018-01-01"]
        self.assertEqual(len(df_filtered),len(df_expected))

    def test_filter_by_date_case2(self):
        #case 2: initial date out of available data, siedco
        head_path = '/Users/anamaria/Desktop/dev/security_project/datasets/'
        file = 'deduplicate_siedco_10032020.csv'
        df = pd.read_csv(head_path + file)
        df = self.my_experiment.add_timestamp(df)
        initial_date = '2021-01-01'
        final_date = '2021-01-02'
        current_dict = self.my_experiment.dataset['data_dict']
        df_filtered = PredictionExperiment.filter_by_date(df,current_dict,initial_date,final_date)
        self.assertEqual(len(df_filtered),0)

    def test_filter_by_date_case3(self):
        #case 3: date on interval, nuse
        self.my_experiment.dataset['name'] = 'NUSE'
        self.my_experiment.dataset['data_dict'] = self.my_experiment.set_dictionary()
        head_path = '/Users/anamaria/Desktop/dev/security_project/datasets/'
        file = 'verify_enrich_nuse_29112019.csv'
        df = pd.read_csv(head_path + file)
        df = self.my_experiment.add_timestamp(df)
        initial_date = '2018-01-01'
        final_date = '2018-01-01'
        current_dict = self.my_experiment.dataset['data_dict']
        df_filtered = PredictionExperiment.filter_by_date(df,current_dict,initial_date,final_date)
        df_expected = df.loc[df['FECHA'] == "2018-01-01"]
        self.assertEqual(len(df_filtered),len(df_expected))
