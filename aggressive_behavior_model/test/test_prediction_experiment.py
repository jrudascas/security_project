import unittest
import pandas as pd


# Can't use from... import directly since the file is into another folder
import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from services.prediction_experiment import PredictionExperiment

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
        self.assertEqual(self.my_experiment.model, {'type':'KDE','parameters_estimation':'custom'})
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
        self.assertNotEqual(response, "File not found, check path and file name")

    def test_filter_by_date_case1(self):
        #case 1: date on interval, siedco
        head_path = '/Users/anamaria/Desktop/dev/security_project/datasets/'
        file = 'deduplicate_siedco_10032020.csv'
        df = pd.read_csv(head_path + file)
        df = self.my_experiment.add_timestamp(df)
        initial_date = self.my_experiment.train_set['initial_date']
        final_date = self.my_experiment.train_set['final_date']
        df_filtered = self.my_experiment.filter_by_date(df,initial_date,final_date)

        mask = (df['TIME_STAMP'] > initial_date) & (df['TIME_STAMP'] <= final_date)
        df_expected = df.loc[mask]
        print(df_expected.TIME_STAMP)
        print(df_filtered.TIME_STAMP)

        self.assertEqual(df_filtered,df_expected)
