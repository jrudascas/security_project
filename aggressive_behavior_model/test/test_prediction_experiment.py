import unittest
import pandas as pd


# Can't use from... import directly since the file is into another folder
import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from services.prediction_experiment import PredictionExperiment

DATASET_PATH = '/Users/anamaria/Desktop/dev/security_project/datasets/deduplicate_siedco_10032020.csv'

class TestCase(unittest.TestCase):

    def setUp(self):
        dataset_info = {'name':'SIEDCO','path':DATASET_PATH}
        custom_filter = {'field':'LOCALIDAD','value':'CIUDAD BOLIVAR'}
        train_dates = {'initial':'2018-01-01','final':'2018-01-07'}
        validation_dates = {'initial':'2018-01-08','final':'2018-01-15'}
        model = "NaiveCounting"
        metrics = {'hit-rate':[0.1],'PAI':[0.1]}
        aggregation_data = "subsequent"
        self.my_experiment = PredictionExperiment(dataset_info, custom_filter, train_dates, validation_dates, model, metrics, aggregation_data)

    def test_set_up(self):
        self.assertEqual(self.my_experiment.dataset_info,{'name':'SIEDCO','path': DATASET_PATH})
        self.assertEqual(self.my_experiment.custom_filter,{'field':'LOCALIDAD','value':'CIUDAD BOLIVAR'})
        self.assertEqual(self.my_experiment.train_dates, {'initial':'2018-01-01','final':'2018-01-07'})
        self.assertEqual(self.my_experiment.validation_dates, {'initial':'2018-01-08','final':'2018-01-15'})
        self.assertEqual(self.my_experiment.model, "NaiveCounting")
        self.assertEqual(self.my_experiment.metrics, {'hit-rate':[0.1],'PAI':[0.1]})
        self.assertEqual(self.my_experiment.aggregation_data, 'subsequent')

    def test_run_ncv(self):
        self.my_experiment.run_ncv_experiment('',150,1)
