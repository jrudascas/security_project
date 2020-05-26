import unittest
import pandas as pd


# Can't use from... import directly since the file is into another folder
import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from services.prediction_experiment import PredictionExperiment

class TestCase(unittest.TestCase):

    def setUp(self):
        dataset_info = {'name':'SIEDCO','path':'//'}
        custom_filter = {'field':'LOCALIDAD','value':'CIUDAD BOLIVAR'}
        train_dates = {'initial':'2018-01-01','final':'2018-01-07'}
        model = "NaiveCounting"
        metrics = {'hit-rate':[0.1],'PAI':[0.1]}
        aggregation_data = "subsequent"
        self.my_experiment = PredictionExperiment(dataset_info, custom_filter, train_dates, model, metrics, aggregation_data)

    def test_set_up(self):
        self.assertEqual(self.my_experiment.dataset_info,{'name':'SIEDCO','path':'//'})
        self.assertEqual(self.my_experiment.custom_filter,{'field':'LOCALIDAD','value':'CIUDAD BOLIVAR'})
        self.assertEqual(self.my_experiment.train_dates, {'initial':'2018-01-01','final':'2018-01-07'})
        self.assertEqual(self.my_experiment.model, "NaiveCounting")
        self.assertEqual(self.my_experiment.metrics, {'hit-rate':[0.1],'PAI':[0.1]})
        self.assertEqual(self.my_experiment.aggregation_data, 'subsequent')
