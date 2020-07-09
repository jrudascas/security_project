import unittest
import pandas as pd
import pickle

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
        self.my_experiment.validation_dates = {'initial':'2018-01-08','final':'2018-01-08'}
        prediction_results = self.my_experiment.run_ncv_experiment(time_unit='',grid_size=150, region=None)
        grid_region = prediction_results[0][2].region()
        self.assertEqual(len(prediction_results),4)

    def test_run_ncv_with_region(self):
        """ Test based on city scenario and predefined region"""
        infile = open('/Users/anamaria/Desktop/dev/security_project/aggressive_behavior_model/pkl/experiment_seppexp_10_2_siedco_prediction.pkl','rb')
        loaded_siedco = pickle.load(infile)
        infile.close()
        grid = loaded_siedco['prediction'].values[0]
        my_region = grid.region()

        csv_path = '/Users/anamaria/Desktop/dev/security_project/datasets/deduplicate_siedco_09062020.csv'
        siedco_info = {'name':'SIEDCO','path':csv_path}
        train_dates = {'initial':'2018-09-01','final':'2018-09-30'}
        validation_dates = {'initial':'2018-10-01','final':'2018-10-01'}
        model = "SEPPexp"
        metrics = ''
        aggregation = ''
        filter_localidad = {'field':'','value':''}
        localidad_experiment = PredictionExperiment(dataset_info=siedco_info, custom_filter=filter_localidad,train_dates=train_dates, validation_dates=validation_dates, model=model,metrics='',aggregation_data='')
        prediction_results = localidad_experiment.run_ncv_experiment(time_unit='',grid_size=150, region=my_region)
        grid_region = prediction_results[0][2].region()
        self.assertEqual(len(prediction_results),4)
        self.assertEqual(grid_region, my_region)

    def test_run_ncv_without_region(self):
        """ Test based on city scenario and 'None' region"""
        my_region = None

        csv_path = '/Users/anamaria/Desktop/dev/security_project/datasets/deduplicate_siedco_09062020.csv'
        siedco_info = {'name':'SIEDCO','path':csv_path}
        train_dates = {'initial':'2018-09-01','final':'2018-09-30'}
        validation_dates = {'initial':'2018-10-01','final':'2018-10-01'}
        model = "SEPPexp"
        metrics = ''
        aggregation = ''
        filter_localidad = {'field':'','value':''}
        localidad_experiment = PredictionExperiment(dataset_info=siedco_info, custom_filter=filter_localidad,train_dates=train_dates, validation_dates=validation_dates, model=model,metrics='',aggregation_data='')
        prediction_results = localidad_experiment.run_ncv_experiment(time_unit='',grid_size=150, region=None)
        grid_region = prediction_results[0][2].region()
        self.assertEqual(len(prediction_results),4)

        infile = open('/Users/anamaria/Desktop/dev/security_project/aggressive_behavior_model/pkl/experiment_seppexp_10_2_siedco_prediction.pkl','rb')
        loaded_siedco = pickle.load(infile)
        infile.close()
        grid = loaded_siedco['prediction'].values[0]
        my_region = grid.region()
        self.assertNotEqual(grid_region, my_region)
