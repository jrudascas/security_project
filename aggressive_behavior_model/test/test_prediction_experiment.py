import unittest

# Can't use from... import directly since the file is into another folder
import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from services.prediction_experiment import predictionExperiment

class TestCase(unittest.TestCase):

    def setUp(self):
        self.my_experiment = predictionExperiment()

    def test_set_up(self):
        self.my_experiment = predictionExperiment()
        self.assertEqual(self.my_experiment.db,{'name':'SIEDCO','path':''})
        self.assertEqual(self.my_experiment.train_set, {'initial_date':'','final_date':''})
        self.assertEqual(self.my_experiment.metrics, {'hit-rate':[0.1],'PAI':[0.1]})
        self.assertEqual(self.my_experiment.validation, {'nested cross-validation':'walk-forward chaining','time_unit':'days'})
        self.assertEqual(self.my_experiment.model, {'type':'KDE','parameters_estimation':'custom'})
        self.assertEqual(self.my_experiment.aggregation_data, 'subsequent')
