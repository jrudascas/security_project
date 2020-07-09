import unittest
import pandas as pd
import pickle

# Can't use from... import directly since the file is into another folder
import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from services.prediction_experiment import PredictionExperiment
from services.process_data import ProcessData
from services import prediction_metrics

class TestCase(unittest.TestCase):

    def SetUp(self):
        pass

    def test_hit_rate_default(self):
        pass

    def test_hit_rate_ground_truth(self):
        pass

    def test_make_counting_grid(self):
        """ Test for a base "well-known" scenario """
        ## Get grid prediction, to use size and region params
        infile = open('/Users/anamaria/Desktop/dev/security_project/aggressive_behavior_model/pkl/experiment_seppexp_10_2_siedco_prediction.pkl','rb')
        loaded_siedco = pickle.load(infile)
        infile.close()
        grid = loaded_siedco['prediction'].values[0]

        ## Select points to represent on counting matrix
        df = pd.read_csv("/Users/anamaria/Desktop/dev/security_project/datasets/deduplicate_siedco_09062020.csv")
        data = ProcessData("SIEDCO","/Users/anamaria/Desktop/dev/security_project/datasets/deduplicate_siedco_09062020.csv")
        df_input = data.add_timestamp(df)
        timed_pts,_ = ProcessData.get_time_space_points(df_input, data.dataset_dict)

        counting_matrix = prediction_metrics.make_counting_grid(grid, timed_pts)
        self.assertEqual(counting_matrix.xoffset, 958645.8182116301)
        self.assertEqual(counting_matrix.yoffset, 904338.0678953262)
        self.assertEqual(counting_matrix.xsize, 150)
        self.assertEqual(counting_matrix.ysize, 150)
        self.assertEqual(counting_matrix._matrix.shape, (816, 343))
        self.assertEqual(counting_matrix._matrix.max(),357)
        self.assertEqual(counting_matrix._matrix.min(),0)
