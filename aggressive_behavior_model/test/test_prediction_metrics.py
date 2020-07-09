import unittest
import numpy as np
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

    def test_hit_rate_default_1(self):
        """ Test hit_rate=1 if all real events falls on hotspots """
        pass

    def test_hit_rate_default_2(self):
        """ Test hit_rate=0 if no events falls on hotspots """
        pass

    def test_hit_rate_default_3(self):
        """ Test based on Candelaria scenario """
        pass

    def test_hit_rate_ground_truth_1(self):
        """ Test hit_rate=1 if all real events falls on hotspots """
        pass

    def test_hit_rate_ground_truth_2(self):
        """ Test hit_rate=0 if no events falls on hotspots """
        pass

    def test_hit_rate_ground_truth_3(self):
        """ Test based on Candelaria scenario """
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

    def test_mse_match(self):
        """ Test mse results match using different methods"""
        infile = open('/Users/anamaria/Desktop/dev/security_project/aggressive_behavior_model/pkl/experiment_seppexp_10_2_siedco_prediction.pkl','rb')
        loaded_siedco = pickle.load(infile)
        infile.close()
        grid = loaded_siedco['prediction'].values[0]
        grid._matrix = ProcessData.normalize_matrix(grid._matrix)
        real_events = loaded_siedco['eval_pts'].values[0]
        mse_method_1 = prediction_metrics.mse(grid,real_events)

        counting_matrix = prediction_metrics.make_counting_grid(grid, real_events)
        counting_matrix._matrix = ProcessData.normalize_matrix(counting_matrix._matrix)
        mse_method_2 = np.sum((grid._matrix.astype("float") - counting_matrix._matrix.astype("float")) ** 2)
        mse_method_2 /= float(grid._matrix.shape[0] * grid._matrix.shape[1])

        self.assertEqual(mse_method_1, mse_method_2)

    def test_mse_1(self):
        """ Test mse=0 if both matrices are equal """
        pass
