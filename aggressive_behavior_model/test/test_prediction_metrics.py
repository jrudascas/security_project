import unittest
import numpy as np
import open_cp
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

    def test_hit_rate_1(self):
        """ Test hit_rate=1 if all real events falls on hotspots """
        df = pd.read_csv("/Users/anamaria/Desktop/dev/security_project/datasets/deduplicate_siedco_09062020.csv")
        data = ProcessData("SIEDCO","/Users/anamaria/Desktop/dev/security_project/datasets/deduplicate_siedco_09062020.csv")
        df_input = data.add_timestamp(df)
        date = '2018-01-01'
        dataset_dict = data.dataset_dict
        df_filtered = ProcessData.filter_by_date(df_input,dataset_dict,date,date)

        timed_pts,region = ProcessData.get_time_space_points(df_filtered, data.dataset_dict)
        counting_kernel = open_cp.naive.CountingGridKernel(grid_width=150, region=region)
        counting_kernel.data = timed_pts
        grid_prediction = counting_kernel.predict()

        coverages = [2,4,6,8,10]
        hit_rates_default = prediction_metrics.measure_hit_rates(grid_prediction,timed_pts,coverages,'default')
        hit_rates_ground_truth = prediction_metrics.measure_hit_rates(grid_prediction,timed_pts,coverages,'ground_truth_coverage')
        self.assertEqual(hit_rates_default, {2: 1.0, 4: 1.0, 6: 1.0, 8: 1.0, 10: 1.0})
        self.assertEqual(hit_rates_ground_truth, {0.46187915216703573: 1.0})

    def test_hit_rate_2(self):
        """ Test hit_rate=0 if no events falls on hotspots """
        df = pd.read_csv("/Users/anamaria/Desktop/dev/security_project/datasets/deduplicate_siedco_09062020.csv")
        data = ProcessData("SIEDCO","/Users/anamaria/Desktop/dev/security_project/datasets/deduplicate_siedco_09062020.csv")
        df_input = data.add_timestamp(df)
        date = '2018-01-01'
        dataset_dict = data.dataset_dict
        df_input= ProcessData.filter_by_date(df_input,dataset_dict,date,date)
        df_1= ProcessData.filter_by_field(df_input,'LOCALIDAD','SUBA')
        df_2= ProcessData.filter_by_field(df_input,'LOCALIDAD','BOSA')

        timed_pts,region = ProcessData.get_time_space_points(df_1, data.dataset_dict)
        counting_kernel = open_cp.naive.CountingGridKernel(grid_width=150, region=region)
        counting_kernel.data = timed_pts
        grid_prediction = counting_kernel.predict()

        coverages = [2,4,6,8,10]
        eval_pts,_ = ProcessData.get_time_space_points(df_2, data.dataset_dict)
        hit_rates_default = prediction_metrics.measure_hit_rates(grid_prediction,eval_pts,coverages,'default')
        hit_rates_ground_truth = prediction_metrics.measure_hit_rates(grid_prediction,eval_pts,coverages,'ground_truth_coverage')
        self.assertEqual(hit_rates_default, {2: 0.0, 4: 0.0, 6: 0.0, 8: 0.0, 10: 0.0})
        self.assertEqual(hit_rates_ground_truth, {0.6632653061224489: 0.0})

    def test_hit_rate_3(self):
        """ Test hit rate based on Candelaria scenario """
        csv_path = '/Users/anamaria/Desktop/dev/security_project/datasets/deduplicate_siedco_09062020.csv'
        siedco_info = {'name':'SIEDCO','path':csv_path}
        train_dates = {'initial':'2018-03-01','final':'2018-09-30'}
        validation_dates = {'initial':'2018-10-03','final':'2018-10-03'}
        model = "SEPPexp"
        metrics = ''
        aggregation = ''
        filter_localidad = {'field':'LOCALIDAD','value':'CANDELARIA'}
        localidad_experiment = PredictionExperiment(dataset_info=siedco_info, custom_filter=filter_localidad,train_dates=train_dates, validation_dates=validation_dates, model=model,metrics='',aggregation_data='')
        prediction_results = localidad_experiment.run_ncv_experiment(time_unit='',grid_size=150, region=None)

        df_siedco = pd.DataFrame(prediction_results, columns =['initial-date','final-date','prediction','eval_pts'])
        coverages = [2,4,6,8,10]
        grid_prediction_1 = df_siedco.prediction.values[1]
        eval_pts_1 = df_siedco.eval_pts.values[1]
        grid_prediction_2 = df_siedco.prediction.values[2]
        eval_pts_2 = df_siedco.eval_pts.values[2]
        hit_rates_default_1 = prediction_metrics.measure_hit_rates(grid_prediction_1,eval_pts_1,coverages,'default')
        hit_rates_default_2 = prediction_metrics.measure_hit_rates(grid_prediction_2,eval_pts_2,coverages,'default')
        hit_rates_ground_truth_1 = prediction_metrics.measure_hit_rates(grid_prediction_1,eval_pts_1,coverages,'ground_truth_coverage')
        hit_rates_ground_truth_2 = prediction_metrics.measure_hit_rates(grid_prediction_2,eval_pts_2,coverages,'ground_truth_coverage')

        self.assertEqual(hit_rates_default_1, {2: -1.0, 4: -1.0, 6: -1.0, 8: -1.0, 10: -1.0})
        self.assertEqual(hit_rates_default_2, {2: 0.0, 4: 0.0, 6: 0.0, 8: 1.0, 10: 1.0})
        self.assertEqual(hit_rates_ground_truth_1, {2: -1.0, 4: -1.0, 6: -1.0, 8: -1.0, 10: -1.0})
        self.assertEqual(hit_rates_ground_truth_2, {0.7692307692307693: 0.0})

    def test_make_counting_grid(self):
        """ Test counting grid for a base "well-known" scenario """
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
        """ Test mse=0 if both matrices (prediction and ground truth) are equal """
        df = pd.read_csv("/Users/anamaria/Desktop/dev/security_project/datasets/deduplicate_siedco_09062020.csv")
        data = ProcessData("SIEDCO","/Users/anamaria/Desktop/dev/security_project/datasets/deduplicate_siedco_09062020.csv")
        df_input = data.add_timestamp(df)
        timed_pts,region = ProcessData.get_time_space_points(df_input, data.dataset_dict)

        counting_kernel = open_cp.naive.CountingGridKernel(grid_width=150, region=region)
        counting_kernel.data = timed_pts
        grid_prediction = counting_kernel.predict()
        mse = prediction_metrics.mse(grid_prediction,timed_pts)
        self.assertEqual(mse, 0)
