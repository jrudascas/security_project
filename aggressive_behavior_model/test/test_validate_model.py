import unittest
import pandas as pd
import pickle

# Can't use from... import directly since the file is into another folder
import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from services.validate_model import ValidateModel

class TestCase(unittest.TestCase):

    def setUp(self):
        df_train = None
        dataset_dict = None
        time_unit = 'days'
        outer_iterations = 6
        self.my_validation = ValidateModel(df_train, dataset_dict, time_unit, outer_iterations)

    def test_set_up(self):
        self.assertEqual(self.my_validation.df_train_validation, None)
        self.assertEqual(self.my_validation.dataset_dict, None)
        self.assertEqual(self.my_validation.time_unit, 'days')
        self.assertEqual(self.my_validation.outer_iterations, 6)

    def test_walk_fwd_chain(self):
        output_path = '/Users/anamaria/Desktop/dev/security_project/aggressive_behavior_model/pkl/'
        file_name = 'df_train'
        infile = open(output_path+file_name+'.pkl','rb')
        df_train_validation = pickle.load(infile)
        infile.close()
        self.my_validation.df_train_validation = df_train_validation
        siedco_dict = {
                        'date': 'FECHA_HECHO',
                        'latitude': 'LATITUD_Y',
                        'longitude': 'LONGITUD_X',
                        'time': 'HORA_HECHO',
                        'time_stamp':'TIME_STAMP'
                       }
        self.my_validation.dataset_dict = siedco_dict
        self.my_validation.walk_fwd_chain(model_name="NaiveCounting", grid_size=150, train_dates_base={'initial':'2018-01-01','final':'2018-01-05'},validation_dates={'initial':'2018-01-06','final':'2018-01-07'},metrics={'hit-rate':[0.1],'PAI':[0.1]})
