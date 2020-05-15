import unittest
import pandas as pd

# Can't use from... import directly since the file is into another folder
import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from services.validate_model import ValidateModel

class TestCase(unittest.TestCase):

    def setUp(self):
        train_set = None
        initial_train_date = '2018-01-01'
        time_unit = 'days'
        outer_iterations = 6
        self.my_validation = ValidateModel(train_set, initial_train_date, time_unit, outer_iterations)

    def test_set_up(self):
        self.assertEqual(self.my_validation.train_set, None)
        self.assertEqual(self.my_validation.initial_train_date, '2018-01-01')
        self.assertEqual(self.my_validation.time_unit, 'days')
        self.assertEqual(self.my_validation.outer_iterations, 6)
