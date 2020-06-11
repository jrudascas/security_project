import unittest
import pandas as pd

# Can't use from... import directly since the file is into another folder
import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from services.process_data import ProcessData

siedco_dict = {
                'date': 'FECHA_HECHO',
                'latitude': 'LATITUD_Y',
                'longitude': 'LONGITUD_X',
                'time': 'HORA_HECHO',
                'time_stamp':''
               }

class TestCase(unittest.TestCase):

    def setUp(self):
        dataset_name = 'SIEDCO'
        head_path = '/Users/anamaria/Desktop/dev/security_project/datasets/'
        file = 'deduplicate_siedco_10032020.csv'
        dataset_path = head_path + file
        self.my_data = ProcessData(dataset_name, dataset_path)

    def test_set_up(self):
        self.assertEqual(self.my_data.dataset_name,'SIEDCO')
        self.assertEqual(self.my_data.dataset_path,'/Users/anamaria/Desktop/dev/security_project/datasets/deduplicate_siedco_10032020.csv')
        self.assertEqual(self.my_data.dataset_dict,siedco_dict)

    def test_filter_by_date_case1(self):
        #case 1: date on interval, siedco
        df = pd.read_csv(self.my_data.dataset_path)
        df = self.my_data.add_timestamp(df)
        initial_date = '2018-01-01'
        final_date = '2018-01-01'
        dataset_dict = self.my_data.dataset_dict
        df_filtered = ProcessData.filter_by_date(df,dataset_dict,initial_date,final_date)
        df_expected = df.loc[df['FECHA_HECHO'] == "2018-01-01"]
        self.assertEqual(len(df_filtered),len(df_expected))

    def test_filter_by_date_case2(self):
        #case 2: initial date out of available data, siedco
        df = pd.read_csv(self.my_data.dataset_path)
        df = self.my_data.add_timestamp(df)
        initial_date = '2021-01-01'
        final_date = '2021-01-02'
        dataset_dict = self.my_data.dataset_dict
        self.assertWarns(UserWarning, lambda: ProcessData.filter_by_date(df,dataset_dict,initial_date,final_date))

    def test_filter_by_date_case3(self):
        #case 3: date on interval, nuse sample
        dataset = {'name':'NUSE','path':''}
        self.my_data.dataset_name = 'NUSE'
        self.my_data.dataset_dict = self.my_data.set_dictionary()
        head_path = '/Users/anamaria/Desktop/dev/security_project/datasets/'
        file = 'verify_enrich_nuse_29112019.csv'
        df = pd.read_csv(head_path + file)
        df = self.my_data.add_timestamp(df)
        initial_date = '2018-01-01'
        final_date = '2018-01-01'
        dataset_dict = self.my_data.dataset_dict
        df_filtered = ProcessData.filter_by_date(df,dataset_dict,initial_date,final_date)
        df_expected = df.loc[df['FECHA'] == "2018-01-01"]
        self.assertEqual(len(df_filtered),len(df_expected))

    def test_filter_by_date_case4(self):
        #case 4: date on interval, nuse full data
        dataset = {'name':'NUSE','path':''}
        self.my_data.dataset_name = 'NUSE'
        self.my_data.dataset_dict = self.my_data.set_dictionary()
        head_path = '/Users/anamaria/Desktop/dev/security_project/datasets/'
        file = 'verify_enrich_nuse_29112019.csv'
        df = pd.read_csv(head_path + file)
        df = self.my_data.add_timestamp(df)
        initial_date = '2018-01-01'
        final_date = '2018-01-01'
        dataset_dict = self.my_data.dataset_dict
        df_filtered = ProcessData.filter_by_date(df,dataset_dict,initial_date,final_date)
        df_expected = df.loc[df['FECHA'] == "2018-01-01"]
        self.assertEqual(len(df_filtered),len(df_expected))

    def test_filter_by_date_case5(self):
        #case 5: date on interval, rnmc
        dataset = {'name':'RNMC','path':''}
        self.my_data.dataset_name = 'RNMC'
        self.my_data.dataset_dict = self.my_data.set_dictionary()
        head_path = '/Users/anamaria/Desktop/dev/security_project/datasets/'
        file = '06. verify_enrich_rnmc_12022020.csv'
        df = pd.read_csv(head_path + file)
        df = self.my_data.add_timestamp(df)
        initial_date = '2018-01-01'
        final_date = '2018-01-01'
        dataset_dict = self.my_data.dataset_dict
        df_filtered = ProcessData.filter_by_date(df,dataset_dict,initial_date,final_date)
        df_expected = df.loc[df['FECHA'] == "2018-01-01"]
        self.assertEqual(len(df_filtered),len(df_expected))

    def test_filter_by_field_case1(self):
        #case 1: filter successful
        df = pd.read_csv(self.my_data.dataset_path)
        df_filtered = ProcessData.filter_by_field(df, 'LOCALIDAD', 'BOSA')
        self.assertEqual(df_filtered.LOCALIDAD.unique()[0],'BOSA')

    def test_filter_by_field_case2(self):
        #case 2: filter successful, without field value
        df = pd.read_csv(self.my_data.dataset_path)
        df_filtered = ProcessData.filter_by_field(df, '', '')
        assertion_proxy = df_filtered.equals(df)
        self.assertEqual(assertion_proxy, True)

    def test_filter_by_field_case3(self):
        #case 3: error, field doesn't exist
        df = pd.read_csv(self.my_data.dataset_path)
        self.assertRaises(ValueError, lambda: ProcessData.filter_by_field(df, 'nombre', 'Pedro'))

    def test_filter_by_field_case4(self):
        #case 4: error, value doesn't exist
        df = pd.read_csv(self.my_data.dataset_path)
        self.assertRaises(ValueError, lambda: ProcessData.filter_by_field(df, 'LOCALIDAD', 'NORMANDIA'))

    # def test_select_data_case1(self):
    #     #case 1: file not found
    #     head_path = '/Users/anamaria/Desktop/dev/security_project/datasets/'
    #     file = 'deduplicate_siedco_10032020'
    #     self.my_data.dataset_info['path'] = head_path + file
    #     data = ProcessData(self.my_data.dataset_info['name'], self.my_data.dataset_info['path'])
    #     self.assertRaises(FileNotFoundError, lambda: data.load_dataset_from_csv())
    #
    # def test_select_data_case2(self):
    #     #case 2: file found
    #     head_path = '/Users/anamaria/Desktop/dev/security_project/datasets/'
    #     file = 'deduplicate_siedco_10032020.csv'
    #     self.my_data.dataset_info['path'] = head_path + file
    #     data = ProcessData(self.my_data.dataset_info['name'], self.my_data.dataset_info['path'])
    #     df = data.get_formated_df()
    #     self.my_data.dataset_info['dict'] = data.dataset_dict
    #     response = self.my_data.select_train_data(df)
    #     self.assertEqual(str(type(response)), "<class 'pandas.core.frame.DataFrame'>")
    #
    # def test_save_element(self):
    #     head_path = '/Users/anamaria/Desktop/dev/security_project/datasets/'
    #     file = 'deduplicate_siedco_10032020.csv'
    #     self.my_data.dataset_info['path'] = head_path + file
    #     df_train = self.my_data.select_train_data()
    #     output_path = '/Users/anamaria/Desktop/dev/security_project/aggressive_behavior_model/pkl/'
    #     file_name = 'df_train'
    #     ProcessData.save_element(output_path, file_name, df_train)
    #     try:
    #         response = open(output_path+'df_train.pkl','rb')
    #         response.close()
    #     except FileNotFoundError:
    #         response = "FileNotFoundError"
    #     self.assertNotEqual(response, "FileNotFoundError")
