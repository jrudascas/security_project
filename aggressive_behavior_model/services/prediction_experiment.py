import pandas as pd

SIEDCO_DICT = {
                'date': '',
                'latitude': 'LATITUD_Y',
                'longitude': 'LONGITUD_X',
                'time': 'HORA',
                'time_stamp':'FECHA_HECHO',
               }

NUSE_DICT = {
                'date': 'FECHA',
                'latitude': 'LATITUD',
                'longitude': 'LONGITUD',
                'time': 'HORA',
                'time_stamp':'',
               }

class predictionExperiment:

    def __init__(self):
        self.db = {'name':'SIEDCO','path':''}
        self.train_set = {'initial_date':'','final_date':''}
        self.metrics = {'hit-rate':[0.1],'PAI':[0.1]}
        self.validation = {'nested cross-validation':'walk-forward chaining','time_unit':'days'}
        self.model = {'type':'KDE','parameters_estimation':'custom'}
        self.aggregation_data = "subsequent"

    def validate_exp_params(self):
        #check path exists
        #check train set is available for defined db
        #check data available for validation/test (e.g., final training date is the last on the defined DB)
        pass

    def select_db_data(self):
        #db is already filtered with the type of event we want to model
        current_path = self.db['']
        pass
