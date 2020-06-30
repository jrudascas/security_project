from datetime import datetime, timedelta

from services.process_data import ProcessData
from services.validate_model import ValidateModel

class PredictionExperiment:

    def __init__(self, dataset_info, custom_filter, train_dates, validation_dates, model, metrics, aggregation_data):
        # TODO: check if df is required as an instance attribute
        """
            :dataset: dictionary with db name, file path location and dataset dictionary
                    (e.g. {'name':'SIEDCO','path':'//','dict':{}})
            :custom_filter: dictionary with column db field_name and value to filter
                    (e.g. {'field':'LOCALIDAD','value':'CIUDAD BOLIVAR'})
            :train_dates: dictionary with initial and final training dates (YYYY-mm-dd)
                    (e.g. {'initial':'2018-01-01','final':'2018-01-07'})
            :model: class name from aggressive_model module (e.g. NaiveCounting)
            :metrics: dictionary with metrics names as keys and list as values
                    (e.g. {'hit-rate':[0.1],'PAI':[0.1]})
            :aggregation_data: string. Allowed values: "a priori", "subsequent"
        """
        self.dataset_info = dataset_info
        self.custom_filter = custom_filter
        self.train_dates = train_dates
        self.validation_dates = validation_dates
        self.model = model
        self.metrics = metrics
        self.aggregation_data = aggregation_data

    def check_exp_params(self):
        #check path exists
        #check initial date is previous to final date
        #check data available for validation/test (e.g., final training date is the last on the defined DB)
        pass

    def run_ncv_experiment(self, time_unit, grid_size, outer_iterations):
        """run nested-cross validation"""
        self.check_exp_params()
        data = ProcessData(self.dataset_info['name'], self.dataset_info['path'])
        df = data.get_formated_df()
        self.dataset_info['dict'] = data.dataset_dict #update dataset dictionary on experiment instance
        dates_interval = {'initial': self.train_dates['initial'], 'final': self.validation_dates['final']}
        df_train_validation = ProcessData.select_data(df, self.dataset_info['dict'], self.custom_filter, dates_interval)
        region = None
        validation = ValidateModel(df_train_validation, self.dataset_info['dict'], time_unit, outer_iterations, region)
        prediction_results = validation.walk_fwd_chain(self.model, grid_size, self.train_dates, self.validation_dates, self.metrics)
        return prediction_results

    def run_ncv_experiment_region(self, time_unit, grid_size, region):
        # TODO: refactor with previous method (run_ncv_experiment)
        """run nested-cross validation"""
        self.check_exp_params()
        data = ProcessData(self.dataset_info['name'], self.dataset_info['path'])
        df = data.get_formated_df()
        self.dataset_info['dict'] = data.dataset_dict #update dataset dictionary on experiment instance
        dates_interval = {'initial': self.train_dates['initial'], 'final': self.validation_dates['final']}
        df_train_validation = ProcessData.select_data(df, self.dataset_info['dict'], self.custom_filter, dates_interval)
        outer_iterations = ''
        validation = ValidateModel(df_train_validation, self.dataset_info['dict'], time_unit, outer_iterations, region)
        prediction_results = validation.walk_fwd_chain(self.model, grid_size, self.train_dates, self.validation_dates, self.metrics)
        return prediction_results

    def run_single_validation(self, grid_size, validation_date, time_unit, outer_iterations):
        data = ProcessData(self.dataset_info['name'], self.dataset_info['path'])
        df = data.get_formated_df()
        self.dataset_info['dict'] = data.dataset_dict #update dataset dictionary on experiment instance
        df_filtered = ProcessData.filter_by_field(df, self.custom_filter['field'], self.custom_filter['value'])
        validation = ValidateModel(df_filtered, self.dataset_info['dict'], time_unit, outer_iterations)
        prediction_results = validation.inner_loop_validation(self.model, grid_size, self.train_dates, datetime.strptime(validation_date,'%Y-%m-%d'), self.metrics)
        return prediction_results
