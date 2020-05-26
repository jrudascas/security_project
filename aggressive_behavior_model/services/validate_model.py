from datetime import datetime, timedelta
import numpy as np
import open_cp
from open_cp import evaluation

from services.process_data import ProcessData
from services.aggressive_model import NaiveCounting, SpaceTimeKDE

class ValidateModel:

    def __init__(self, df_train, dataset_dict, time_unit, outer_iterations):
        self.df_train = df_train
        self.dataset_dict = dataset_dict
        self.time_unit = time_unit
        self.outer_iterations = outer_iterations

    def check_validation_params(self):
        #check number of iterations vs time_unit and train set
        pass

    def inner_loop_validation(self, model_class_name, grid_size, train_subset_dates, current_validation_date, metrics):
        model_object = globals()[model_class_name]()
        df_train_subset = ProcessData.filter_by_date(self.df_train, self.dataset_dict, train_subset_dates['initial'], train_subset_dates['final'])
        trained_model = model_object.train(df_train_subset, self.dataset_dict, grid_size)
        ## TODO: save trained_model

        ### validation
        delta_hour = 6
        hotspot_percentage = 20
        area_rates = [10,20,30,40,50,60,70,80,90,100]
        area_rates = list(map(lambda a: a*hotspot_percentage/100, area_rates))

        validation_dates = {'initial':current_validation_date,'final':current_validation_date}
        df_validation = ProcessData.filter_by_date(self.df_train, self.dataset_dict, validation_dates['initial'], validation_dates['final'])
        validation_points, validation_region = ProcessData.get_time_space_points(df_validation, self.dataset_dict)
        prediction_date = current_validation_date
        flag_performance_array = True

        for hour in range(0, 24, delta_hour):
            time_predicts = np.datetime64(prediction_date) + np.timedelta64(hour, 'h')
            score_end_time = time_predicts + np.timedelta64(delta_hour, 'h')
            mask = (validation_points.timestamps >= time_predicts) & (validation_points.timestamps < score_end_time)
            eval_pts = validation_points[mask]

            prediction_datetime = prediction_date+timedelta(hours=hour)
            prediction = model_object.predict(trained_model, prediction_datetime)

            hitrates = open_cp.evaluation.hit_rates(prediction, eval_pts, area_rates)
            if flag_performance_array==True:
                flag_performance_array = False
                performance_metrics = np.array([prediction_datetime,hitrates]);
            else:
                performance_metrics = np.vstack((performance_metrics, [prediction_datetime,hitrates]))
        return performance_metrics

    def second_half(self, df, time_unit, initial_date):
        pass

    def update_train_validation_subsets(train_subset_dates,current_validation_date):
        train_subset_dates['final'] = train_subset_dates['final'] + timedelta(days=1)
        current_validation_date = current_validation_date + timedelta(days=1)
        return (train_subset_dates, current_validation_date)

    def walk_fwd_chain(self, model_class_name, grid_size, dates, metrics):
        initial_date = datetime.strptime(dates['initial'],'%Y-%m-%d')
        final_date = datetime.strptime(dates['final'],'%Y-%m-%d')
        current_validation_date = initial_date+timedelta(days=1)
        train_subset_dates = {'initial':initial_date,'final':initial_date}
        performance_array = []
        while current_validation_date <= final_date:
            performance_metrics = self.inner_loop_validation(model_class_name, grid_size, train_subset_dates, current_validation_date, metrics)
            performance_array.append(performance_metrics)
            train_subset_dates, current_validation_date = ValidateModel.update_train_validation_subsets(train_subset_dates,current_validation_date)
        ## TODO:  find average performance
        return performance_array
