from datetime import datetime, timedelta
import numpy as np
import open_cp
from open_cp import evaluation

from services.process_data import ProcessData
from services.aggressive_model import NaiveCounting, SpaceTimeKDE, SEPPexp

class ValidateModel:

    def __init__(self, df_train_validation, dataset_dict, time_unit, outer_iterations):
        self.df_train_validation = df_train_validation
        self.dataset_dict = dataset_dict
        self.time_unit = time_unit
        self.outer_iterations = outer_iterations

    def check_validation_params(self):
        #check number of iterations vs time_unit and train set
        pass

    def inner_loop_validation(self, model_class_name, grid_size, train_subset_dates, current_validation_date, metrics):
        model_object = globals()[model_class_name]()
        df_train_subset = ProcessData.filter_by_date(self.df_train_validation, self.dataset_dict, train_subset_dates['initial'], train_subset_dates['final'])
        print(len(df_train_subset))
        trained_model = model_object.train(df_train_subset, self.dataset_dict, grid_size)
        ## TODO: save trained_model

        ### validation
        delta_hour = 6
        hotspot_percentage = 20
        area_rates = [10,20,30,40,50,60,70,80,90,100]
        area_rates = list(map(lambda a: a*hotspot_percentage/100, area_rates))

        validation_dates = {'initial':current_validation_date,'final':current_validation_date}
        df_validation = ProcessData.filter_by_date(self.df_train_validation, self.dataset_dict, validation_dates['initial'], validation_dates['final'])
        try:
            validation_points, validation_region = ProcessData.get_time_space_points(df_validation, self.dataset_dict)
        except: #if no points (e.g. crimes) are not reported on data interval
            pass
        prediction_date = current_validation_date
        flag_performance_array = True

        for init_interval_hour in range(0, 24, delta_hour):
            prediction_datetime = prediction_date+timedelta(hours=init_interval_hour)
            if df_validation.empty: #if no points (e.g. crimes) are reported on data interval
                hitrates = { i : -1.0 for i in area_rates }
            else:
                time_predicts = np.datetime64(prediction_date) + np.timedelta64(init_interval_hour, 'h')
                score_end_time = time_predicts + np.timedelta64(delta_hour, 'h')
                mask = (validation_points.timestamps >= time_predicts) & (validation_points.timestamps < score_end_time)
                eval_pts = validation_points[mask]
                prediction_by_hour = []
                hour_step = 1
                for hour in range(0,delta_hour,hour_step):
                    print(hour)
                    prediction_datetime = prediction_datetime+timedelta(hours=hour_step)
                    prediction = model_object.predict(trained_model, prediction_datetime)
                    print(prediction_datetime)
                    print(prediction)
                    print(list(prediction.__dict__.keys()))
                    print(prediction.xoffset)
                    print(prediction.yoffset)
                    print(prediction.xsize)
                    print(prediction.ysize)
                    print(prediction._matrix)
                    prediction_by_hour.append(prediction)
                print(prediction_by_hour)
                xoffset_avg = sum([p._xoffset for p in prediction_by_hour]) / len(prediction_by_hour)
                print('xoffset_avg',xoffset_avg)
                yoffset_avg = sum([p._yoffset for p in prediction_by_hour]) / len(prediction_by_hour)
                print('yoffset_avg',yoffset_avg)
                xsize_avg = sum([p._xsize for p in prediction_by_hour]) / len(prediction_by_hour)
                print('xsize_avg',xsize_avg)
                ysize_avg = sum([p._ysize for p in prediction_by_hour]) / len(prediction_by_hour)
                print('ysize_avg',ysize_avg)
                matrix_avg = sum([p._matrix for p in prediction_by_hour]) / len(prediction_by_hour)
                print('matrix_avg',matrix_avg)

                # TODO: Include on test
                #Test: since the prediction is the same for all hours using NaiveCounting model, average instance values should be equivalent to specific prediction values
                print(prediction._xoffset == xoffset_avg)
                print(prediction._yoffset == yoffset_avg)
                print(prediction._xsize == xsize_avg)
                print(prediction._ysize == ysize_avg)
                print((prediction._matrix == matrix_avg).all())

                avg_grid_pred_array = open_cp.predictors.GridPredictionArray(xsize=xsize_avg, ysize=ysize_avg, matrix=matrix_avg, xoffset=xoffset_avg, yoffset=yoffset_avg)
                print(avg_grid_pred_array)

                hitrates = open_cp.evaluation.hit_rates(avg_grid_pred_array, eval_pts, area_rates)

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

    def walk_fwd_chain(self, model_class_name, grid_size, train_dates_base, validation_dates, metrics):
        initial_train_date = datetime.strptime(train_dates_base['initial'],'%Y-%m-%d')
        final_train_date = datetime.strptime(train_dates_base['final'],'%Y-%m-%d')
        initial_validation_date = datetime.strptime(validation_dates['initial'],'%Y-%m-%d')
        final_validation_date = datetime.strptime(validation_dates['final'],'%Y-%m-%d')
        current_validation_date = initial_validation_date
        train_subset_dates = {'initial':initial_train_date,'final':final_train_date}
        performance_array = []
        while current_validation_date <= final_validation_date:
            print('train_subset_dates', train_subset_dates)
            print('current_validation_date', current_validation_date)
            performance_metrics = self.inner_loop_validation(model_class_name, grid_size, train_subset_dates, current_validation_date, metrics)
            performance_array.append(performance_metrics)
            train_subset_dates, current_validation_date = ValidateModel.update_train_validation_subsets(train_subset_dates,current_validation_date)
        ## TODO:  find average performance
        return performance_array
