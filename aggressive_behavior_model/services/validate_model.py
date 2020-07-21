from datetime import datetime, timedelta
import numpy as np
import open_cp

from services.aggressive_model import NaiveCounting, SpaceTimeKDE, SEPPexp, SEPPexpWeekDay
from services.process_data import ProcessData


class ValidateModel:

    def __init__(self, df_train_validation, dataset_dict, time_unit, region):
        self.df_train_validation = df_train_validation
        self.dataset_dict = dataset_dict
        self.time_unit = time_unit
        self.region = region

    def check_validation_params(self):
        #check number of iterations vs time_unit and train set
        pass

    def inner_loop_validation(self, model_name, grid_size, train_subset_dates,current_validation_date):

        model_object = globals()[model_name]()
        df_train_subset = ProcessData.filter_by_date(self.df_train_validation,
                                                     self.dataset_dict,
                                                     train_subset_dates['initial'],
                                                     train_subset_dates['final'])
        trained_model = model_object.train(df_train_subset, self.dataset_dict,
                                            grid_size,
                                            week_day= current_validation_date.strftime("%A"),
                                            region= self.region)
        print(len(trained_model.data.timestamps))
        ## TODO: save trained_model?

        ### validation
        interval_duration = 6 ## TODO: set this var as a parameter
        validation_dates = {'initial':current_validation_date,'final':current_validation_date}
        df_validation = ProcessData.filter_by_date(self.df_train_validation,
                                                   self.dataset_dict,
                                                   validation_dates['initial'],
                                                   validation_dates['final'])
        prediction_date = current_validation_date
        flag_array = True
        prediction_results = np.array([])

        for interval_hour_start in range(0, 24, interval_duration):
            initial_prediction_datetime = prediction_date+timedelta(hours=interval_hour_start)
            final_prediction_datetime = initial_prediction_datetime+timedelta(hours=interval_duration)
            if df_validation.empty: #if no points (e.g. crimes) are reported on data interval
                eval_pts = []
            else:
                validation_pts, _ = ProcessData.get_time_space_points(df_validation,
                                                                      self.dataset_dict)
                eval_pts = ValidateModel.select_timed_points(prediction_date,
                                                             interval_hour_start,
                                                             interval_duration,
                                                             validation_pts)

            prediction_by_hour = ValidateModel.predict_on_interval(initial_prediction_datetime,
                                                                   interval_duration,
                                                                   model_object,
                                                                   trained_model)
            average_prediction = ValidateModel.interval_average_prediction(prediction_by_hour)

            element = np.array([initial_prediction_datetime,
                                final_prediction_datetime,
                                average_prediction,
                                eval_pts])
            flag_array, prediction_results = ProcessData.fill_array(flag_array,
                                                                    prediction_results,
                                                                    element)
        return prediction_results

    def interval_average_prediction(prediction_array):
        xoffset_avg = sum([p._xoffset for p in prediction_array]) / len(prediction_array)
        yoffset_avg = sum([p._yoffset for p in prediction_array]) / len(prediction_array)
        xsize_avg = sum([p._xsize for p in prediction_array]) / len(prediction_array)
        ysize_avg = sum([p._ysize for p in prediction_array]) / len(prediction_array)
        matrix_avg = sum([p._matrix for p in prediction_array]) / len(prediction_array)
        #print('risk intensity sum: ',np.sum(matrix_avg))
        #print(matrix_avg.size)

        avg = open_cp.predictors.GridPredictionArray(xsize=xsize_avg,
                                                     ysize=ysize_avg,
                                                     matrix=matrix_avg,
                                                     xoffset=xoffset_avg,
                                                     yoffset=yoffset_avg)

        return avg

    def predict_on_interval(initial_prediction_datetime, interval_duration, model_object, trained_model):
        prediction_by_hour_array = []
        hour_step = 1
        prediction_datetime = initial_prediction_datetime
        for hour in range(0,interval_duration,hour_step):
            prediction_datetime = initial_prediction_datetime+timedelta(hours=hour)
            prediction = model_object.predict(trained_model, prediction_datetime)
            prediction_by_hour_array.append(prediction)
            #print(prediction_datetime)
        return prediction_by_hour_array

    def second_half(self, df, time_unit, initial_date):
        pass

    def select_timed_points(prediction_date, interval_hour_start, interval_duration, validation_pts):
        time_predicts = np.datetime64(prediction_date) + np.timedelta64(interval_hour_start, 'h')
        score_end_time = time_predicts + np.timedelta64(interval_duration, 'h')
        mask = (validation_pts.timestamps >= time_predicts) & (validation_pts.timestamps < score_end_time)
        return validation_pts[mask]

    def update_train_validation_subsets(train_subset_dates,current_validation_date):
        train_subset_dates['final'] = train_subset_dates['final'] + timedelta(days=1)
        current_validation_date = current_validation_date + timedelta(days=1)
        return (train_subset_dates, current_validation_date)

    def walk_fwd_chain(self, model_name, grid_size, train_dates_base, validation_dates, metrics):
        initial_train_date = datetime.strptime(train_dates_base['initial'],'%Y-%m-%d')
        final_train_date = datetime.strptime(train_dates_base['final'],'%Y-%m-%d')
        initial_validation_date = datetime.strptime(validation_dates['initial'],'%Y-%m-%d')
        final_validation_date = datetime.strptime(validation_dates['final'],'%Y-%m-%d')
        current_validation_date = initial_validation_date
        train_subset_dates = {'initial':initial_train_date,'final':final_train_date}
        prediction_historical = []
        while current_validation_date <= final_validation_date:
            print('train_subset_dates', train_subset_dates)
            print('current_validation_date', current_validation_date)
            day_prediction = self.inner_loop_validation(model_name, grid_size,
                                                        train_subset_dates, current_validation_date)
            prediction_historical.append(day_prediction)
            a,b = ValidateModel.update_train_validation_subsets(train_subset_dates,
                                                                current_validation_date)
            train_subset_dates, current_validation_date = a,b

        prediction_historical_list = [item for sublist in prediction_historical for item in sublist]
        return prediction_historical_list
