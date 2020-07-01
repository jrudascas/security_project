import open_cp
import open_cp.predictors
import open_cp.naive as naive
import open_cp.kde as kde
import open_cp.seppexp as seppexp

from services.process_data import ProcessData

class AggressiveModel:

    def __init__(self):
        pass

    def predict(self, trained_model, prediction_datetime):
        """ prediction_datetime is not used on naive predictors (e.g. CountingGridKernel, KDE)"""
        return trained_model.predict()

class NaiveCounting(AggressiveModel):

    def train(self, df_train_subset, dataset_dict, grid_size, **kwargs):
        train_pts, train_region = ProcessData.get_time_space_points(df_train_subset, dataset_dict)
        if kwargs['region'] != None:
            train_region = kwargs['region']
        trained_model = naive.CountingGridKernel(grid_width=grid_size, region=train_region)
        trained_model.data = train_pts
        return trained_model

class SpaceTimeKDE(AggressiveModel):

    def train(self, df_train_subset, dataset_dict, grid_size, **kwargs):
        train_pts, train_region = ProcessData.get_time_space_points(df_train_subset, dataset_dict)
        if kwargs['region'] != None:
            train_region = kwargs['region']
        trained_model = kde.KDE(region=train_region, grid_size=grid_size)
        trained_model.time_kernel = kde.ExponentialTimeKernel(1)
        trained_model.space_kernel = kde.GaussianBaseProvider()
        trained_model.data = train_pts
        return trained_model

class SEPPexp(AggressiveModel):

    def train(self, df_train_subset, dataset_dict, grid_size, **kwargs):
        train_pts, train_region = ProcessData.get_time_space_points(df_train_subset, dataset_dict)
        #if kwargs['region'] != 'default':
        if 'region' in kwargs:
            train_region = kwargs['region']
        trainer = seppexp.SEPPTrainer(region=train_region, grid_size=grid_size)
        trainer.data = train_pts
        trained_model = trainer.train(iterations=50, use_corrected=True)
        trained_model.data = train_pts
        return trained_model

    def predict(self, trained_model, prediction_datetime):
        return trained_model.predict(prediction_datetime)

class SEPPexpWeekDay(AggressiveModel):
    """This model is trained according to the week-day that will be predicted.
    Just historical data of same week-day is selected to train the model.
    """
    def train(self, df_train_subset, dataset_dict, grid_size, **kwargs):
        df_train_subset['weekday'] = df_train_subset['TIME_STAMP'].dt.day_name()
        df_train_subset = ProcessData.filter_by_field(df_train_subset,
                                                      'weekday',
                                                      kwargs['week_day'])

        train_pts, train_region = ProcessData.get_time_space_points(df_train_subset, dataset_dict)
        if kwargs['region'] != None:
            train_region = kwargs['region']
        trainer = seppexp.SEPPTrainer(region=train_region, grid_size=grid_size)
        trainer.data = train_pts
        trained_model = trainer.train(iterations=50, use_corrected=True)
        trained_model.data = train_pts
        return trained_model

    def predict(self, trained_model, prediction_datetime):
        return trained_model.predict(prediction_datetime)

class SEPPCov(AggressiveModel):
    # TODO:
    def train(self, df_train_subset, dataset_dict, grid_size, **kwargs):
        #should return fitted model
        pass
