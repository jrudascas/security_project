import open_cp
import open_cp.predictors
import open_cp.naive as naive
import open_cp.kde as kde
import open_cp.seppexp as sepp

from services.process_data import ProcessData

class AggressiveModel:

    def __init__(self):
        pass

    def predict(self, trained_model, prediction_datetime):
        """ prediction_datetime is not used on naive predictors (e.g. CountingGridKernel, KDE)"""
        return trained_model.predict()

class NaiveCounting(AggressiveModel):

    def train(self, df_train_subset, dataset_dict, grid_size):
        training_points, training_region = ProcessData.get_time_space_points(df_train_subset, dataset_dict)
        trained_model = naive.CountingGridKernel(grid_width=grid_size, region=training_region)
        trained_model.data = training_points
        return trained_model

class SpaceTimeKDE(AggressiveModel):

    def train(self, df_train_subset, dataset_dict, grid_size):
        training_points, training_region = ProcessData.get_time_space_points(df_train_subset, dataset_dict)
        trained_model = kde.KDE(region=training_region, grid_size=grid_size)
        trained_model.time_kernel = kde.ExponentialTimeKernel(1)
        trained_model.space_kernel = kde.GaussianBaseProvider()
        trained_model.data = training_points
        return trained_model

    # TODO: check predict method

class SEPP(AggressiveModel):

    def train(self, df_train_subset, dataset_dict, grid_size):
        training_points, training_region = ProcessData.get_time_space_points(df_train_subset, dataset_dict)
        trainer = sepp.SEPPTrainer(region=training_points, grid_size=grid_size)
        trainer.data = training_points
        trained_model = trainer.train(iterations=100, use_corrected=True)
        return trained_model

    def predict(self, trained_model, prediction_datetime):
        # TODO: define prediction_points
        trained_model.data = prediction_points
        return trained_model.predict(prediction_datetime)

class SEPPCovModel(AggressiveModel):
    # TODO:
    def train(self, df_train_subset, dataset_dict, grid_size):
        #should return fitted model
        pass
