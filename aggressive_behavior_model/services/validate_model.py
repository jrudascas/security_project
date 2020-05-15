from services.aggressive_model import KDEModel

class ValidateModel:

    def __init__(self, train_set, initial_train_date, time_unit, outer_iterations):
        self.train_set = train_set
        self.initial_train_date = initial_train_date
        self.time_unit = time_unit
        self.outer_iterations = outer_iterations

    def check_validation_params(self):
        #check number of iterations vs time_unit and train set
        #check initial_train_date into train_set
        pass

    def walk_fwd_chain(prediction_experiment_object, model):
        # TODO: check how we propagate model object, is a new object on each iteration (tune parameters)?
        # iterate over outer_iterations
        # split train set: consider time unit to build first train+validation+test
            # train model on iteration
            # test model on iteration
            # measure performance
        # save performance metric
        # find average performance
        pass

    def second_half(self, df, time_unit, initial_date):
        pass
