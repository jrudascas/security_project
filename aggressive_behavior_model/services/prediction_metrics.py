import open_cp
from open_cp import evaluation

class PredictionMetrics:

    def __init__(self):
        pass

    def hit_rate(self, grid_real, grid_prediction, coverage):
        #should return hit_rate value
        pass

    def measure_hitrates(prediction, real_events):
        hotspot_percentage = 20
        area_rates = [10,20,30,40,50,60,70,80,90,100]
        area_rates = list(map(lambda a: a*hotspot_percentage/100, area_rates))
        if not real_events:
            return { i : -1.0 for i in area_rates }
        else:
            return open_cp.evaluation.hit_rates(prediction, real_events, area_rates)

    def pai(self, hit_rate_value, coverage):
        pass
