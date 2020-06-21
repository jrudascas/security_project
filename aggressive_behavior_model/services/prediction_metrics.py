import numpy as _np
import open_cp
from open_cp import evaluation as evaluation


def hit_counts(grid_pred, timed_points, percentage_coverage):
    """As :func:`hit_rates` but return pairs `(captured_count, total_count)`
    instead of the rate `captured_count / total_count`.
    """
    if len(timed_points.xcoords) == 0:
        return {cov : (0,0) for cov in percentage_coverage}
    risk = grid_pred.intensity_matrix
    out = dict()
    for coverage in percentage_coverage:
        covered = evaluation.top_slice(risk, coverage / 100)
        gx, gy = grid_pred.grid_coord(timed_points.xcoords, timed_points.ycoords)
        gx, gy = gx.astype(_np.int), gy.astype(_np.int)
        print('gx, gy', gx, gy)
        mask = (gx < 0) | (gx >= covered.shape[1]) | (gy < 0) | (gy >= covered.shape[0])
        gx, gy = gx[~mask], gy[~mask]
        print('gx, gy',gx, gy)
        print('len gx',len(gx))
        count = _np.sum(covered[(gy,gx)])
        print(covered[(gy,gx)])
        print('count',count)

        out[coverage] = (count, len(timed_points.xcoords))
    return out

def measure_hitrates(prediction, real_events, coverages):
    #coverages = [2,4,6,8,10,12,14,16,18,20]
    if not real_events:
        return { c : -1.0 for c in coverages }
    if len(real_events.xcoords) == 0:
        return { c : -1.0 for c in coverages }
    out = hit_counts(prediction, real_events, coverages)
    return {k : a/b for k, (a,b) in out.items()}


class PredictionMetrics:

    def __init__(self):
        pass

    def hit_rate(self, grid_real, grid_prediction, coverage):
        #should return hit_rate value
        pass

    def measure_hitrates_2(prediction, real_events, coverages):
        #coverages = [2,4,6,8,10,12,14,16,18,20]
        if not real_events:
            return { i : -1.0 for i in coverages }
        else:
            return open_cp.evaluation.hit_rates_2(prediction, real_events, coverages)

    def pai(self, hit_rate_value, coverage):
        pass
