import numpy as _np
import open_cp
from open_cp import evaluation
import sklearn
from sklearn.metrics import mean_squared_error

from services.process_data import ProcessData


def hit_counts_ground_truth(grid_pred, timed_points, percentage_coverage):
    """percentage_coverage is used as a top restriction, but hit rates are
    estimated over a fixed coverage: the one determined by the area where real
    events occurred (ground truth). This aproximation leads to a more accurate
    model performance measure (prediction vs. ground truth).
    """
    if len(timed_points.xcoords) == 0:
        return {cov : (0,0) for cov in percentage_coverage}
    risk = grid_pred.intensity_matrix
    out = dict()

    for coverage in percentage_coverage:
        # this is the ground_truth
        gx, gy = grid_pred.grid_coord(timed_points.xcoords, timed_points.ycoords)
        gx, gy = gx.astype(_np.int), gy.astype(_np.int)
        ground_events_coverage = len(gx) * 100 / grid_pred.intensity_matrix.size
        # select top hotspots on prediction array, according to coverage of ground truth events
        if ground_events_coverage > coverage:
            covered = open_cp.evaluation.top_slice(risk, coverage / 100)
        else:
            covered = open_cp.evaluation.top_slice(risk, ground_events_coverage / 100)
            coverage = ground_events_coverage
        mask = (gx < 0) | (gx >= covered.shape[1]) | (gy < 0) | (gy >= covered.shape[0])
        gx, gy = gx[~mask], gy[~mask]
        count = _np.sum(covered[(gy,gx)]) # count events that occur on selected hotspots
        out[coverage] = (count, len(timed_points.xcoords))
    return out

def make_counting_grid(grid_pred, timed_points):
    """ Use naive counting predictor as proxy to compute the counting
        matrix of ground truth events.
    """
    counting_kernel = open_cp.naive.CountingGridKernel(grid_pred.xsize,
                                                       grid_pred.ysize,
                                                       grid_pred.region())
    counting_kernel.data = timed_points
    counting_matrix = counting_kernel.predict()
    return counting_matrix

def measure_hit_rates(prediction, real_events, coverages, method):
    """ This is a true positives measure.
    """
    if not real_events:
        return { c : -1.0 for c in coverages }
    if len(real_events.xcoords) == 0:
        return { c : -1.0 for c in coverages }

    if method == 'default':
        out = open_cp.evaluation.hit_counts(prediction, real_events, coverages)
    if method == 'ground_truth_coverage':
        out = hit_counts_ground_truth(prediction, real_events, coverages)

    return {k : a/b for k, (a,b) in out.items()}

def mse(grid_pred, real_events):
    """ Computes the "mean aquared error" between the prediction and the ground
        truth. Is computed as the sum of each cell squared difference.

    :param grid_pred: An instance of :class:`GridPrediction` matrix attribute
                      must be normalized.
    :param real_events: An instance of :class: open_cep.data.TimedPoints

    :return: A non-negative floating point value
    """

    counting_matrix = make_counting_grid(grid_pred, real_events)
    counting_matrix._matrix = ProcessData.normalize_matrix(counting_matrix._matrix)
    return mean_squared_error(grid_pred._matrix, counting_matrix._matrix)
