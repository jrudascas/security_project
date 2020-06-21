import numpy as _np
import open_cp
from open_cp import evaluation

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
        mask = (gx < 0) | (gx >= covered.shape[1]) | (gy < 0) | (gy >= covered.shape[0])
        gx, gy = gx[~mask], gy[~mask]
        count = _np.sum(covered[(gy,gx)]) # count events that occur on selected hotspots
        out[coverage] = (count, len(timed_points.xcoords))
    return out

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
