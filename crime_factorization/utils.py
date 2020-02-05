import numpy as np


def fast_abs_percentile(data, percentile=80):
    data = np.abs(data)
    data = data.ravel()
    index = int(data.size * .01 * percentile)
    # Partial sort: faster than sort
    data = np.partition(data, index)
    return data[index]
