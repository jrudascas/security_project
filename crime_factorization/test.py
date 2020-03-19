"""
Image compression via tensor decomposition
==========================================

Example on how to use :func:`tensorly.decomposition.parafac`and :func:`tensorly.decomposition.tucker` on images.
"""

import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np
from scipy.misc import face
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker, non_negative_parafac
from math import ceil
import matplotlib
from geopandas import GeoDataFrame
import statsmodels.api as sm

Y = np.random.rand(19,2)
X = np.random.rand(19,40)

print(Y.shape)
print(X.shape)

data = sm.datasets.scotland.load(as_pandas=False)
print(data.exog.shape)
print(data.endog.shape)

gamma_model = sm.GLM(Y, X, family=sm.families.Gamma())

gamma_results = gamma_model.fit()

print(gamma_results.summary())