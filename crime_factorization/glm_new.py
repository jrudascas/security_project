import numpy as np
from geopandas import GeoDataFrame
import pandas as pd
from shapely.geometry import Point
from sklearn.decomposition import FastICA, PCA
from utils import fast_abs_percentile
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, ElasticNet
from sklearn import linear_model
from nilearn import plotting
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

data = pd.read_csv('/home/jrudascas/Downloads/arboles_localidad.csv')

Y = data['RINAS2018']

X = np.asarray([data['ARBOLES'], data['AREA'], data['LUMINARIA'], data['VENDEDORES'], data['SUICIDIO18'], data['EMBADOLENCENTE18'], data['POBLACION']])
X = np.transpose(X)

print(Y.shape)
print(X.shape)

gamma_model = sm.GLM(Y, X, family=sm.families.Gamma())

gamma_results = gamma_model.fit()

print(gamma_results.summary())