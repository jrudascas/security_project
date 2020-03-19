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


def compute_hypergraph_elastic_net(time_series, alpha=0.1, threshold=0):
    # 1 / (2 * n_samples) * ||y - Xw||^2_2 + alpha * l1_ratio * ||w||_1 + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

    clf = ElasticNet(alpha=alpha, l1_ratio=0.25, max_iter=8000, tol=1e-2)

    hypergraph = np.zeros((time_series.shape[1], time_series.shape[1]))

    for i in range(time_series.shape[1]):
        X = time_series.copy()
        Y = X[:, i].copy()
        X[:, i] = 0

        hypergraph[i, :] = clf.fit(X, Y).coef_
        hypergraph[i, np.where(hypergraph[i, :] > threshold)] = 1
        hypergraph[i, np.where(hypergraph[i, :] < -threshold)] = 1
        hypergraph[i, np.where(hypergraph[i, :] != 1)] = 0

    print('HyperGraph computing finished')
    return hypergraph


def compute_experiment(spatial_scale_geometry, temporal_filter, threshold=0.1):
    latitude_field_name = 'LATITUD'
    longitude_field_name = 'LONGITUD'
    date_field_name = 'FECHA'
    min_date, max_date = temporal_filter
    df = pd.read_csv('/home/jrudascas/Downloads/Copia de 06. verify_enrich_nuse_11022020.csv')
    boros = GeoDataFrame.from_file(spatial_scale_geometry)

    df['DATE'] = pd.to_datetime(df[date_field_name]).dt.strftime('%Y-%m-%d')
    df = df[(df['DATE'] >= min_date) & (df['DATE'] <= max_date)]

    gdf = GeoDataFrame(df.drop([latitude_field_name, longitude_field_name], axis=1),
                       geometry=[Point(xy) for xy in zip(df[longitude_field_name], df[latitude_field_name])])
    #gdf = gdf[:1000]
    print(df.shape)
    in_map_by_geometry = np.array([gdf.geometry.within(geom) for geom in boros.geometry])

    raw = []

    for pos, val in enumerate(boros.geometry):
        gdf_by_geometry = gdf[in_map_by_geometry[pos]]
        gdf_by_geometry_grouped = pd.DataFrame(gdf_by_geometry.groupby(['ANIO']).size(), columns=["EVENTS"]).sort_index()

        raw.append(gdf_by_geometry_grouped['EVENTS'])

    np_raw = np.array(raw)
    np_raw = np.delete(np_raw, (8), axis=0) #It is removed Sumapaz locality

    return np_raw


temporal_filters_years = [2018]
spatial_scale_geometry = '/home/jrudascas/Downloads/locashp/Loca.shp'
labels = ['ANTONIO NARIÑO',
          'TUNJUELITO',
          'RAFAEL URIBE URIBE',
          'CANDELARIA',
          'BARRIOS UNIDOS',
          'TEUSAQUILLO',
          'PUENTE ARANDA',
          'LOS MARTIRES',
          'SUMAPAZ',
          'USAQUEN',
          'CHAPINERO',
          'SANTA FE',
          'SAN CRISTOBAL',
          'USME',
          'CIUDAD BOLIVAR',
          'BOSA',
          'KENNEDY',
          'FONTIBON',
          'ENGATIVA',
          'SUBA']
results = []

filter = ('2018-01-01', '2018-12-31')
data = compute_experiment(spatial_scale_geometry=spatial_scale_geometry, temporal_filter=filter)

print('kokokokokoo')
print(data)

reg = linear_model.LinearRegression()

df_additional = pd.read_csv('/home/jrudascas/Downloads/arboles_localidad.csv')

mean_score = []

confound_label = 'LUMINARIA'
for j, data_year in enumerate(results):
    data_year = np.transpose(data_year)
    coeficientes_matrix = np.zeros((data_year.shape[1], data_year.shape[1] + 1))
    scores = np.zeros(data_year.shape[1])
    for i in range(data_year.shape[1]):
        X = data_year.copy()
        Y = X[:, i].copy()
        X[:, i] = 0

        other_variables = np.full((data_year.shape[0], 1), df_additional[confound_label][i], dtype=int)

        new_X = np.concatenate((X, other_variables), axis=1)

        model = reg.fit(new_X, Y)
        coeficientes_matrix[i, :] = model.coef_
        scores[i] = model.score(new_X, Y)

    figure = plt.figure(figsize=(6, 6))
    plotting.plot_matrix(coeficientes_matrix, figure=figure)
    figure.savefig('coeficientes_matrix_' + confound_label + '_' + str(j) + '.png', dpi=200)
    plt.close(figure)

    np.savetxt('beta_values_' + confound_label + '_' + str(j) + '.txt', coeficientes_matrix, delimiter=',', fmt='%.2f')
    np.savetxt('scores_values_' + confound_label + '_' + str(j) + '.txt', scores, delimiter=',', fmt='%.2f')

    mean_score.append(np.mean(scores))

print(mean_score)

reg = linear_model.LinearRegression()

mean_score = []
confound_label = 'ARBOLES'
for j, data_year in enumerate(results):
    data_year = np.transpose(data_year)
    coeficientes_matrix = np.zeros((data_year.shape[1], data_year.shape[1] + 1))
    scores = np.zeros(data_year.shape[1])
    for i in range(data_year.shape[1]):
        X = data_year.copy()
        Y = X[:, i].copy()
        X[:, i] = 0

        other_variables = np.full((data_year.shape[0], 1), df_additional[confound_label][i], dtype=int)

        new_X = np.concatenate((X, other_variables), axis=1)

        model = reg.fit(new_X, Y)
        coeficientes_matrix[i, :] = model.coef_
        scores[i] = model.score(new_X, Y)

    figure = plt.figure(figsize=(6, 6))
    plotting.plot_matrix(coeficientes_matrix, figure=figure)
    figure.savefig('coeficientes_matrix_' + confound_label + '_' + str(j) + '.png', dpi=200)
    plt.close(figure)

    np.savetxt('beta_values_' + confound_label + '_' + str(j) + '.txt', coeficientes_matrix, delimiter=',', fmt='%.2f')
    np.savetxt('scores_values' + confound_label + '_' + str(j) + '.txt', scores, delimiter=',', fmt='%.2f')

    mean_score.append(np.mean(scores))

print(mean_score)