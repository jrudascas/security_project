from nilearn.decomposition import CanICA
from sklearn.datasets import fetch_olivetti_faces
from numpy.random import RandomState
import numpy as np
from geopandas import GeoDataFrame
import pandas as pd
from shapely.geometry import Point
from sklearn.decomposition import FastICA, PCA
from utils import fast_abs_percentile
import matplotlib.pyplot as plt


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

    in_map_by_geometry = np.array([gdf.geometry.within(geom) for geom in boros.geometry])

    raw = []

    date_sequence = [d.strftime('%Y-%m-%d') for d in pd.date_range(min_date, max_date, freq='D')]

    for pos, val in enumerate(boros.geometry):
        gdf_by_geometry = gdf[in_map_by_geometry[pos]]
        gdf_by_geometry_grouped = pd.DataFrame(gdf_by_geometry.groupby(['DATE']).size(), columns=["EVENTS"]).sort_index()

        values_fitted = []
        for i, value in enumerate(date_sequence):
            if not value[-5:] == '02-29':
                values_fitted.append(
                    0 if value not in gdf_by_geometry_grouped.index.values else gdf_by_geometry_grouped.loc[value][
                        'EVENTS'])

        raw.append(values_fitted)

    np_raw = np.array(raw)
    np_raw[np.isnan(np_raw)] = 0
    np_raw[np.isinf(np_raw)] = 0
    np_raw[np.isneginf(np_raw)] = 0

    return np_raw

n_components = 5
spatial_scale_geometry = '/home/jrudascas/Downloads/locashp/Loca.shp'

temporal_filters_years = [2014, 2015, 2016, 2017, 2018]

results = []
for year in temporal_filters_years:
    filter = (str(year) + '-01-01', str(year) + '-12-31')
    data = compute_experiment(spatial_scale_geometry=spatial_scale_geometry, temporal_filter=filter)
    results.append(data)


for i in results:
    if not 'concatened' in locals():
        concatened = i.copy()
    else:
        data_aux = i.copy()
        concatened = np.concatenate((concatened, data_aux), axis=1)

X = np.transpose(concatened)

n_samples, n_features = X.shape

# global centering
X = X - X.mean(axis=0)
# local centering
#X -= X.mean(axis=1).reshape(n_samples, -1)

print(X.shape)

descomposition = FastICA(n_components=n_components, random_state = 1)
spatial_ic_ = descomposition.fit_transform(X)
temporal_ic = descomposition.mixing_

print(spatial_ic_.shape)
print(temporal_ic.shape)
temporal_ic = np.abs(temporal_ic)
temporal_ic[np.where(temporal_ic < fast_abs_percentile(temporal_ic))] = 0

boros = GeoDataFrame.from_file(spatial_scale_geometry)

s_gdf = GeoDataFrame(pd.DataFrame(temporal_ic, columns=['C' + str(i + 1) for i in range(n_components)]), geometry=boros.geometry)

f, ax = plt.subplots(1, n_components)

for i in range(n_components):
    s_gdf.plot(ax=ax[i], cmap='Reds', column='C' + str(i + 1), vmin=0, vmax=np.max(temporal_ic))
    ax[i].set_title('IC ' + str(i + 1))

plt.savefig('groups_ica', dpi=700)
plt.close()