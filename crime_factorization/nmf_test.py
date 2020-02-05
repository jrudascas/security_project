from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point, Polygon
import numpy as np
import matplotlib
from scipy.stats import pearsonr


def compute_experiment(n_components, spatial_scale_geometry, temporal_filter, output_name, threshold=1e-02):
    latitude_field_name = 'LATITUD'
    longitude_field_name = 'LONGITUD'
    date_field_name = 'FECHA'
    min_date, max_date = temporal_filter
    figure_name = output_name + '_IC_' + str(n_components)
    df = pd.read_csv('/home/jrudascas/Downloads/verify_enrich_nuse_29112019.csv')
    boros = GeoDataFrame.from_file(spatial_scale_geometry)

    gdf = GeoDataFrame(df.drop([latitude_field_name, longitude_field_name], axis=1),
                       geometry=[Point(xy) for xy in zip(df[longitude_field_name], df[latitude_field_name])])
    gdf['DATE'] = pd.to_datetime(gdf[date_field_name]).dt.strftime('%Y%m%d')
    gdf = gdf[:10000]

    in_map_by_geometry = np.array([gdf.geometry.within(geom) for geom in boros.geometry])

    # print(in_map_by_geometry.shape)

    raw = []

    date_sequence = [d.strftime('%Y%m%d') for d in pd.date_range(min_date, max_date, freq='D')]

    for pos, val in enumerate(boros.geometry):
        gdf_by_geometry = gdf[in_map_by_geometry[pos]]
        gdf_by_geometry_grouped = pd.DataFrame(gdf_by_geometry.groupby(['DATE']).size(), columns=["EVENTS"]).sort_index(
            by=['DATE'])

        values_fitted = []
        for i, value in enumerate(date_sequence):
            values_fitted.append(
                0 if value not in gdf_by_geometry_grouped.index.values else gdf_by_geometry_grouped.loc[value][
                    'EVENTS'])

        raw.append(values_fitted)

    np_raw = np.array(raw)
    np_raw[np.isnan(np_raw)] = 0
    np_raw[np.isinf(np_raw)] = 0
    np_raw[np.isneginf(np_raw)] = 0

    model = NMF(n_components=n_components, init='random', random_state=0)
    W = model.fit_transform(np_raw)
    H = model.components_

    print(W.shape)
    print(H.shape)

    return W, H


spatial_scale_geometry = '/home/jrudascas/Downloads/locashp/Loca.shp'

filter = ('2017-01-01', '2017-12-31')
output_name = 'localidades_2017'
out_put_ic_3_exp_1, out_put_temp_3_exp_1 = compute_experiment(n_components=3,
                                                              spatial_scale_geometry=spatial_scale_geometry,
                                                              temporal_filter=filter, output_name=output_name)
