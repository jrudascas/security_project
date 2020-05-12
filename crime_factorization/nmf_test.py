from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point, Polygon
import numpy as np
from tensorly.decomposition import tucker
import tensorly as tl
from tensorly.decomposition import parafac


def compute_experiment(n_components, spatial_scale_geometry, temporal_filter, output_name, threshold=1e-02):
    latitude_field_name = 'LATITUD'
    longitude_field_name = 'LONGITUD'
    date_field_name = 'FECHA'
    min_date, max_date = temporal_filter
    figure_name = output_name + '_IC_' + str(n_components)
    df = pd.read_csv('/home/jrudascas/Downloads/Copia de 06. verify_enrich_nuse_11022020.csv')

    boros = GeoDataFrame.from_file(spatial_scale_geometry)

    df['DATE'] = pd.to_datetime(df[date_field_name]).dt.strftime('%Y-%m-%d')
    df = df[(df['DATE'] >= min_date) & (df['DATE'] <= max_date)]

    print(df.shape)

    gdf = GeoDataFrame(df.drop([latitude_field_name, longitude_field_name], axis=1),
                       geometry=[Point(xy) for xy in zip(df[longitude_field_name], df[latitude_field_name])])
    #gdf['DATE'] = pd.to_datetime(gdf[date_field_name]).dt.strftime('%Y%m%d')
    gdf = gdf[:1000]

    in_map_by_geometry = np.array([gdf.geometry.within(geom) for geom in boros.geometry])

    # print(in_map_by_geometry.shape)

    raw = []

    date_sequence = [d.strftime('%Y-%m-%d') for d in pd.date_range(min_date, max_date, freq='D')]

    for pos, val in enumerate(boros.geometry):
        gdf_by_geometry = gdf[in_map_by_geometry[pos]]
        gdf_by_geometry_grouped = pd.DataFrame(gdf_by_geometry.groupby(['DATE']).size(), columns=["EVENTS"]).sort_index()

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

    df_additional = pd.read_csv('/home/jrudascas/Downloads/arboles_localidad.csv')

    np_raw = np.delete(np_raw, (8), axis=0) #It is removed Sumapaz locality

    tensor = np.zeros((np_raw.shape[0], np_raw.shape[1], 19, 19))

    print(tensor.shape)
    tensor[:,:,0,0] = np_raw

    tensor[0,0,:,0] = df_additional['ARBOLES']
    tensor[0,0,0,:] = df_additional['LUMINARIA']

    W, H = parafac(tensor, rank=n_components, init='random', tol=10e-6)

    return W, H


spatial_scale_geometry = '/home/jrudascas/Downloads/locashp/Loca.shp'

filter = ('2018-01-01', '2018-12-31')
output_name = 'localidades_2018'
out_put_ic_3_exp_1, out_put_temp_3_exp_1 = compute_experiment(n_components=3,
                                                              spatial_scale_geometry=spatial_scale_geometry,
                                                              temporal_filter=filter, output_name=output_name)


