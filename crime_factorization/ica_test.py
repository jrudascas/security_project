import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point, Polygon
import numpy as np
import matplotlib
from scipy.stats import pearsonr

font = {'size': 3}
matplotlib.rc('font', **font)
plt.axis('off')
plt.xticks([])
plt.yticks([])


def compare_ica_results(*args):
    similarity_list = []
    comparator_number = len(args)

    for p in range(comparator_number):
        for q in range(p + 1, comparator_number, 1):
            for i in range(args[p].shape[1]):
                for j in range(args[q].shape[1]):
                    similarity_list = pearsonr(args[p][:, i], args[q][:, j])[0]
    return np.mean(similarity_list)


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
    # gdf = gdf[:10000]

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

    ica = FastICA(n_components=n_components)
    spatial_ic_ = ica.fit_transform(np_raw)
    temporal_ic = ica.mixing_

    # print(spatial_ic_.shape)
    # print(temporal_ic.shape)

    spatial_ic_ = np.abs(spatial_ic_)
    spatial_ic_[np.where(spatial_ic_ < threshold)] = 0

    s_gdf = GeoDataFrame(pd.DataFrame(spatial_ic_, columns=['C' + str(i + 1) for i in range(n_components)]),
                         geometry=boros.geometry)
    # a_gdf = GeoDataFrame(pd.DataFrame(A_, columns=list(range(1, n_components + 1))), geometry=boros.geometry)

    f, ax = plt.subplots(1, n_components)

    for i in range(n_components):
        s_gdf.plot(ax=ax[i], cmap='RdBu', column='C' + str(i + 1), legend=True, vmin=threshold, vmax=0.5)
        ax[i].set_title('IC ' + str(i + 1))

    plt.savefig(figure_name, dpi=700)
    plt.close()

    return spatial_ic_, temporal_ic


spatial_scale_geometry = '/home/jrudascas/Downloads/locashp/Loca.shp'

filter = ('2017-01-01', '2017-12-31')
output_name = 'localidades_2017'
out_put_ic_3_exp_1, out_put_temp_3_exp_1 = compute_experiment(n_components=3,
                                                              spatial_scale_geometry=spatial_scale_geometry,
                                                              temporal_filter=filter, output_name=output_name)
out_put_ic_4_exp_1, out_put_temp_4_exp_1 = compute_experiment(n_components=4,
                                                              spatial_scale_geometry=spatial_scale_geometry,
                                                              temporal_filter=filter, output_name=output_name)
out_put_ic_5_exp_1, out_put_temp_5_exp_1 = compute_experiment(n_components=5,
                                                              spatial_scale_geometry=spatial_scale_geometry,
                                                              temporal_filter=filter, output_name=output_name)
out_put_ic_7_exp_1, out_put_temp_7_exp_1 = compute_experiment(n_components=7,
                                                              spatial_scale_geometry=spatial_scale_geometry,
                                                              temporal_filter=filter, output_name=output_name)

filter = ('2018-01-01', '2018-12-31')
output_name = 'localidades_2018'
out_put_ic_3_exp_2, out_put_temp_3_exp_2 = compute_experiment(n_components=3,
                                                              spatial_scale_geometry=spatial_scale_geometry,
                                                              temporal_filter=filter, output_name=output_name)
out_put_ic_4_exp_2, out_put_temp_4_exp_2 = compute_experiment(n_components=4,
                                                              spatial_scale_geometry=spatial_scale_geometry,
                                                              temporal_filter=filter, output_name=output_name)
out_put_ic_5_exp_2, out_put_temp_5_exp_2 = compute_experiment(n_components=5,
                                                              spatial_scale_geometry=spatial_scale_geometry,
                                                              temporal_filter=filter, output_name=output_name)
out_put_ic_7_exp_2, out_put_temp_7_exp_2 = compute_experiment(n_components=7,
                                                              spatial_scale_geometry=spatial_scale_geometry,
                                                              temporal_filter=filter, output_name=output_name)

print('IC consistency across years')

print(compare_ica_results(out_put_ic_3_exp_1, out_put_ic_3_exp_2))
print(compare_ica_results(out_put_ic_4_exp_1, out_put_ic_4_exp_2))
print(compare_ica_results(out_put_ic_5_exp_1, out_put_ic_5_exp_2))
print(compare_ica_results(out_put_ic_7_exp_1, out_put_ic_7_exp_2))

filter = ('2017-01-01', '2017-01-31')
output_name = 'localidades_2017_1'
out_put_ic_3_exp_11, out_put_temp_3_exp_11 = compute_experiment(n_components=3,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_4_exp_11, out_put_temp_4_exp_11 = compute_experiment(n_components=4,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_5_exp_11, out_put_temp_5_exp_11 = compute_experiment(n_components=5,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_7_exp_11, out_put_temp_7_exp_11 = compute_experiment(n_components=7,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)

filter = ('2017-02-01', '2017-02-28')
output_name = 'localidades_2017_2'
out_put_ic_3_exp_12, out_put_temp_3_exp_12 = compute_experiment(n_components=3,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_4_exp_12, out_put_temp_4_exp_12 = compute_experiment(n_components=4,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_5_exp_12, out_put_temp_5_exp_12 = compute_experiment(n_components=5,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_7_exp_12, out_put_temp_7_exp_12 = compute_experiment(n_components=7,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)

filter = ('2017-03-01', '2017-03-31')
output_name = 'localidades_2017_3'
out_put_ic_3_exp_13, out_put_temp_3_exp_13 = compute_experiment(n_components=3,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_4_exp_13, out_put_temp_4_exp_13 = compute_experiment(n_components=4,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_5_exp_13, out_put_temp_5_exp_13 = compute_experiment(n_components=5,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_7_exp_13, out_put_temp_7_exp_13 = compute_experiment(n_components=7,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)

filter = ('2017-04-01', '2017-04-30')
output_name = 'localidades_2017_4'
out_put_ic_3_exp_14, out_put_temp_3_exp_14 = compute_experiment(n_components=3,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_4_exp_14, out_put_temp_4_exp_14 = compute_experiment(n_components=4,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_5_exp_14, out_put_temp_5_exp_14 = compute_experiment(n_components=5,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_7_exp_14, out_put_temp_7_exp_14 = compute_experiment(n_components=7,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)

filter = ('2017-05-01', '2017-05-31')
output_name = 'localidades_2017_5'
out_put_ic_3_exp_15, out_put_temp_3_exp_15 = compute_experiment(n_components=3,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_4_exp_15, out_put_temp_4_exp_15 = compute_experiment(n_components=4,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_5_exp_15, out_put_temp_5_exp_15 = compute_experiment(n_components=5,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_7_exp_15, out_put_temp_7_exp_15 = compute_experiment(n_components=7,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)

filter = ('2017-06-01', '2017-06-30')
output_name = 'localidades_2017_6'
out_put_ic_3_exp_16, out_put_temp_3_exp_16 = compute_experiment(n_components=3,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_4_exp_16, out_put_temp_4_exp_16 = compute_experiment(n_components=4,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_5_exp_16, out_put_temp_5_exp_16 = compute_experiment(n_components=5,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_7_exp_16, out_put_temp_7_exp_16 = compute_experiment(n_components=7,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)

filter = ('2017-07-01', '2017-07-31')
output_name = 'localidades_2017_7'
out_put_ic_3_exp_17, out_put_temp_3_exp_17 = compute_experiment(n_components=3,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_4_exp_17, out_put_temp_4_exp_17 = compute_experiment(n_components=4,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_5_exp_17, out_put_temp_5_exp_17 = compute_experiment(n_components=5,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_7_exp_17, out_put_temp_7_exp_17 = compute_experiment(n_components=7,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)

filter = ('2017-08-01', '2017-08-31')
output_name = 'localidades_2017_8'
out_put_ic_3_exp_18, out_put_temp_3_exp_18 = compute_experiment(n_components=3,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_4_exp_18, out_put_temp_4_exp_18 = compute_experiment(n_components=4,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_5_exp_18, out_put_temp_5_exp_18 = compute_experiment(n_components=5,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_7_exp_18, out_put_temp_7_exp_18 = compute_experiment(n_components=7,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)

filter = ('2017-09-01', '2017-09-30')
output_name = 'localidades_2017_9'
out_put_ic_3_exp_19, out_put_temp_3_exp_19 = compute_experiment(n_components=3,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_4_exp_19, out_put_temp_4_exp_19 = compute_experiment(n_components=4,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_5_exp_19, out_put_temp_5_exp_19 = compute_experiment(n_components=5,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)
out_put_ic_7_exp_19, out_put_temp_7_exp_19 = compute_experiment(n_components=7,
                                                                spatial_scale_geometry=spatial_scale_geometry,
                                                                temporal_filter=filter, output_name=output_name)

filter = ('2017-10-01', '2017-10-31')
output_name = 'localidades_2017_10'
out_put_ic_3_exp_110, out_put_temp_3_exp_110 = compute_experiment(n_components=3,
                                                                  spatial_scale_geometry=spatial_scale_geometry,
                                                                  temporal_filter=filter, output_name=output_name)
out_put_ic_4_exp_110, out_put_temp_4_exp_110 = compute_experiment(n_components=4,
                                                                  spatial_scale_geometry=spatial_scale_geometry,
                                                                  temporal_filter=filter, output_name=output_name)
out_put_ic_5_exp_110, out_put_temp_5_exp_110 = compute_experiment(n_components=5,
                                                                  spatial_scale_geometry=spatial_scale_geometry,
                                                                  temporal_filter=filter, output_name=output_name)
out_put_ic_7_exp_110, out_put_temp_7_exp_110 = compute_experiment(n_components=7,
                                                                  spatial_scale_geometry=spatial_scale_geometry,
                                                                  temporal_filter=filter, output_name=output_name)

filter = ('2017-11-01', '2017-11-30')
output_name = 'localidades_2017_11'
out_put_ic_3_exp_111, out_put_temp_3_exp_111 = compute_experiment(n_components=3,
                                                                  spatial_scale_geometry=spatial_scale_geometry,
                                                                  temporal_filter=filter, output_name=output_name)
out_put_ic_4_exp_111, out_put_temp_4_exp_111 = compute_experiment(n_components=4,
                                                                  spatial_scale_geometry=spatial_scale_geometry,
                                                                  temporal_filter=filter, output_name=output_name)
out_put_ic_5_exp_111, out_put_temp_5_exp_111 = compute_experiment(n_components=5,
                                                                  spatial_scale_geometry=spatial_scale_geometry,
                                                                  temporal_filter=filter, output_name=output_name)
out_put_ic_7_exp_111, out_put_temp_7_exp_111 = compute_experiment(n_components=7,
                                                                  spatial_scale_geometry=spatial_scale_geometry,
                                                                  temporal_filter=filter, output_name=output_name)

filter = ('2017-12-01', '2017-12-31')
output_name = 'localidades_2017_12'
out_put_ic_3_exp_112, out_put_temp_3_exp_112 = compute_experiment(n_components=3,
                                                                  spatial_scale_geometry=spatial_scale_geometry,
                                                                  temporal_filter=filter, output_name=output_name)
out_put_ic_4_exp_112, out_put_temp_4_exp_112 = compute_experiment(n_components=4,
                                                                  spatial_scale_geometry=spatial_scale_geometry,
                                                                  temporal_filter=filter, output_name=output_name)
out_put_ic_5_exp_112, out_put_temp_5_exp_112 = compute_experiment(n_components=5,
                                                                  spatial_scale_geometry=spatial_scale_geometry,
                                                                  temporal_filter=filter, output_name=output_name)
out_put_ic_7_exp_112, out_put_temp_7_exp_112 = compute_experiment(n_components=7,
                                                                  spatial_scale_geometry=spatial_scale_geometry,
                                                                  temporal_filter=filter, output_name=output_name)

print('IC consistency across months')

print(compare_ica_results(out_put_ic_3_exp_11,
                          out_put_ic_3_exp_12,
                          out_put_ic_3_exp_13,
                          out_put_ic_3_exp_14,
                          out_put_ic_3_exp_15,
                          out_put_ic_3_exp_16,
                          out_put_ic_3_exp_17,
                          out_put_ic_3_exp_18,
                          out_put_ic_3_exp_19,
                          out_put_ic_3_exp_110,
                          out_put_ic_3_exp_111,
                          out_put_ic_3_exp_112
                          ))

print(compare_ica_results(out_put_ic_4_exp_11,
                          out_put_ic_4_exp_12,
                          out_put_ic_4_exp_13,
                          out_put_ic_4_exp_14,
                          out_put_ic_4_exp_15,
                          out_put_ic_4_exp_16,
                          out_put_ic_4_exp_17,
                          out_put_ic_4_exp_18,
                          out_put_ic_4_exp_19,
                          out_put_ic_4_exp_110,
                          out_put_ic_4_exp_111,
                          out_put_ic_4_exp_112
                          ))

print(compare_ica_results(out_put_ic_5_exp_11,
                          out_put_ic_5_exp_12,
                          out_put_ic_5_exp_13,
                          out_put_ic_5_exp_14,
                          out_put_ic_5_exp_15,
                          out_put_ic_5_exp_16,
                          out_put_ic_5_exp_17,
                          out_put_ic_5_exp_18,
                          out_put_ic_5_exp_19,
                          out_put_ic_5_exp_110,
                          out_put_ic_5_exp_111,
                          out_put_ic_5_exp_112
                          ))

print(compare_ica_results(out_put_ic_7_exp_11,
                          out_put_ic_7_exp_12,
                          out_put_ic_7_exp_13,
                          out_put_ic_7_exp_14,
                          out_put_ic_7_exp_15,
                          out_put_ic_7_exp_16,
                          out_put_ic_7_exp_17,
                          out_put_ic_7_exp_18,
                          out_put_ic_7_exp_19,
                          out_put_ic_7_exp_110,
                          out_put_ic_7_exp_111,
                          out_put_ic_7_exp_112
                          ))
# spatial_scale_geometry = '/home/jrudascas/Downloads/upla/UPla.shp'
# output_name = 'up_2017'
# compute_experiment(n_components=3, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
# compute_experiment(n_components=4, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
# compute_experiment(n_components=5, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
# compute_experiment(n_components=7, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
# compute_experiment(n_components=9, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)


# spatial_scale_geometry = '/home/jrudascas/Downloads/barriolegalizado/BarrioLegalizado.shp'
# output_name = 'barrios_legalizados_2018'
# compute_experiment(n_components=3, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
# compute_experiment(n_components=4, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
# compute_experiment(n_components=5, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
# compute_experiment(n_components=7, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
# compute_experiment(n_components=9, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)


# spatial_scale_geometry = '/home/jrudascas/Downloads/dshapemanz12/shapemanz12/manz12.shp'
# output_name = 'manzanas_2018'
# compute_experiment(n_components=3, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
# compute_experiment(n_components=4, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
# compute_experiment(n_components=5, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
# compute_experiment(n_components=7, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
# compute_experiment(n_components=9, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name##)
