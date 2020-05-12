import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point, Polygon
import numpy as np
import matplotlib
from scipy.stats import pearsonr
from utils import fast_abs_percentile

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


def compute_experiment(n_components, spatial_scale_geometry, temporal_filter, output_name, threshold=0.1):
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
    #gdf = gdf[:1000]

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

    ica = FastICA(n_components=n_components, random_state=1)
    spatial_ic_ = ica.fit_transform(np_raw)
    temporal_ic = ica.mixing_

    # print(spatial_ic_.shape)
    # print(temporal_ic.shape)

    spatial_ic_ = np.abs(spatial_ic_)
    spatial_ic_[np.where(spatial_ic_ < fast_abs_percentile(spatial_ic_))] = 0

    s_gdf = GeoDataFrame(pd.DataFrame(spatial_ic_, columns=['C' + str(i + 1) for i in range(n_components)]),
                         geometry=boros.geometry)

    f, ax = plt.subplots(1, n_components)

    for i in range(n_components):
        s_gdf.plot(ax=ax[i], cmap='Reds', column='C' + str(i + 1), legend=True, vmin=0, vmax=0.5)
        ax[i].set_title('IC ' + str(i + 1))

    plt.savefig(figure_name, dpi=700)
    plt.close()

    return spatial_ic_, temporal_ic


spatial_scale_geometry = '/home/jrudascas/Downloads/locashp/Loca.shp'

temporal_filters_years = [2014, 2015, 2016, 2017, 2018]

results = {}
for year in temporal_filters_years:
    filter = (str(year) + '-01-01', str(year) + '-12-31')
    output_name = 'localidades_' + str(year)
    print(filter)

    print('IC # 3')
    out_put_ic_3_exp, out_put_temp_3_exp = compute_experiment(n_components=3,
                                                              spatial_scale_geometry=spatial_scale_geometry,
                                                              temporal_filter=filter, output_name=output_name)

    print('IC # 4')
    out_put_ic_4_exp, out_put_temp_4_exp = compute_experiment(n_components=4,
                                                              spatial_scale_geometry=spatial_scale_geometry,
                                                              temporal_filter=filter, output_name=output_name)

    print('IC # 5')
    out_put_ic_5_exp, out_put_temp_5_exp = compute_experiment(n_components=5,
                                                              spatial_scale_geometry=spatial_scale_geometry,
                                                              temporal_filter=filter, output_name=output_name)

    print('IC # 7')
    out_put_ic_7_exp, out_put_temp_7_exp = compute_experiment(n_components=7,
                                                              spatial_scale_geometry=spatial_scale_geometry,
                                                              temporal_filter=filter, output_name=output_name)

    results[year] = (out_put_ic_3_exp, out_put_ic_4_exp, out_put_ic_5_exp, out_put_ic_7_exp)

print('IC consistency across years')

print(compare_ica_results(results[2014][0], results[2015][0], results[2016][0], results[2017][0], results[2018][0]))
print(compare_ica_results(results[2014][1], results[2015][1], results[2016][1], results[2017][1], results[2018][1]))
print(compare_ica_results(results[2014][2], results[2015][2], results[2016][2], results[2017][2], results[2018][2]))
print(compare_ica_results(results[2014][3], results[2015][3], results[2016][3], results[2017][3], results[2018][3]))

temporal_filters_months = [('-01-01', '-01-31'),
                           ('-02-01', '-02-28'),
                           ('-03-01', '-03-31'),
                           ('-04-01', '-04-30'),
                           ('-05-01', '-05-31'),
                           ('-06-01', '-06-30'),
                           ('-07-01', '-07-31'),
                           ('-08-01', '-08-31'),
                           ('-09-01', '-09-30'),
                           ('-10-01', '-10-31'),
                           ('-11-01', '-11-30'),
                           ('-12-01', '-12-31')]

results_months = {}

for year in temporal_filters_years:
    month_list = []
    for i, month in enumerate(temporal_filters_months):
        filter = (str(year) + month[0], str(year) + month[1])
        output_name = 'localidades_' + str(year) + ' _' + str(i)
        print(filter)
        out_put_ic_3_exp, out_put_temp_3_exp = compute_experiment(n_components=3,
                                                                  spatial_scale_geometry=spatial_scale_geometry,
                                                                  temporal_filter=filter, output_name=output_name)
        out_put_ic_4_exp, out_put_temp_4_exp = compute_experiment(n_components=4,
                                                                  spatial_scale_geometry=spatial_scale_geometry,
                                                                  temporal_filter=filter, output_name=output_name)
        out_put_ic_5_exp, out_put_temp_5_exp = compute_experiment(n_components=5,
                                                                  spatial_scale_geometry=spatial_scale_geometry,
                                                                  temporal_filter=filter, output_name=output_name)
        out_put_ic_7_exp, out_put_temp_7_exp = compute_experiment(n_components=7,
                                                                  spatial_scale_geometry=spatial_scale_geometry,
                                                                  temporal_filter=filter, output_name=output_name)
        month_list.append((out_put_ic_3_exp, out_put_ic_4_exp, out_put_ic_5_exp, out_put_ic_7_exp))
        results_months[year] = month_list

print('IC consistency across months')

for i in range(12):
    print('Month # ' + str(i))
    print(compare_ica_results(results_months[2014][i][0], results_months[2015][i][0], results_months[2016][i][0],
                              results_months[2017][i][0], results_months[2018][i][0]))

    print(compare_ica_results(results_months[2014][i][1], results_months[2015][i][1], results_months[2016][i][1],
                              results_months[2017][i][1], results_months[2018][i][1]))

    print(compare_ica_results(results_months[2014][i][2], results_months[2015][i][2], results_months[2016][i][2],
                              results_months[2017][i][2], results_months[2018][i][2]))

    print(compare_ica_results(results_months[2014][i][3], results_months[2015][i][3], results_months[2016][i][3],
                              results_months[2017][i][3], results_months[2018][i][3]))

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
