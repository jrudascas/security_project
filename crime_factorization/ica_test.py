import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point, Polygon
import numpy as np
import matplotlib

font = {'size':3}
matplotlib.rc('font', **font)
plt.axis('off')
plt.xticks([])
plt.yticks([])


def compute_experiment(n_components, spatial_scale_geometry, output_name):
    latitude_field_name = 'LATITUD'
    longitude_field_name = 'LONGITUD'
    date_field_name = 'FECHA'
    min_date, max_date = '2017-01-01', '2017-12-31'
    figure_name = output_name + '_IC_' + str(n_components)
    df = pd.read_csv('/home/jrudascas/Downloads/verify_enrich_nuse_29112019.csv')
    boros = GeoDataFrame.from_file(spatial_scale_geometry)

    gdf = GeoDataFrame(df.drop([latitude_field_name, longitude_field_name], axis=1),
                       geometry=[Point(xy) for xy in zip(df[longitude_field_name], df[latitude_field_name])])
    gdf['DATE'] = pd.to_datetime(gdf[date_field_name]).dt.strftime('%Y%m%d')
    #gdf = gdf[:1000]

    in_map_by_geometry = np.array([gdf.geometry.within(geom) for geom in boros.geometry])

    print(in_map_by_geometry.shape)

    raw = []

    date_sequence = [d.strftime('%Y%m%d') for d in pd.date_range(min_date, max_date, freq='D')]

    for pos, val in enumerate(boros.geometry):
        gdf_by_geometry = gdf[in_map_by_geometry[pos]]
        gdf_by_geometry_grouped = pd.DataFrame(gdf_by_geometry.groupby(['DATE']).size(), columns=["EVENTS"]).sort_index(
            by=['DATE'])

        values_fitted = []
        for i, value in enumerate(date_sequence):
            values_fitted.append(
                0 if value not in gdf_by_geometry_grouped.index.values else gdf_by_geometry_grouped.loc[value]['EVENTS'])

        raw.append(values_fitted)

    np_raw = np.array(raw)

    ica = FastICA(n_components=n_components)
    S_ = ica.fit_transform(np_raw)
    A_ = ica.mixing_

    print(S_.shape)
    print(A_.shape)

    s_gdf = GeoDataFrame(pd.DataFrame(S_, columns=['C' + str(i + 1) for i in range(n_components)]), geometry=boros.geometry)
    #a_gdf = GeoDataFrame(pd.DataFrame(A_, columns=list(range(1, n_components + 1))), geometry=boros.geometry)

    f, ax = plt.subplots(1, n_components)

    for i in range(n_components):
        s_gdf.plot(ax=ax[i],  cmap='RdBu', column='C' + str(i + 1), legend=True, vmin=0.0, vmax = 0.5)
        ax[i].set_title('IC ' + str(i + 1))

    plt.savefig(figure_name, dpi=700)
    plt.close()


spatial_scale_geometry = '/home/jrudascas/Downloads/locashp/Loca.shp'
output_name = 'localidades_2017'
compute_experiment(n_components=3, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
compute_experiment(n_components=4, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
compute_experiment(n_components=5, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
compute_experiment(n_components=7, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
#compute_experiment(n_components=9, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)


spatial_scale_geometry = '/home/jrudascas/Downloads/upla/UPla.shp'
output_name = 'up_2017'
compute_experiment(n_components=3, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
compute_experiment(n_components=4, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
compute_experiment(n_components=5, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
compute_experiment(n_components=7, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
#compute_experiment(n_components=9, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)


#spatial_scale_geometry = '/home/jrudascas/Downloads/barriolegalizado/BarrioLegalizado.shp'
#output_name = 'barrios_legalizados_2018'
#compute_experiment(n_components=3, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
#compute_experiment(n_components=4, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
#compute_experiment(n_components=5, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
#compute_experiment(n_components=7, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
#compute_experiment(n_components=9, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)


#spatial_scale_geometry = '/home/jrudascas/Downloads/dshapemanz12/shapemanz12/manz12.shp'
#output_name = 'manzanas_2018'
#compute_experiment(n_components=3, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
#compute_experiment(n_components=4, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
#compute_experiment(n_components=5, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
#compute_experiment(n_components=7, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name)
#compute_experiment(n_components=9, spatial_scale_geometry=spatial_scale_geometry, output_name=output_name##)