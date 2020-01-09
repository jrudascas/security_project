import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA, PCA
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point, Polygon
import numpy as np

n_components = 4
df = pd.read_csv('/home/jrudascas/Downloads/verify_enrich_nuse_29112019.csv')
gdf = GeoDataFrame(df.drop(['LATITUD', 'LONGITUD'], axis=1),
                   geometry=[Point(xy) for xy in zip(df.LONGITUD, df.LATITUD)])
gdf['DATE'] = pd.to_datetime(gdf['FECHA']).dt.strftime('%Y%m%d')
#gdf = gdf[:1000]

boros = GeoDataFrame.from_file('/home/jrudascas/Downloads/locashp/Loca.shp')
# xmin, xmax, ymin, ymax = -74.0, -74.5, 2, 6
# resolution = 100

# print(np.linspace(xmin, xmax, resolution).shape)
# print(np.linspace(ymin, ymax, resolution).shape)

# xx, yy = np.meshgrid(np.linspace(xmin, xmax, resolution), np.linspace(ymin, ymax, 1000))
# xc = xx.flatten()
# yc = yy.flatten()
# in_map = np.array([pts.within(geom) for geom in boros.geometry]).sum(axis=0)

in_map_by_geometry = np.array([gdf.geometry.within(geom) for geom in boros.geometry])

print(in_map_by_geometry.shape)

raw = []

date_sequence = [d.strftime('%Y%m%d') for d in pd.date_range('2018-01-01', '2018-12-31', freq='D')]

for pos, val in enumerate(boros.geometry):
    gdf_by_geometry = gdf[in_map_by_geometry[pos]]
    gdf_by_geometry_grouped = pd.DataFrame(gdf_by_geometry.groupby(['DATE']).size(), columns=["EVENTS"]).sort_index(
        by=['DATE'])

    #gdf_by_geometry_grouped.set_index(['DATE'])
    values_fitted = []
    for i, value in enumerate(date_sequence):
        values_fitted.append(
            0 if value not in gdf_by_geometry_grouped.index.values else gdf_by_geometry_grouped.loc[value]['EVENTS'])

    raw.append(values_fitted)

    #    print('Geometry ', pos + 1, ' Len: ', len(gdf_by_geometry_grouped['EVENTS'].tolist()))

np_raw = np.array(raw)
print(np_raw.shape)
ica = FastICA(n_components=n_components)
S_ = ica.fit_transform(np_raw)
A_ = ica.mixing_

print(S_.shape)
print(A_.shape)

s_gdf = GeoDataFrame(pd.DataFrame(S_, columns=['C1', 'C2', 'C3', 'C4', 'C5']), geometry=boros.geometry)
a_gdf = GeoDataFrame(pd.DataFrame(A_, columns=list(range(1, n_components + 1))), geometry=boros.geometry)

print(s_gdf.head())
# pts = GeoSeries([val for pos, val in enumerate(pts) if in_map[pos]])

f, ax = plt.subplots(1, n_components)

for i in range(n_components):
    s_gdf.plot(ax=ax[i],  cmap='RdBu', column='C' + str(i + 1), legend=True)
    ax[i].set_title('Component # ' + str(i + 1))

plt.show()
# boros.plot(ax=ax)
# pts.plot(marker='*', color='red', alpha = 0.4, markersize=1, ax = ax)


# plt.show()
# print(X.shape)
# ica = FastICA(n_components=3)
# S_ = ica.fit_transform(X)
# A_ = ica.mixing_

