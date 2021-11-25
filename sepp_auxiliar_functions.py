__name__ = "Utilities SEPP"
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import geopandas as gpd
import gmaps
import gmaps.datasets
import random 
import timeit
import datetime
from scipy.optimize import fsolve
from scipy.spatial.distance import pdist
from shapely import wkt
from pyproj import Proj
from shapely.geometry import Point, LineString, Polygon
from shapely.geometry import MultiPoint, MultiLineString, MultiPolygon, box
from shapely.ops import cascaded_union
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
from pyproj import Proj, transform
from pyproj import Transformer
from collections import namedtuple
import fiona

import pandas_profiling
from IPython.display import display, HTML
import plotly.express as px
import plotly.graph_objects as go
import calendar
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)  
from wordcloud import WordCloud, STOPWORDS
from folium import plugins
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import logging
import time
import pyspark
from pyspark.sql import SparkSession
from datetime import datetime, date
from pyspark.sql import Row
import shutup 
shutup.please()

def cells_on_map(b1, h1):
    """
    Calculate the cells (rectangle) of the grid on any localidad or upz. 
    b1: base lenght
    h1: height lenght
    setting = 1: loc
    setting = 2: upz
    num: setting (1 or 2) number
    """
    # Externals Bogota map points
    point1 = Point(-74.2235814249999, 4.836779094841296)
    point2 = Point(-73.98653427799991, 4.836779094841296)
    point3 = Point(-73.98653427799991, 4.269664096859796)
    point4 = Point(-74.2235814249999, 4.269664096859796)

    # EPSG to which we are going to project (32718 also works)
    nys = Proj(init='EPSG:3857') 

    # Points projected in the new EPSG
    p1_proj = nys(point1.x, point1.y)
    p2_proj = nys(point2.x, point2.y)
    p3_proj = nys(point3.x, point3.y)
    p4_proj = nys(point4.x, point4.y)

    # Length of the base and the height of the grid
    longitud_base = Point(p1_proj).distance(Point(p2_proj))
    longitud_altura = Point(p2_proj).distance(Point(p3_proj))

    # External grid where we locate the extreme points of the Bogota map
    topLeft = p1_proj
    topRight = Point(Point(p1_proj).x + math.ceil(longitud_base/b1)*b1, Point(p2_proj).y)
    bottomRight = Point(topRight.x, Point(p2_proj).y - math.ceil(longitud_altura/h1)*h1)
    bottomLeft = Point(Point(p1_proj).x, bottomRight.y) 

    # Convertion the grid to polygon
    poligono_mayor = Polygon([topLeft, topRight, bottomRight, bottomLeft])

    # Columns and rows of the grid
    cols = np.linspace(np.array(bottomLeft)[0], np.array(bottomRight)[0], math.ceil(longitud_base/b1) + 1) 
    rows = np.linspace(np.array(topLeft)[1], np.array(bottomLeft)[1], math.ceil(longitud_altura/h1) + 1)     

    # Polygons of the cells that make up the grid
    poligonos = [Polygon([Point(cols[i], rows[j]), Point(cols[i+1], rows[j]), Point(cols[i+1], rows[j+1]), 
                          Point(cols[i], rows[j+1]), Point(cols[i], rows[j])]) for i in range(len(cols)-1) 
                 for j in range(len(rows)-1)]

    poligonos_series = gpd.GeoSeries(poligonos)

    poligonos_df = gpd.GeoDataFrame({'geometry': poligonos})
    ###poligonos_df['cells'] = poligonos_df.index
    grid = MultiPolygon(poligonos)

    # Mapa de las localidad de Bogotá (excluimos Sumapaz)
    boundary_bogota = gpd.read_file('poligonos-localidades.geojson')
    boundary_bogota.crs = "EPSG:4326"
    boundary_bogota = boundary_bogota.to_crs("EPSG:3857")

    array_index = np.array([])            # Arreglo en el que almacenamos los poligonos que estan sobre bogota

    for i in range(0,len(poligonos_df)):
        if poligonos_df.geometry.iloc[i].intersects(boundary_bogota.geometry.iloc[0]) == True:
            #print(i)
            poligonos_df.geometry.iloc[i] = poligonos_df.geometry.iloc[i].intersection(boundary_bogota.geometry.iloc[0])
            array_index = np.append(array_index,i)

    array_index = array_index.astype('int')
    poligonos_df = poligonos_df.loc[array_index]
    poligonos_df = poligonos_df.reset_index()
    poligonos_df.drop(columns='index',inplace=True)
            
                                    #######################################
                                    ###### Covariados de las celdas  ######
                                    #######################################

    ########################## DataFrame de Manzanas-Estratificación 2019 #########################
    #Son unidades geográficas tipo manzana a las cuales se les asocia la variable de estrato socioeconómico, siendo esta, 
    #la clasificación de los inmuebles residenciales que deben recibir servicios públicos. Se realiza principalmente para 
    #cobrar de manera diferencial por estratos los servicios públicos domiciliarios, permitiendo asignar subsidios y cobrar 
    #contribuciones. (# 3982 # 24462 # 43310 # 43819 # 43971  PRESENTAN PROBLEMAS EN LA TOPOLOGÍA)
    estrat_df = gpd.read_file('estrat_df.geojson')
    estrat_df = estrat_df.to_crs("EPSG:3857")
    poligonos_df['Promedio Estrato 2019'] = ''

    #########################  DATAFRAME DE CUADRANTES DE POLICÍA 2020 ######################## 
    # Es un sector geográfico fijo, que a partir de sus características sociales, demográficas y geográficas, recibe distintos tipos
    # de atención de servicio policial, entre los cuales se cuentan la prevención, la disuasión, control de delitos y contravenciones 
    # y la educación ciudadana en seguridad y convivencia.
    cuadr_pol_df = gpd.read_file('cuadrantes_policia.geojson')
    cuadr_pol_df = cuadr_pol_df.to_crs("EPSG:3857") 
    poligonos_df['Area de Cuadrantes de Policia 2020'] = 0
     
    #########################  DATAFRAME DE COMANDO DE ATENCION INMEDIATA ######################## 
    # Es un sector geográfico fijo, que a partir de sus características sociales, demográficas y geográficas, recibe distintos tipos
    # de atención de servicio policial, entre los cuales se cuentan la prevención, la disuasión, control de delitos y contravenciones 
    # y la educación ciudadana en seguridad y convivencia.
    comando_atencion_df = gpd.read_file('comando_atencion_inmediata.geojson')
    comando_atencion_df = comando_atencion_df.to_crs("EPSG:3857") 
    poligonos_df['Comando de Atencion Inmediata'] = ''

    #########################  DATAFRAME DE ESTACIONES DE POLICIA ######################## 
    # Es un sector geográfico fijo, que a partir de sus características sociales, demográficas y geográficas, recibe distintos tipos
    # de atención de servicio policial, entre los cuales se cuentan la prevención, la disuasión, control de delitos y contravenciones 
    # y la educación ciudadana en seguridad y convivencia.
    estaciones_policia_df = gpd.read_file('estaciones_policia.geojson')
    estaciones_policia_df = estaciones_policia_df.to_crs("EPSG:3857") 
    poligonos_df['Estaciones Policia'] = ''

                                    #########################################
                                    ###### Poligonos y sus covariados  ######
                                    #########################################
    
    for i in range(0, len(poligonos_df)):

        ############################### PROMEDIO DE ESTRATO 2019 ################################## 
        array1 = np.array([])
        array2 = np.array([])
        for j in range (0, len(estrat_df)):
        
            if  poligonos_df['geometry'][i].intersects(estrat_df['geometry'][j]) == True:
                array1 = np.append(array1, poligonos_df['geometry'][i].intersection(estrat_df.geometry[j]).area*estrat_df['ESTRATO'][j])
                array2 = np.append(array2, poligonos_df['geometry'][i].intersection(estrat_df.geometry[j]).area)
    
        if len(array2) != 0:
            poligonos_df.loc[i,'Promedio Estrato 2019'] = sum(array1)/sum(array2)
    
        else:
            poligonos_df.loc[i,'Promedio Estrato 2019'] = 0
    
        ############################### AREA DE CUADRANTES DE POLICIA 2021  ################################## 
        array1 = np.array([])
        for j in range (0, len(cuadr_pol_df)):
            if  poligonos_df['geometry'][i].intersects(cuadr_pol_df['geometry'][j]) == True:
                  array1 = np.append(array1, poligonos_df['geometry'][i].intersection(cuadr_pol_df.geometry[j]).area)
        poligonos_df.loc[i, 'Area de Cuadrantes de Policia 2020'] = sum(array1)
	     
        ############################### COMANDO DE ATENCION INMEDIATA 2021 ##################################          
        array1 = np.array([])
        for j in range (0, len(comando_atencion_df)):
            if  poligonos_df['geometry'][i].contains(comando_atencion_df['geometry'][j]) == True:
                  array1 = np.append(array1, j)
        poligonos_df.loc[i, 'Comando de Atencion Inmediata'] = len(array1)
     
        ############################### ESTACIONES DE POLICIA 2021 ##################################          
        array1 = np.array([])
        for j in range (0, len(estaciones_policia_df)):
            if  poligonos_df['geometry'][i].contains(estaciones_policia_df['geometry'][j]) == True:
                  array1 = np.append(array1, j)
        poligonos_df.loc[i, 'Estaciones Policia'] = len(array1)
	

    poligonos_df = gpd.GeoDataFrame(poligonos_df, crs="EPSG:3857", geometry=poligonos_df.geometry)
    poligonos_df['Int'] = 1
    for column in poligonos_df.columns[1:-1]:
        poligonos_df[column] = pd.to_numeric(poligonos_df[column])
        if poligonos_df[column].max() - poligonos_df[column].min()==0:
            pass
        else:
            poligonos_df[column] = (poligonos_df[column]-poligonos_df[column].min())/(poligonos_df[column].max()-poligonos_df[column].min())
    poligonos_df['cells'] = np.arange(0, len(poligonos_df)).astype('int')    
    print(poligonos_df.columns)
    poligonos_df.to_file("poligonos_covariados.geojson", driver='GeoJSON')
    
    return poligonos_df
                                        

                                        ####################################
                                        #####   Datos de los eventos   ##### 
                                        ####################################


def eventos_covariados(df_eventos, poligonos_covariados, fecha_inicial, fecha_final):    
    events_df = df_eventos
    poligonos_df = poligonos_covariados
    events_df = events_df[(events_df['FECHA'] >= str(fecha_inicial)) & (events_df['FECHA'] <= str(fecha_final))].sort_values(by=["FECHA"])
    events_df['FECHA'] = events_df['FECHA'].astype(str)
    def dejar_solo_cifras(txt):
        return "".join(c for c in txt if c.isdigit())
    def FECHA_mod(txt):
        return txt.replace("T"," ")
    events_df['FECHA'] = events_df['FECHA'].map(FECHA_mod)
    events_gdf = gpd.GeoDataFrame(events_df, geometry=gpd.points_from_xy(events_df.LONGITUD, events_df.LATITUD),crs="EPSG:3857")
    nys = Proj(init='EPSG:3857') 
    for i in range(0, len(events_gdf)):
        events_gdf['geometry'].iloc[i] = Point(nys(events_gdf['geometry'].iloc[i].x, events_gdf['geometry'].iloc[i].y)) 
    
    events_gdf['TimeStamp'] = ' '
    for i in range(0, len(events_gdf)):
        events_gdf.TimeStamp.iloc[i] = datetime.fromisoformat(str(events_gdf.FECHA.iloc[i])).timestamp() - datetime.fromisoformat('2021-01-01 00:00:00').timestamp()  # ---------> init date simulation
    events_gdf['TimeStamp'] = pd.to_numeric(events_gdf['TimeStamp'])/3600
            
    events_gdf = events_gdf.loc[:,['FECHA','TimeStamp','geometry']]
    poligonos_df = gpd.GeoDataFrame(poligonos_covariados, crs="EPSG:3857", geometry=poligonos_df.geometry)
    poligonos_df['Int'] = 1
    for column in poligonos_df.columns[1:-1]:
        poligonos_df[column] = pd.to_numeric(poligonos_df[column])
        if poligonos_df[column].max() - poligonos_df[column].min()==0:
            pass
        else:
            poligonos_df[column] = (poligonos_df[column]-poligonos_df[column].min())/(poligonos_df[column].max()-poligonos_df[column].min())
    poligonos_df['cells'] = np.arange(0, len(poligonos_df)).astype('int')
    
    events_gdf['cells'] = ' '
    for i in range(0, len(poligonos_df)):
        for j in range(0, len(events_gdf)):
            if poligonos_df.loc[i,'geometry'].contains(events_gdf.loc[j,'geometry']) == True:
                events_gdf.loc[j,'cells'] = int(i)

    events_gdf['X'] = events_gdf.geometry.x
    events_gdf['Y'] = events_gdf.geometry.y
    events_gdf = pd.merge(events_gdf, poligonos_df, on='cells')
    def FECHA_mod(txt):
        return txt.replace("T"," ")
    events_gdf['FECHA'] = events_gdf['FECHA'].map(FECHA_mod)
    events_gdf = events_gdf.sort_values(['FECHA']).reset_index().rename(columns={'geometry_x':'geometry'}).drop(columns={'geometry_y','index'})
    return events_gdf
   
######################################################################
######################################################################
######################################################################

#=====================
#  optimizacion beta 
#=====================

def syst(arg, *data):
    """
    set equation to solve and find the beta optimal value
    
    :param arg: variable of the set equation
    :param *data: external parameters
                  T: temporal window
                  sc: area of the cells
                  sp_factor: spatial factor of the data
                  cov_norm_cell_m: covariate matrix of each cell
                  pbe: prob background events
    :return equations: equations to solve (array)
    """
    beta = arg
    T, sc, cov_norm_cell_m, cov_norm_eventos_m, pbe = data
    equations = np.array([])
    for i in range(0, len(beta)):
        equations = np.append(equations, np.dot(pbe, cov_norm_eventos_m[:,i]) 
                              - T * sc * np.dot(cov_norm_cell_m[:,i], 
                                np.exp((beta * cov_norm_cell_m).sum(axis=1))))
    return equations
   
def root_beta(beta, data):
    """
    Find the optimal value of beta:
    :param beta: initial condition to find the beta optimal value
    :param data: external parameters
                  T: temporal window
                  sc: area of the cells
                  sp_factor: spatial factor of the data
                  cov_norm_cell_m: covariate matrix of each cell
                  pbe: prob background events
    :return _: the beta optimal value
    """
    return fsolve(syst, beta, args=data, xtol=10e-5, maxfev=1000000)

#=======================
#  optimizacion sigma2  
#=======================

def update_sigma21(sigma2, p_ij, pte, d2_ev):
    """
    Find the sigma optimal value
    
    :param sigma2: inital value of sigma2
    :param p_ij: prob event i is triggered from event j
    :param pte: prob event i is triggered from some event j
    :param d2_ev: squared matrix with the squared distance between events
    :return sigma2: sigma2 optimal value
    """
    num_sigma2 = 0
    for i in range(0, len(pte)):
        for j in range (0, len(pte)):
            num_sigma2 += p_ij[i,j]*d2_ev[i,j]
    sigma2 = 0.5*num_sigma2/sum(pte)
    return sigma2

#=======================
#  optimizacion  omega
#=======================

def funct_omega1(omega, tiempo_eventos, diferencia_tiempos, T, p_ij, pte):
    """
    Ecuación que debemos resolver para encontrar el vlor optimo de omega usando el metodo de punto fijo
    
    :param omega: omega
    :param tiempo_eventos: arreglo de los tiempos de los eventos
    :param diferencia_tiempos: matriz de la diferencia de los tiempos de los eventos
    :param T: ventana temporal
    :param p_ij: probabilidad de que el evento i sea descencadenado por el evento j 
    :param pte: probabilidad de que el evento i sea un evento descencadenado 
    :return _: ecuacion que debemos resolver para omeva
    """
    term1, suma = 0, 0
    for i in range(0, len(tiempo_eventos)):
        t_diff = T-tiempo_eventos[i]
        suma += np.exp(-omega*t_diff) * t_diff
        for j in range(0, len(tiempo_eventos)):
            term1 += p_ij[i,j]*diferencia_tiempos[i,j]
    return sum(pte)/(suma+term1)

def new_omega1(omega, tiempo_eventos, diferencia_tiempos, T, p_ij, pte):
    """
    Solves the equation to find the optimal value of omega through fixed point
    
    :param omega: omega
    :param tiempo_eventos: arreglo de los tiempos de los eventos 
    :param T: ventana temporal
    :param p_ij: probabilidad de que el evento i sea descendiente de un evento j 
    :param pte: probabilidad de que el evento i sea un evento descencadenado
    :return omega: valor optimo de omega
    """
    while abs(omega-funct_omega1(omega, tiempo_eventos, diferencia_tiempos, T, p_ij, pte)) > 1e-5:
        omega = funct_omega1(omega, tiempo_eventos, diferencia_tiempos, T, p_ij, pte)
    return omega


def cov_join_events(gpd_events, poligonos_df_cov):
    """
    Join dataframe of the events with the polygons with its covariates
    
    Input
    par gpd_events: dataframe of the events (train_data for example)
    par poligonos_df_cov: dataframe with the geometry of each polygons
                      with the covariates associated to it.
    Output
    cov_norm_cell_m: array of covariates of each cell of grid
    cov_norm_eventos_m: array of covariates of each cell of grid where occured the event
    poligonos_df_cov: polygons with the covariates
    gpd_events: dataframe with the events with the column "cells" and covariate of the cells
    """
    gpd_events['cells'] = ' ' 
    for i in range(0, len(poligonos_df_cov)):
        for j in range(0, len(gpd_events)):
            if poligonos_df_cov.loc[i,'geometry'].contains(gpd_events.loc[j,'geometry']) == True:
                gpd_events.loc[j,'cells'] = int(i)
    gpd_events['X'] = gpd_events.geometry.x
    gpd_events['Y'] = gpd_events.geometry.y
    gpd_events = pd.merge(gpd_events, poligonos_df_cov, on='cells')
    gpd_events.TimeStamp = pd.to_numeric(gpd_events.TimeStamp)
    gpd_events = gpd_events.sort_values(['TimeStamp']).reset_index().rename(columns={'geometry_x':'geometry'}).drop(columns={'geometry_y','index'})
    cov_norm_cell_m = poligonos_df_cov[['Promedio Estrato 2019', 'Area de Cuadrantes de Policia 2020','Comando de Atencion Inmediata','Estaciones Policia', 'Int']].to_numpy()
    cov_norm_eventos_m = gpd_events[['Promedio Estrato 2019', 'Area de Cuadrantes de Policia 2020', 'Comando de Atencion Inmediata', 'Estaciones Policia', 'Int']].to_numpy()
    
    gpd_events = gpd_events[['TimeStamp','geometry','cells','X','Y']]
    
    return cov_norm_cell_m, cov_norm_eventos_m, poligonos_df_cov, gpd_events

    
######################################
# Data for training and test process #
######################################

def training_data(init_data, final_data, gpd_events):
    '''
    Gets the data for the training process

    :param init_data: start date of the training process
    :param final_data: end date of the training process
    :param gpd_events: siedo dataframe with all events in the locality or upz chosen
    :return: data for the training process 
    '''
    gpd_events_train = gpd_events[(gpd_events.FECHA>=init_data) & (gpd_events.FECHA<=final_data)]
    gpd_events_train = gpd_events_train.reset_index() 
    return gpd_events_train

def test_data(init_data, final_data, gpd_events):
    '''
    Gets the data for the test process
        
    :param init_data: start date of the test process
    :param final_data: end date of the test process
    :param gpd_events: siedo dataframe with all events in the locality or upz chosen
    :return: data for the test process 
    '''
    gpd_events_test = gpd_events[(gpd_events.FECHA>=init_data) & (gpd_events.FECHA<=final_data)]
    gpd_events_test = gpd_events_test.reset_index()
    return gpd_events_test



#########################################################################
# Array of number of events and number of cell where the events ocurred #
#########################################################################

def arr_cells_events_data(array_data_events, array_cells_events_sim):
    """
    Give an array with the number of events in each cell and the number of cell
    for array of events of dataset on polygons, with NUSE events, can be train or test events
    
    :param array_data_events: array of the test data events with the cell number 
    :param array_cells_events_sim: array of the simulated data events with the cell number 
    :return array_cells_events_data: array with the number of test events for cell and number of the cell 
    """
    array_data_events.cells = array_data_events.cells.astype(int) 
    array_cells_events_data_prev1 = []
    array_cells_events_data_prev2 = []
    array_cells_events_data_prev1 = array_data_events.cells.value_counts().sort_values().rename_axis('cell').reset_index(name='events').to_numpy()
    array_cells_events_data_prev1 = array_cells_events_data_prev1[array_cells_events_data_prev1[:,0].astype(int).argsort()]
    for i in range(0, len(array_cells_events_data_prev1)):
        array_cells_events_data_prev2.append([array_cells_events_data_prev1[:,0][i],array_cells_events_data_prev1[:,1][i]]) 
    list1 = [i[0] for i in array_cells_events_sim]
    list2 = [i[0] for i in array_cells_events_data_prev2]
    set_difference = set(list1) - set(list2)
    list_difference = list(set_difference)

    array_cells_events_data_prev3 = []
    for i in range(0, len(list_difference)):
        array_cells_events_data_prev3.append([list_difference[i], 0])
    
    merged_list = array_cells_events_data_prev2 + array_cells_events_data_prev3
    array_cells_events_data = sorted(merged_list, key=lambda x: x[0])
        
    return array_cells_events_data

###################################################################
# Filtering of data: we take only the events in the hotspot cells #
###################################################################

def filtering_data(percentage_area, array_cells_events_tst_data_1_cells, two_dim_array_pred_cell, events_gpd_pred, init_date):
    # array of tst_data_1_cells SORTED
    poly = gpd.read_file('poligonos_covariados.geojson')
    array_cells_events_tst_data_1_cells_sorted = sorted(array_cells_events_tst_data_1_cells, key=lambda x: x[1], reverse=True)
    # array of two_dim_array_pred_cell SORTED
    array_cell_events_pred_sorted = sorted(two_dim_array_pred_cell, key=lambda x: x[1], reverse=True)
    # number of cells according to the chosen percentage
    length = math.ceil(len(array_cells_events_tst_data_1_cells)*percentage_area/100)
    array_cells_hotspots_tsts_data_1 = array_cells_events_tst_data_1_cells_sorted[:length]
    array_cells_hotspots_tst_data_1_number_cell = list(np.array(array_cells_hotspots_tsts_data_1)[:,0])
    puntos_gdf_cells = cov_join_events(events_gpd_pred, poly)[3]
    puntos_gdf_cells_4326 = events_gpd_pred.loc[events_gpd_pred['cells'].isin(array_cells_hotspots_tst_data_1_number_cell)].to_crs("EPSG:4326")
    puntos_gdf_cells_4326['TimeStamp'] = pd.to_datetime(puntos_gdf_cells_4326['TimeStamp'], unit='h',origin=pd.Timestamp(init_date))
    puntos_gdf_cells_4326 = puntos_gdf_cells_4326[['TimeStamp','geometry']]
    puntos_gdf_cells_4326 = puntos_gdf_cells_4326.rename(columns={'TimeStamp':'Fecha'})
    #puntos_gdf_cells_4326['Fecha'] = puntos_gdf_cells_4326['Fecha'].to_pydatetime()
    def FECHA_mod(txt):
        return txt.replace("T"," ")
    def without_milisecond(txt):
        return txt[:19]
    def FECHA_mod1(date):
        return date.replace("T", " ") #.replace(".000Z", "")

    
    print(type(str(puntos_gdf_cells_4326.Fecha.iloc[0])))
    #puntos_gdf_cells_4326_2 = puntos_gdf_cells_4326
    
    #date_format_str = '%Y-%m-%d %H:%M:%S'
    for i in range(0, len(puntos_gdf_cells_4326)):
        puntos_gdf_cells_4326['Fecha'].iloc[i] = str(puntos_gdf_cells_4326['Fecha'].iloc[i]).replace("T"," ")
        puntos_gdf_cells_4326['Fecha'].iloc[i] = str(puntos_gdf_cells_4326['Fecha'].iloc[i])[:19]
    

        #puntos_gdf_cells_4326['Fecha'].iloc[i] = puntos_gdf_cells_4326['Fecha'].iloc[i].replace("T"," ")
    #puntos_gdf_cells_4326['Fecha'] = puntos_gdf_cells_4326['Fecha'].map(FECHA_mod)

    #print(type(puntos_gdf_cells_4326.Fecha.iloc[0]))

    #    puntos_gdf_cells_4326['Fecha'].iloc[i] = datetime.utcfromtimestamp(puntos_gdf_cells_4326['Fecha'].iloc[i]).strftime('%Y-%m-%d %H:%M:%S')

    #puntos_gdf_cells_4326['Fecha'] = datetime.strptime(puntos_gdf_cells_4326['Fecha'], date_format_str)
    #puntos_gdf_cells_4326['Fecha'] = datetime.fromisoformat(puntos_gdf_cells_4326['Fecha'])
    #datetime. strptime(date_time_str
    #print(type(puntos_gdf_cells_4326_2.Fecha.iloc[0]))

    #puntos_gdf_cells_4326_2['Fecha'] = puntos_gdf_cells_4326_2['Fecha'].map(without_milisecond)
    #print(type(puntos_gdf_cells_4326.Fecha.iloc[0]))
    print(puntos_gdf_cells_4326)
    puntos_gdf_cells_4326.to_file("predicted_events.geojson", driver='GeoJSON')
    #puntos_gdf_cells_4326.to_csv('predicted_events.csv', index=False)
    number_ev_pred_on_hotspots = np.array([])
    for i in range(0, len(array_cells_hotspots_tst_data_1_number_cell)):
        for j in range(0, len(array_cell_events_pred_sorted)):
            if array_cells_hotspots_tst_data_1_number_cell[i] == array_cell_events_pred_sorted[j][0]:
                number_ev_pred_on_hotspots = np.append(number_ev_pred_on_hotspots,array_cell_events_pred_sorted[j][1])
    number_ev_pred_on_hotspots
    total_ev = np.array([])
    for i in range(0, len(two_dim_array_pred_cell)):
        total_ev = np.append(total_ev, two_dim_array_pred_cell[i][1])
    total_ev = sum(total_ev)
    total_ev    
    return sum(number_ev_pred_on_hotspots), total_ev, puntos_gdf_cells_4326


def Data():
    """
    Gives the dataframe of events and polygons of cells with the covariates
    
    :return data: dataframe of events
    :return poly: polygons of cells with the covariates
    """
    columns = np.array(['geometry', 'Promedio Estrato 2019', 'Num de Estaciones de Policia 2020', 'cells', 'Int'])
    data = gpd.read_file("data_events.geojson")
    data.FECHA = pd.to_datetime(data.FECHA)
    poly = gpd.read_file("poligonos_gdf.geojson")
    poly = poly[columns]
    for column in poly.columns[1:-1]:
        poly[column] = pd.to_numeric(poly[column])
    
    return data, poly 

def time_distance(array_t):
    N  = len(array_t)                         # Cantidad de eventos
    TS = np.zeros(N*(N-1)//2)
    Ti = array_t
    n = 0
    for i in range(N-1):
        m = N-i-1
        TS[n:n+m] = Ti[i+1:N] - Ti[i]
        n = n+m
    return TS

def index_list(N):
    ndx = []
    for i in range(N):
        cn  = i
        inc = N-2
        ind = []
        for j in range(i):
            ind.append(cn-1)
            cn  += inc
            inc -= 1
        ndx.append(np.array(ind, dtype=np.int))
    return ndx

def get_index(Z, N):
    i = int((2*N-1-math.sqrt((2*N-1)**2 - 8*Z))/2)
    j = int(Z+1-i*(2*N-i-3)/2)
    return i, j

def iget_index(i, j, N):
    i, j = (j, i) if j<i else (i, j)
    return int(i*(2*N-i-3)/2 + j -1)

def dist2_ev(two_dim_arr_xy):
    return pdist(two_dim_arr_xy, 'sqeuclidean')/1e6

def prob(beta, omega, sigma2, tiempo_eventos, cov_norm_eventos_m, d2_ev):
    invSigma2  = -1/(2*sigma2)
    invOmega   = -omega
    fnormalz   = omega/(2*np.pi*sigma2)
    d_temp = time_distance(tiempo_eventos)
    ndx = index_list(len(tiempo_eventos))                    # Lista de índices para totalizar por filas
    rdx = np.array([j for i in range(1, len(tiempo_eventos)) 
                for j in range(i, len(tiempo_eventos))])
    print(beta)
    print(cov_norm_eventos_m.shape)
    print(np.exp((cov_norm_eventos_m*beta).sum(axis=1)))
    mu = np.array(np.exp((cov_norm_eventos_m*beta).sum(axis=1)))
    Gst = fnormalz * np.exp(invOmega*d_temp) * np.exp(invSigma2*d2_ev) 
    Gi  = [sum(Gst[ndx[i]]) for i in range(len(tiempo_eventos))]
    iL   = 1/(mu + Gi)
    Po   = mu * iL
    Pij  = Gst * iL[rdx]
    Pte  = np.ones(len(tiempo_eventos))- Po
    
    return Gst, mu, 1/iL, Pij, Pte, Po

def cleaning(data_initial, data_final):
    spark=SparkSession.builder.appName("Python Spark SQL basic example").config("spark.jars","postgresql-42.2.23.jar").config("spark.executor.memory", "70g").config("spark.driver.memory", "50g").config("spark.memory.offHeap.enabled",True).config("spark.memory.offHeap.size","16g") .getOrCreate()
    df=spark.read.format("jdbc").option("url","jdbc:postgresql://localhost:5432/analiticades").option("dbtable","data_nuse_rinas").option("user", "unal").option("password", "Unal123").option("driver", "org.postgresql.Driver").load()
    df = df.filter((df.fecha_hora >= data_initial) & (df.fecha_hora <= data_final))
    df_pandas = df.toPandas()
    df_pandas['fecha'] = pd.to_datetime(df_pandas['fecha'])
    df_pandas.columns= df_pandas.columns.str.upper()
    df_pandas = df_pandas.drop(['FECHA'], axis=1)
    df_pandas = df_pandas.rename(columns={'FECHA_HORA': 'FECHA'})
    df_pandas = df_pandas.sort_values(by='FECHA')
    df_pandas = df_pandas[df_pandas['TIPO_DETALLE'] == 'RIÑA']
    df_pandas.to_csv('merged_nuse.csv')    
    
    return df_pandas

def limpieza_datos(df):
    merged_nuse = df
    
    root_logger= logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # or whatever
    handler = logging.FileHandler('text.log', 'w', 'utf-8') # or whatever
    formatter = logging.Formatter('%(name)s %(message)s') # or whatever
    handler.setFormatter(formatter) # Pass handler as a parameter, not assign
    root_logger.addHandler(handler)


    merged_nuse.to_csv(r'/home/unal/modelo_rinas/UNAL/merged_nuse.csv',index=None)


    ##print('----- 1 -----')

    # 2. Rebuild missing data

    localidadCodDictionaryNuse = {1:'USAQUEN',
                                  2:'CHAPINERO',
                                  3:'SANTA FE',
                                  4:'SAN CRISTOBAL',
                                  5:'USME',
                                  6:'TUNJUELITO',
                                  7:'BOSA',
                                  8:'KENNEDY',
                                  9:'FONTIBON',
                                  10:'ENGATIVA',
                                  11:'SUBA',
                                  12:'BARRIOS UNIDOS',
                                  13:'TEUSAQUILLO',
                                  14:'LOS MARTIRES',
                                  15:'ANTONIO NARIÑO',
                                  16:'PUENTE ARANDA',
                                  17:'CANDELARIA',
                                  18:'RAFAEL URIBE URIBE',
                                  19:'CIUDAD BOLIVAR',
                                  20:'SUMAPAZ',
                                  99:'SIN LOCALIZACION'}

    ##print('----- 2 -----')

    # Methods to rebuild

    import import_ipynb
    from selenium import webdriver
    import ws_address
    from selenium.common.exceptions import TimeoutException
    import re
    import unidecode     

    def find_between( s, first, last ):
        logging.debug('Inicia find_between. Parametros: s={}, first={}, last={}'.format(s, first, last)) 
        try:
            start = s.index( first ) + len( first )
            end = s.index( last, start )
            return s[start:end]
        except ValueError:
            return ""
        logging.debug('Termina find_between')

    tags = ["Dirección ingresada: ","Dirección encontrada: ","Tipo dirección: ","Código postal: ","Sector catastral: ",
            "UPZ: ","Localidad: ","Latitud: ","Longitud: ","CHIP: "]
    def parse_address_ws(ws_result):
        logging.debug('Inicia parse_address_ws. Parametros: ws_result={}, first={}, last={}'.format(ws_result)) 
        try:
            location = {}
            for idx in range(len(tags)-1):
                location[tags[idx].replace(': ','')] = find_between(ws_result,tags[idx],tags[idx+1])
            return location
        except Exception as e:
            logging.error(e)
            pass
        logging.debug('Termina parse_address_ws')

    def assign_upz(original_df,index,UPZ_ws_field):
        logging.debug('Inicia assign_upz. Parametros: original_df={}, index={}, UPZ_ws_field={}'.format(original_df,index,UPZ_ws_field))
        try:
            original_df.at[index,'COD_UPZ'] = find_between(UPZ_ws_field, '(', ')')
            original_df.at[index,'UPZ'] = find_between(UPZ_ws_field, '', ' (')
        except Exception as e:
            logging.error(e)
            pass
        logging.debug('Termina assign_upz')

    def get_cod_localidad(localidad_name):
        logging.debug('Inicia get_cod_localidad. Parametros: localidad_name={} '.format(localidad_name))
        try:
            return [key  for (key, value) in localidadCodDictionaryNuse.items() if value == localidad_name][0]
        except Exception as e:
            logging.error(e)
            pass
        logging.debug('Termina get_cod_localidad')

    def rebuild_location_in_nuse(original_df, index, driver):
        logging.debug('Inicia rebuild_location_in_nuse. Parametros: original_df={}, index={}, driver={} '.format(original_df, index, driver))
        try:
            address = original_df.at[index,'STR_DIRECCION_INCIDENTE']
            result_ws = ws_address.web_scrap_address(driver,address)
            ws_address.delete_address(driver,address)
    
            if result_ws != "Not found":
                parsed_result = parse_address_ws(result_ws)
                print(parsed_result)
                if parsed_result["Dirección ingresada"] != address:
                    return "Error loading address"
                else:            
                    original_df.at[index,'LATITUD'] = float(parsed_result['Latitud'])
                    original_df.at[index,'LONGITUD'] = float(parsed_result['Longitud'])
                    parsed_localidad = parsed_result['Localidad']
                    if parsed_localidad == 'ANTONIO NARIÑO':
                        original_df.at[index,'LOCALIDAD'] = parsed_localidad
                    else:
                        original_df.at[index,'LOCALIDAD'] = unidecode.unidecode(parsed_localidad)
                    original_df.at[index,'COD_LOCALIDAD'] = int(get_cod_localidad(original_df.at[index,'LOCALIDAD']))
                    original_df.at[index,'SEC_CATASTRAL'] = parsed_result['Sector catastral']
                    assign_upz(original_df,index,parsed_result['UPZ'])
                    return "Rebuilt"
            else:
                return "Not processed"
        except Exception as e:
            logging.error(e)
            pass
        logging.debug('Termina rebuild_location_in_nuse')


    def rebuild_address_in_nuse(original_df, index):
        logging.debug('Inicia rebuild_address_in_nuse. Parametros: index={}, driver={} '.format(index, driver))
        try:
            log_text = original_df.at[index,'LOG_TEXT']
            address_found = re.search(address_regex,log_text)

            if address_found != None:
                parsed_address = clean_address(address_found)
                original_df.at[index,'STR_DIRECCION_INCIDENTE'] = parsed_address.strip()
                return "Rebuilt"
            else:
                original_df.at[index,'STR_DIRECCION_INCIDENTE'] = 'ND'
                return "Not processed"
        except Exception as e:
            logging.error(e)
            pass
        logging.debug('Termina rebuild_address_in_nuse')

    def clean_address(address_found):
        logging.debug('Inicia clean_address. Parametros: address_found={}'.format(address_found))
        try:
            exclude_char_list = ['~','/','*','(',')']
            one_occurrence = address_found.group().split(',,,')[0].replace(',',' ')
            final_address = one_occurrence
        
            for char in exclude_char_list:
                if char in one_occurrence:
                    final_address = final_address.split(char)[0]
                
            numbers_in_substring = re.sub('[^0-9]','', final_address)
            numbers_proportion = len(numbers_in_substring)/len(final_address)
        
            if numbers_proportion < 0.2:
                final_address = 'ND'
        
            return final_address
        except Exception as e:
            logging.error(e)
            pass
        logging.debug('Termina clean_address')

    ##print('----- 3 -----')

    # Implement rebuild methods

    logging.debug('Inicia rebuild methods')
    try:

        data_location = '/home/unal/modelo_rinas/UNAL/merged_nuse.csv'
        merged_nuse=pd.read_csv(data_location,delimiter=",")

        pd.DataFrame({"Tipo de dato":merged_nuse.dtypes.values,
                    "Celdas con valor '-'":(merged_nuse == '-').sum().values,
                    "Celdas con valor ''":(merged_nuse == '').sum().values,
                    "Celdas con valor ' '":(merged_nuse == ' ').sum().values,
                    "Celdas vacías": merged_nuse.isna().sum().values},
                    index=merged_nuse.columns)
    except Exception as e:
        logging.error(e)
        pass
    logging.debug('Termina rebuild methods')


    ##print('----- 4 -----')

    # Rebuild address through log_text

    logging.debug('Inica Rebuild address through log_text')
    try:
        #Try to rebuild missing address through log_text field
        df_empty_locations_without_address = merged_nuse.loc[merged_nuse['STR_DIRECCION_INCIDENTE'] == '-']
        list_idx_rebuild_address = list(df_empty_locations_without_address.index.values)

        address_regex= '(CL|DG|KR|TV)+\s\d+.*(,,)'
        registers_to_process = len(list_idx_rebuild_address)
        rebuilt_registers = 0
        registers_not_processed = 0
        other_condition_counter = 0
    except Exception as e:
        logging.error(e)
        pass
    logging.debug('Termina Rebuild address through log_text')

    ##print('----- 5 -----')

    for index in list_idx_rebuild_address:
        logging.debug('Comienza Ciclo de list_idx_rebuild_address')
        try:
            state = rebuild_address_in_nuse(merged_nuse, index)
    
            if state == "Rebuilt":
                rebuilt_registers += 1
            elif state == "Not processed":
                registers_not_processed += 1
            else:
                other_condition_counter += 1
        except Exception as e:
            logging.error(e)
            pass
        logging.debug('Ciclo de list_idx_rebuild_address')  

    ##print('----- 6 -----')


    logging.debug('Inicia carga dataframe')
    try:
        merged_nuse.to_csv(r'/home/unal/modelo_rinas/UNAL/merged_nuse.csv',index=None)

        pd.DataFrame({"Tipo de dato":merged_nuse.dtypes.values,
                    "Celdas con valor '-'":(merged_nuse == '-').sum().values,
                    "Celdas con valor 'ND'":(merged_nuse == 'ND').sum().values,
                    "Celdas vacías": merged_nuse.isna().sum().values},
                    index=merged_nuse.columns)

    except Exception as e:
        logging.error(e)
        pass
    logging.debug('Termina carga dataframe')  
    
    ##print('----- 7 -----')

    logging.debug('Inicia Rebuild location through address')
    try:   
        # Rebuild location through address
        data_location = '/home/unal/modelo_rinas/UNAL/merged_nuse.csv'
        df_input = pd.read_csv(data_location,delimiter=",")

        df1 = df_input.loc[df_input['COD_UPZ'] == '-']
        df2 = df_input.loc[df_input['UPZ'] == '-']
        df3 = df_input.loc[df_input['COD_SEC_CATAST'] == '-']
        df4 = df_input.loc[df_input['SEC_CATASTRAL'] == '-']
        df5 = df_input.loc[df_input['COD_BARRIO'] == '-']
        df6 = df_input.loc[df_input['BARRIO'] == '-']
    except Exception as e:
            logging.error(e)
            pass
    logging.debug('Termina Rebuild location through address') 

    ##print('----- 8 -----')

    df1.equals(df2) and df1.equals(df3) and df1.equals(df4) and df1.equals(df5) and df1.equals(df6)

    ##print('----- 9 -----')

    #Try to rebuild 'sector catastral', 'UPZ', 'localidad', 'latitud', 'longitud' through address

    df_empty_locations_with_address = df1.loc[df1['STR_DIRECCION_INCIDENTE'] != 'ND']
    list_idx_rebuild_location = list(df_empty_locations_with_address.index.values)

    ##print('----- 10 -----')

    #Rebuild 'sector catastral', 'UPZ', 'localidad', 'latitud', 'longitud' using web scraping
    df_output = df_input
   
    ##print('----- 11 -----')

    logging.debug('Inicia el almacenamiento de DataFrame rebuild_locations_nuse_29112019')
    try:
        df_output.to_csv(r'/home/unal/modelo_rinas/UNAL/merged_nuse.csv',index=None)

        pd.DataFrame({"Tipo de dato":df_output.dtypes.values,
                    "Celdas con valor '-'":(df_output == '-').sum().values,
                    "Celdas con valor 'ND'":(df_output == 'ND').sum().values,
                    "Celdas vacías": df_output.isna().sum().values},
                    index=df_output.columns)
    except Exception as e:
            logging.error(e)
            pass
    logging.debug('Termina el almacenamiento de DataFrame rebuild_locations_nuse_29112019')

    #assign ND to df_empty_locations_without_address on location fields
    #'SEC_CATASTRAL', 'UPZ', 'COD_UPZ', 'LATITUD'', 'LONGITUD', 'LOCALIDAD', 'COD_LOCALIDAD'

    ##print('----- 12 -----')

    logging.debug('Inicia el almacenamiento de DataFrame rebuild_locations_nuse_29112019')
    try:
        data_location = '/home/unal/modelo_rinas/UNAL/merged_nuse.csv'
        df_input = pd.read_csv(data_location,delimiter=",")

        #Registers without address or coordinates can not be rebuilt
        df_empty_locations_without_address = df_input.loc[(df_input['STR_DIRECCION_INCIDENTE'] == 'ND') & (df_input['LATITUD']==-1) & (df_input['LONGITUD']==-1)]
        list_idx_not_rebuild = list(df_empty_locations_without_address.index.values)
    except Exception as e:
            logging.error(e)
            pass
    logging.debug('Termina el almacenamiento de DataFrame rebuild_locations_nuse_29112019')

    df_output = df_input

    ##print('----- 13 -----')

    for index in list_idx_not_rebuild:
        logging.debug('--')
        try:
            df_output.at[index,'SEC_CATASTRAL'] = 'ND'
            df_output.at[index,'UPZ'] = 'ND'
            df_output.at[index,'COD_UPZ'] = 'ND'
            df_output.at[index,'LOCALIDAD'] = 'SIN LOCALIZACION'
            df_output.at[index,'LATITUD'] = 99
        except Exception as e:  
            logging.error(e)
            pass
        logging.debug('--')

    df_output.to_csv(r'/home/unal/modelo_rinas/UNAL/merged_nuse.csv',index=None)

    # 3. Standardise

    data_location = '/home/unal/modelo_rinas/UNAL/merged_nuse.csv'
    df_input = pd.read_csv(data_location,delimiter=",")

    ##print('----- 14 -----')

    #create timpestamp col to handle time ranges on unique event process
    df_input['time_stamp']=pd.to_datetime(df_input['FECHA']) 

    pd.DataFrame({"Tipo de dato":df_input.dtypes.values,
                  "Celdas con valor '-'":(df_input == '-').sum().values,
                  "Celdas con valor 'ND'":(df_input == 'ND').sum().values,
                  "Celdas vacías": df_input.isna().sum().values},
                 index=df_input.columns)

    # 3.1 One register per event: event that occurs within 400 mts radius and 20 minutes time interval
    # Find duplicated events

    ##print('----- 15 -----')

    import time, datetime
    time_offset = 20
    coor_offset = 0.001

    def find_duplicated_events(df, row):
        logging.debug('Inicia find_duplicated_events. Parametros: df={}, row={}'.format(df, row))
        try:
            current_time = row['time_stamp']
            current_lat = row['LATITUD']
            current_lon = row['LONGITUD']
            current_point=Point(current_lon,current_lat)

            duplicated_event_idx = {}
            limit_time_interval = current_time + datetime.timedelta(minutes = time_offset)
            df_event_time = df.loc[(df['time_stamp'] >= current_time) & (df['time_stamp'] < limit_time_interval)]
        
            lat_point_list = [current_lat-coor_offset, current_lat-coor_offset, current_lat+coor_offset, current_lat+coor_offset]
            lon_point_list = [current_lon+coor_offset, current_lon-coor_offset, current_lon-coor_offset, current_lon+coor_offset]
            polygon_event = Polygon(zip(lon_point_list, lat_point_list))
        
            for index, row in df_event_time.iterrows():
                point=Point(row['LONGITUD'],row['LATITUD'])
                if point.within(polygon_event):
                    
                   duplicated_event_idx[index] = row['STR_NUMERO_INTERNO']
            return duplicated_event_idx
        except Exception as e:
            logging.error(e)
            pass
        logging.debug('Termina find_duplicated_events')

    df_output = df_input.copy()

    ##print('----- 16 -----')

    df_output['dup_event'] = df_output.apply (lambda row: find_duplicated_events(df_output, row), axis=1)

    df_output.to_csv(r'/home/unal/modelo_rinas/UNAL/merged_nuse.csv',index=None)

    # Delete duplicated events: preserve the first event on dup_event column

    data_location = '/home/unal/modelo_rinas/UNAL/merged_nuse.csv'
    df_input = pd.read_csv(data_location,delimiter=",")

    pd.DataFrame({"Tipo de dato":df_input.dtypes.values,
                  "Celdas con valor '-'":(df_input == '-').sum().values,
                  "Celdas con valor 'ND'":(df_input == 'ND').sum().values,
                  "Celdas vacías": df_input.isna().sum().values},
                 index=df_input.columns)

    ##print('----- 17 -----')

    #Get index of registers that should be deleted
    import ast
    df = df_input
    list_idx_repeated = []
    list_idx_preserved = []
    registers_to_process = len(df)
    list_idx_processed =[]
    counter_processed = 0

    for index, row in df.iterrows():
        logging.debug('--')
        try:
            dup_event_x = ast.literal_eval(df.at[index,'dup_event'])
            current_dup_events = list(dup_event_x.keys())

            if (current_dup_events[0] not in list_idx_processed) & (current_dup_events[0] not in list_idx_preserved):
                list_idx_preserved.append(current_dup_events[0])
                list_idx_processed.append(current_dup_events[0])
                current_dup_events.pop(0)

            for idx_event in current_dup_events:
                if idx_event not in list_idx_processed:
                    list_idx_repeated.append(idx_event)
                    list_idx_processed.append(idx_event)
                    
            counter_processed += 1
        except Exception as e:
            logging.error(e)
            pass
        logging.debug('--')
    
    

    #check (quantitatively) ID of preserved and repeated events index was succesful
    #print(len(list_idx_repeated)+len(list_idx_preserved))
    #print(len(list_idx_processed))
    join_list = list_idx_preserved + list_idx_repeated

    ##print('----- 18 -----')

    import collections
    seen = set()
    uniq = []
    for x in join_list:
        logging.debug('--')
        try:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        except Exception as e:
            logging.error(e)
            pass
        logging.debug('--')

    ##print('----- 19 -----')

    lst = join_list
    dupItems = []
    uniqItems = {}
    for x in lst:
        logging.debug('--')
        try:
            if x not in uniqItems:
                uniqItems[x] = 1
            else:
                if uniqItems[x] == 1:
                    dupItems.append(x)
                uniqItems[x] += 1
        except Exception as e:
            logging.error(e)
            pass
        logging.debug('--')
    
    ##print('----- 20 -----')

    df_output = df_input.copy()

    df_output=df_output.drop(list_idx_repeated)
    df_output.drop(columns=['dup_event','time_stamp'],inplace=True)
    df_output.reset_index(inplace=True)

    df_output.to_csv(r'/home/unal/modelo_rinas/UNAL/merged_nuse.csv',index=None)


    ##print('----- 21 -----')

    # 4. Normalise
    logging.debug('--')
    try:
        data_location = '/home/unal/modelo_rinas/UNAL/merged_nuse.csv'
        df_input = pd.read_csv(data_location,delimiter=",")

        pd.DataFrame({"Tipo de dato":df_input.dtypes.values,
                    "Celdas con valor '-'":(df_input == '-').sum().values,
                    "Celdas con valor 'ND'":(df_input == 'ND').sum().values,
                    "Celdas vacías": df_input.isna().sum().values},
                    index=df_input.columns)
    except Exception as e:
            logging.error(e)
            pass
    logging.debug('--')
    # Verify FECHA             

    # It´s a REGEX with the form: nn-www-nnnn
    regex_fecha = '^\d{1,2}-\w{3}-\d{1,2}$'
    df_input['FECHA'].str.contains(regex_fecha, regex=True).all()

    # It´s a regex:
    regex_hora = '^[0-2][0-9][0-5]|[0-9]$'
    df_input['HORA'].apply(str).str.contains(regex_hora, regex=True).all()

    #Verify ANIO

    # It´s a number between 2017 and 2019

    df_input['ANIO'].between(2017,2030).all()

    # Verify MES

    # It´s a number between 1 and 12

    df_input['MES'].between(1,12).all()

    # Verify COD_LOCALIDAD - LOCALIDAD

    var_aux = 'STR_NUMERO_INTERNO'
    df_input.groupby(['COD_LOCALIDAD','LOCALIDAD']).agg({var_aux:'count'}).reset_index().rename(columns={var_aux:'Frecuencia'})

    ##print('----- 22 -----')

    # Verify LATITUD, LONGITUD

    # Should be in Bogotá
    bog_loc=gpd.read_file('bogota_polygon.geojson')

    df_output=df_input.copy()

    def check_bog_location(df, row):
        logging.debug('Inicia check_bog_location')
        try:
            lat = row['LATITUD']
            lon = row['LONGITUD']
            current_point = Point(lon,lat)
            if bog_loc.geometry.contains(current_point)[0]:
                return True
            else:
                return False
        except Exception as e:
            logging.error(e)
            pass
        logging.debug('--')

    df_output['in_bogota?'] = df_output.apply (lambda row: check_bog_location(df_output, row), axis=1)

    df_output.loc[(df_output['in_bogota?'] == False)]

    df_output.to_csv(r'/home/unal/modelo_rinas/UNAL/merged_nuse.csv',index=None)

    #Get index of registers out of Bogota and drop it
    list_index_out_bogota=df_output[(df_output['in_bogota?'] == False)].index
    df_output=df_output.drop(list_index_out_bogota)
    df_output['in_bogota?'].all()

    df_output.to_csv(r'/home/unal/modelo_rinas/UNAL/merged_nuse.csv',index=None)

    ##print('----- 23 -----')

    # 5. De-duplicate

    data_location = '/home/unal/modelo_rinas/UNAL/merged_nuse.csv'
    df_input = pd.read_csv(data_location,delimiter=",")


    pd.DataFrame({"Tipo de dato":df_input.dtypes.values,
                  "Celdas con valor '-'":(df_input == '-').sum().values,
                  "Celdas con valor 'ND'":(df_input == 'ND').sum().values,
                  "Celdas vacías": df_input.isna().sum().values},
                 index=df_input.columns)

    # Verify there are not identycal rows

    ##print("Filas duplicadas",df_input.duplicated().sum())

    # Verify unique STR_NUMERO_INTERNO

    len(df_input) == len(df_input['STR_NUMERO_INTERNO'].unique())

    df_input.to_csv(r'/home/unal/modelo_rinas/UNAL/merged_nuse.csv',index=None)

    ##print('----- 24 -----')

    # 6. Verify and enrich

    data_location = '/home/unal/modelo_rinas/UNAL/merged_nuse.csv'
    df_input = pd.read_csv(data_location,delimiter=",")

    pd.DataFrame({"Tipo de dato":df_input.dtypes.values,
                  "Celdas con valor '-'":(df_input == '-').sum().values,
                  "Celdas con valor 'ND'":(df_input == 'ND').sum().values,
                  "Celdas vacías": df_input.isna().sum().values},
                 index=df_input.columns)

    df_output=df_input.copy()

    # Verify columns with empty or anomalous values

    # Check COD_UPZ, UPZ, SEC_CATASTRAL with '-' values
    df1 = df_output.loc[df_output['COD_UPZ']=='-']
    df2 = df_output.loc[df_output['UPZ']=='-']
    df3 = df_output.loc[df_output['SEC_CATASTRAL']=='-']
    df1.equals(df2) and df1.equals(df3)

    # Check COD_UPZ, UPZ, SEC_CATASTRAL with empty values
    df_output.loc[df_output['COD_UPZ'].isna(),'COD_UPZ'] = '-'
    df_output.loc[df_output['UPZ'].isna(),'UPZ'] = '-'
    df_output.loc[df_output['SEC_CATASTRAL'].isna(),'SEC_CATASTRAL'] = '-'

    # Check ESTADO_INCIDENTE with empty values
    df_output['ESTADO_INCIDENTE'].value_counts()
    #rebuild empty values with 'CERRADO'
    df_output.loc[df_output['ESTADO_INCIDENTE'].isna(),'ESTADO_INCIDENTE'] = 'CERRADO'

    # Check BARRIO and COD_BARRIO with '-' values
    df1 = df_output.loc[df_output['BARRIO']=='-']
    df2 = df_output.loc[df_output['COD_BARRIO']=='-']
    df3 = df_output.loc[df_output['COD_SEC_CATAST']=='-']
    df1.equals(df2) and df1.equals(df3)

    # Check STR_DIRECCION_INCIDENTE with 'ND' values
    df_output.loc[(df_output['STR_DIRECCION_INCIDENTE'] == 'ND')]
    df_output.loc[(df_output['STR_DIRECCION_INCIDENTE'] == 'ND') & (df_output['COD_LOCALIDAD'] == 99)]


    # Delete aditional columns created on cleaning process

    #df_output.drop(columns=['index','in_bogota?'],inplace=True)
    df_output.reset_index(inplace=True)

    #df_output.drop(columns=['index'],inplace=True)

    pd.DataFrame({"Tipo de dato":df_output.dtypes.values,
                  "Celdas con valor '-'":(df_output == '-').sum().values,
                  "Celdas con valor 'ND'":(df_output == 'ND').sum().values,
                  "Celdas vacías": df_output.isna().sum().values},
                 index=df_output.columns)

    df_output.to_csv(r'/home/unal/modelo_rinas/UNAL/merged_nuse.csv',index=None) 
    logging.debug('Termina la ejecucion de codigo') 


    #print("--- %s seconds ---" % (time.time() - start_time))
    
    return(df_output)

    def FECHA_mod(txt):
        return txt.replace("T"," ")
