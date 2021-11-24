__name__ = "Class for the SEPP Model"
#%matplotlib inline
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
import scipy.stats
import random 
from scipy.optimize import fsolve
from scipy.optimize import minimize
import timeit
import datetime
from shapely import wkt
from pyproj import Proj
from shapely.geometry import Point, LineString, Polygon
from shapely.geometry import MultiPoint, MultiLineString, MultiPolygon, box
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
from pyproj import Proj, transform
from pyproj import Transformer
from collections import namedtuple
from sepp_auxiliar_functions import *
import fiona
import logging
import requests
import traceback
import json
import constants_manager as c
from datetime import datetime
from pyspark.sql import SparkSession, functions
import logging
import pandas as pd
from datetime import timedelta, datetime
from utilis import get_estados_ejecucion, get_tipos_proceso, get_token_acces, update_process_state
from constants_manager import ESTADO_EXITO, ESTADO_ERROR, ESTADO_PROCESO, ESTADO_CANCELADO, NAME_PREDICCION, NAME_ENTRENAMIENTO, NAME_PREPROCESAMIENTO, NAME_VALIDACION
import shutup 
shutup.please()

class ModeloBase:

    def preprocdatos_model(self):
        pass
    
    def train_model(self):
        pass
       
    def predict_model(self):
        pass
    
    def validation_model(self):
        pass


class ModeloRinhas(ModeloBase):
    """
    Use the Self Exciting Point Process to Train the model (training_model). It makes a simulation of events for some 
    number of hours. It is the prediction_model. After, for the validation_model, it is used the hit rate metric.
    
    :param setting: 0 if the data are real data, 1 if the data are simulated data
    :param events_gdf: dataframe with the events data for the training model
    :param cov_norm_cell_m: array of covariates of each cell of the grid 
    :param cov_norm_eventos_m: array of covariates of each cell where the events ocurred
    :param temp_factor: temporal scale for the time data of the dataframe
    :param sp_factor: spatial scale for the spatial time as polygon area and distance between events
    :param number_hours: number of hours to simulate data or make the prediction 
    :param poligonos_df: dataframe with the polygons of each cell of grid ant its respective covariate values
    :return beta: optimization of the parameter related with the constant background rate
    :return omega: optimization of the parameter related with ocurrence of triggering events
    :return sigma2: optimization of the parameter related with the spread of triggering events
    :return puntos_gdf: dataframe with the simulated events
    :return array_cells_events_sim: two dimensional array with the number of cell and the number of events ocurred on it
    :return h_r: the validation process is make with the hit rate metric for the simulated events
    """
    def __init__(self):
        self.tipos_proceso = get_tipos_proceso(get_token_acces())
        self.estados_ejecucion = get_estados_ejecucion(get_token_acces())

    #########################################
    # Training Process (Parameter fitting) #
    ########################################

    def preprocdatos_model(self, fecha_inicial, fecha_final):
    #    
    #    Realiza el preprocesamiento de los datos que se extraen de la base de datos del servidor
    #    
    #    :return df: dataframe con los datos preprocesados (limpios)
        root_logger= logging.getLogger()
        root_logger.setLevel(logging.DEBUG) # or whatever
        handler = logging.FileHandler('text.log', 'w', 'utf-8') # or whatever
        formatter = logging.Formatter('%(name)s %(message)s') # or whatever
        handler.setFormatter(formatter) # Pass handler as a parameter, not assign
        root_logger.addHandler(handler)
        
        logging.debug("Comienza el preprocesamiento de datos para el modelo de rinas de seguridad.") 
  
        update_process_state(self.tipos_proceso[NAME_PREPROCESAMIENTO], self.estados_ejecucion[ESTADO_PROCESO], get_token_acces())
        
        try:
            poly = gpd.read_file('poligonos_covariados.geojson') 
            date_format_str = '%Y-%m-%d %H:%M:%S'
            fecha_inicial = datetime.strptime(fecha_inicial, date_format_str)
            fecha_final = datetime.strptime(fecha_final, date_format_str)
            datos_eventos_servidor = cleaning(fecha_inicial, fecha_final)
            long_datos_eventos_servidor = len(datos_eventos_servidor)
            datos_eventos_servidor_limpios = limpieza_datos(datos_eventos_servidor)
            long_datos_eventos_servidor_limpios = len(datos_eventos_servidor_limpios)
            def FECHA_mod(txt):
                return txt.replace("T"," ")
            datos_eventos_servidor_limpios['FECHA'] = datos_eventos_servidor_limpios['FECHA'].map(FECHA_mod)
            eventos_cov = eventos_covariados(datos_eventos_servidor_limpios, poly, fecha_inicial, fecha_final)
            eventos_cov.to_file("eventos_covariados.geojson", driver='GeoJSON')
            logging.debug("Termina el preprocesamiento de los datos para el modelo de rinas de seguridad.")
            update_process_state(self.tipos_proceso[NAME_PREPROCESAMIENTO], self.estados_ejecucion[ESTADO_EXITO], get_token_acces())
            return eventos_cov, long_datos_eventos_servidor, long_datos_eventos_servidor_limpios
        
        except Exception as e:
            update_process_state(self.tipos_proceso[NAME_PREPROCESAMIENTO], self.estados_ejecucion[ESTADO_ERROR], get_token_acces())
            msg_error = "No se completo el preprocesamiento de los datos del modelo de rinas de seguridad."   
            logging.error(msg_error)
            raise Exception(msg_error + " / " + str(e))

    def train_model(self, data_eventos):
        """
        Calculate the optimized values for the conditional intensity for the SEPP model

        :param events_gdf: dataframe with the events data for the training model
        :param cov_norm_cell_m: array of covariates of each cell of the grid 
        :param cov_norm_eventos_m: array of covariates of each cell where the events ocurred
        :param temp_factor: temporal scale for the time data of the dataframe
        :param sp_factor: spatial scale for the spatial time as polygon area and distance between events
        :return beta: optimization of the parameter related with the constant background rate
        :return omega: optimization of the parameter related with ocurrence of triggering events
        :return sigma2: optimization of the parameter related with the spread of triggering events
        """
        root_logger= logging.getLogger()
        root_logger.setLevel(logging.DEBUG) # or whatever
        handler = logging.FileHandler('text.log', 'w', 'utf-8') # or whatever
        formatter = logging.Formatter('%(name)s %(message)s') # or whatever
        handler.setFormatter(formatter) # Pass handler as a parameter, not assign
        root_logger.addHandler(handler)

        logging.debug("Comienza el entrenamiento para el modelo de rinas de seguridad.") 

        update_process_state(self.tipos_proceso[NAME_ENTRENAMIENTO], self.estados_ejecucion[ESTADO_PROCESO], get_token_acces())
        
        try:
            if len(data_eventos) > 1000 and len(data_eventos) <= 150000:
                poligonos_df = gpd.read_file('poligonos_covariados.geojson') 
                def FECHA_mod(txt):
                    return txt.replace("T"," ")
                data_eventos['FECHA'] = data_eventos['FECHA'].map(FECHA_mod)
                cov_norm_eventos_m = np.array(data_eventos[['Promedio Estrato 2019','Area de Cuadrantes de Policia 2020','Comando de Atencion Inmediata','Estaciones Policia','Int']])
                data_eventos = data_eventos.rename(columns={'X':'Lon','Y':'Lat','cells':'Cell','TimeStamp':'Time'})
                data_eventos['Time'] = pd.to_numeric(data_eventos['Time'], downcast="float").to_numpy()
                data_eventos = data_eventos[['Lon','Lat','Time','Cell']]
                posiciones = np.array(data_eventos[["Lon", "Lat"]])
                cov_norm_cell_m = np.array(poligonos_df[['Promedio Estrato 2019','Area de Cuadrantes de Policia 2020','Comando de Atencion Inmediata','Estaciones Policia','Int']])
                sp_factor = 10
                sq_dist = np.tril(np.sum((posiciones[:, np.newaxis, :] - posiciones[np.newaxis, :, :]) ** 2, axis = -1)*10e-7, -1)
                tiempo_eventos = np.array(data_eventos.Time)
                dif_tiempo = np.tril(tiempo_eventos[..., np.newaxis] - tiempo_eventos[np.newaxis, ...], -1)
                beta = np.ones(cov_norm_eventos_m.shape[1])
                omega = 1
                sigma2 =  1
                T = pd.to_numeric(data_eventos.Time, downcast="float").to_numpy()[-1]
                sc = 0.03*sp_factor

                #============================
                #== Parameter Optimization ==
                #============================
                while True:
                    cte1 = 1/(2 * sigma2)
                    cte2 = omega /(2 * np.pi *  sigma2)
                # Matriz gij de los eventos tal que los elementos de tiempo j>=i son iguales a 0
                    gij = np.tril(cte2*np.exp(-sq_dist*cte1)*np.exp(-dif_tiempo*omega), -1)
                # Suma de las filas de gij
                    kernel = np.sum(gij, axis=1)
                # Media de los eventos del background
                    mu = np.exp((beta * cov_norm_eventos_m.astype(float)).sum(axis=1))
                # Probabilidad condicional
                    Lambda = mu + kernel
                # Probabilidad de que el evento j sea generador por el evento i
                    pij = gij/Lambda
                # prob for event i was triggering for some event ocurred before i
                    pte = kernel/Lambda
                # prob of background event
                    pbe = mu/Lambda
                # Argumentos para la solucion de la ecuacion de beta
                    data = (T, sc, cov_norm_cell_m, cov_norm_eventos_m, pbe)
                    betac = root_beta(beta, data)
                    omegac = new_omega1(omega, tiempo_eventos, dif_tiempo, T, pij, pte)
                    sigma2c = update_sigma21(sigma2, pij, pte, sq_dist)
                    dif_beta = abs(beta - betac)
                    dif_omega = abs(omega - omegac)
                    dif_sigma2 = abs(sigma2 - sigma2c)
                    if (dif_beta.all() > 1e-5) and (dif_omega > 1e-5) and (dif_sigma2 > 1e-5):
                        beta = betac
                        omega = omegac
                        sigma2 = sigma2c
                    else:
                        break
                file = open("parametros_optimizados.txt", "w")
                file.write(str(beta[0]) + '\n')
                file.write(str(beta[1]) + '\n')
                file.write(str(beta[2]) + '\n')
                file.write(str(beta[3]) + '\n')
                file.write(str(beta[4]) + '\n')
                file.write(str(omega) + '\n')
                file.write(str(sigma2) + '\n')
                file.close()
                logging.debug("Termina el entrenamiento para el modelo de rinas de seguridad.")
                update_process_state(self.tipos_proceso[NAME_ENTRENAMIENTO], self.estados_ejecucion[ESTADO_EXITO], get_token_acces())
                return beta, omega, sigma2

        except Exception as e:
            update_process_state(self.tipos_proceso[NAME_ENTRENAMIENTO], self.estados_ejecucion[ESTADO_ERROR], get_token_acces())
            msg_error = "No se completo el proceso de entrenamiento del modelo de rinas de seguridad."
            logging.error(msg_error)
            raise Exception(msg_error + " / " + str(e))

    ##############
    # Simulation #
    ##############
    def predict_model(self, fecha_inicial, fecha_final):
        """
        Simulate events during a time
        :param beta: weight for the covariates
        :param omega: how much decay temporally the probability of triggering event
        :param sigma2: how much spread the triggering effect from background event
        :param number_hours = time for simulating events
        :param poligonos_df = poligonos_df
        :return puntos_gdf: points of the simulate events 
        :return array_cells_events_sim: array of the simulated events and its cell number 
        :return array_cells_events_data: array of the siedco data events and its cell number
        """
        root_logger= logging.getLogger()
        root_logger.setLevel(logging.DEBUG) # or whatever
        handler = logging.FileHandler('text.log', 'w', 'utf-8') # or whatever
        formatter = logging.Formatter('%(name)s %(message)s') # or whatever
        handler.setFormatter(formatter) # Pass handler as a parameter, not assign
        root_logger.addHandler(handler)
        
        logging.debug("Comienza la prediccion para el modelo de rinas de seguridad.") 
        update_process_state(self.tipos_proceso[NAME_PREDICCION], self.estados_ejecucion[ESTADO_PROCESO], get_token_acces())

        try:
            date_format_str = '%Y-%m-%d %H:%M:%S'
            fecha_inicial = datetime.strptime(fecha_inicial, date_format_str)
            fecha_final = datetime.strptime(fecha_final, date_format_str)
            diff = (fecha_final - fecha_inicial).total_seconds()/3600
            if diff > 0 and diff <= 168: 
                poligonos_df = gpd.read_file('poligonos_covariados.geojson') 
                cov_norm_cell_m = np.array(poligonos_df[['Promedio Estrato 2019','Area de Cuadrantes de Policia 2020','Comando de Atencion Inmediata','Estaciones Policia','Int']])
                parameters = np.array([])
                filename = "parametros_optimizados.txt"
                with open(filename) as f_obj:
                    for line in f_obj:
                        parameters = np.append(parameters,float(line.rstrip()))
                beta = np.array([parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]])
                omega = parameters[5]
                sigma2 = parameters[6] 
                window_size = diff 
                Event = namedtuple("Event", ["t", "loc_x", "loc_y"])
                alpha = 1
                sigma = np.sqrt(sigma2)
                def get_random_point_in_polygon_back(poly):
                    """
                    Generates random background events inside a polygon
                    """
                
                    minx, miny, maxx, maxy = poly.bounds
                    while True:
                        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                        if poly.contains(p):
                            return p

                def get_random_point_in_polygon_trig(poly):
                    """
                    Generates random triggering events inside a polygon
                    :param poly: geometry of the polygon
                    :return p: point inside of polygon
                    """
                
                    while True:
                        loc_x = np.random.normal(loc=parent.loc_x, scale=sigma)
                        loc_y = np.random.normal(loc=parent.loc_y, scale=sigma)
                        p = Point(loc_x, loc_y)
                        if poly.contains(p):
                            return p

                def sort_with_causes(points, caused_by):
                    """
                    Sorts events in time, and maintains caused by information
                    :param points: array of events
                    :parm caused_by: array with the number of the event that caused it
                    :return events: sorted point array
                    :return new_caused_by: sorted caused_by array
                    """
                    # caused_by[i] = j<i if i caused by j, or i if i background
                    tagged = list(enumerate(points))
                    tagged.sort(key = lambda pair : pair[1].t)
                    new_caused_by = []
                    for old_index, event in tagged:
                        old_cause_index = caused_by[old_index]
                        new_cause_index,_ = next(x for x in enumerate(tagged) if x[1][0] == old_cause_index)
                        new_caused_by.append(new_cause_index)
                    events = [x[1] for x in tagged]
                    return events, new_caused_by

                def simulate_sub_process(parent : Event, k):
                    """
                    Generates the triggered events
                    :param parent : Event: parent event
                    :param k: number of cell where ocurred the event
                    """
                    points = []
                    t = 0
                    while True:                
                        t += np.random.exponential(1/omega)
                        if t >= alpha:
                            return points
                        def get_random_point_in_polygon_trig(poly):
                            while True:
                                loc_x = np.random.normal(loc=parent.loc_x, scale=sigma)
                                loc_y = np.random.normal(loc=parent.loc_y, scale=sigma)
                                p = Point(loc_x, loc_y)
                                if poly.contains(p):
                                    return p
                        p = get_random_point_in_polygon_trig(poligonos_df.geometry[k])
                        loc_x = p.x
                        loc_y = p.y
                        points.append(Event(parent.t + np.log(alpha/abs(alpha-t*omega))/omega , loc_x, loc_y))

                def _add_point(points, omega):
                    """
                    Generates interarrival times between consecutive events and adds it to a list
                    :param points: list of points
                    :param mu: events rate for the cell
                    """
            
                    wait_time = np.random.exponential(1/omega)
                    if len(points) == 0:
                        last = 0
                    else:
                        last = points[-1]
                    points.append(last + wait_time)
            
                def sample_poisson_process_hom(window_size, omega):
                    """
                    Generates the background events according to the background events rate in some window temporal size 
                    :param window_size: window temporal size
                    :param omega: background events rate
                    :return points: simulated background points
                    """
                    points = []
                    _add_point(points, omega)
                    while points[-1] < window_size:
                        _add_point(points, omega)
                    return points

                def simulate(window_size, k):
                    """
                    Generates the background events with their descendant events in some window temporal size in one cell
                    :param window_size: window temporal size
                    :param k: number of cell
                    :return puntos_gdf: dataframe with the simulated events
                    :return array_cells_events_sim: two dimensional array with cell number and its corresponding number of events
                    """
                    backgrounds = sample_poisson_process_hom(window_size, mu[k])
                    backgrounds = backgrounds[:-1]
                    points = []
                    for i in range(0, len(backgrounds)):
                        t = backgrounds[i]
                        p = get_random_point_in_polygon_back(poligonos_df.geometry[k])
                        px = p.x
                        py = p.y
                        st_point = Event(t, px, py)
                        points.append(st_point)
                    backgrounds = points    
                    caused_by = [ i for i in range(len(points))]
                    to_process = [(i,p) for i, p in enumerate(points)]
                    while len(to_process) > 0:
                        (index, next_point) = to_process.pop()
                        for event in simulate_sub_process(next_point,k):
                            if event.t < window_size:
                                points.append(event)
                                caused_by.append(index)
                                to_process.append((len(points) - 1,event))
                    points, caused_by = sort_with_causes(points, caused_by)
                    return points, backgrounds, caused_by

                events_sim_on_cells = np.array([])
                all_events_sim = np.array([])

                mu = np.exp((beta*cov_norm_cell_m).sum(axis=1).astype(float))
        
                for k in range(0, len(cov_norm_cell_m)):
                    random.seed(7)
                    ev = simulate(window_size, k)[0]
                    # events of all cells
                    all_events_sim = np.append(all_events_sim, ev)
                    # number of event per cell
                    events_sim_on_cells = np.array(np.append(events_sim_on_cells, len(ev)),int)
            
                # simulated events on cells (number of events on each cell) (t,x,y) each event   
                all_events_sim = all_events_sim.reshape(int(len(all_events_sim)/3), 3)
            
                puntos_gdf = gpd.GeoDataFrame(all_events_sim, columns=["TimeStamp", "X", "Y"])
                geometry = [Point(xy) for xy in zip(puntos_gdf['X'], puntos_gdf['Y'])]
                crs = {'init': 'epsg:3857'}
                puntos_gdf = gpd.GeoDataFrame(puntos_gdf, crs=crs, geometry=geometry)
                puntos_gdf = puntos_gdf.sort_values('TimeStamp').reset_index().drop(columns = 'index')
            
                array_cells_events_sim = np.arange(0, len(events_sim_on_cells), 1).tolist()
                array_cells_events_sim = list(map(list,zip(array_cells_events_sim,events_sim_on_cells)))

                logging.debug("Termina la prediccion para el modelo de rinas de seguridad.")
                update_process_state(self.tipos_proceso[NAME_PREDICCION], self.estados_ejecucion[ESTADO_EXITO], get_token_acces())
                return puntos_gdf, array_cells_events_sim

        except Exception as e:
            update_process_state(self.tipos_proceso[NAME_PREDICCION], self.estados_ejecucion[ESTADO_ERROR], get_token_acces())
            msg_error = "No se completo el proceso de prediccion del modelo de rinas de seguridad."   
            logging.error(msg_error)
            raise Exception(msg_error + " / " + str(e))

    ##############
    # Validation #
    ##############

    def validation_model(self, number_ev_pred_on_hotspots_sum, total_ev_pred):
        """
        Calculate the hit rate for the coveraged area for the simulated events
        :param sim_points_filtered: two dimensional array with filtered events according to the porcentage area covered
        :param sim_points_without_filtered: two dimensional array with events without filtered
        :return h_r: hit rate metric for the simulated events
        """
        logging.debug("Comienza la validacion de datos para el modelo de rinas de seguridad.") 
  
        update_process_state(self.tipos_proceso[NAME_VALIDACION], self.estados_ejecucion[ESTADO_PROCESO], get_token_acces())
        try:
            h_r = number_ev_pred_on_hotspots_sum/total_ev_pred
            file = open("validacion.txt", "w")
            file.write(str(h_r*100) )
            file.close()  
            logging.debug("Termina la validacion de datos para el modelo de rinas de seguridad.")
            update_process_state(self.tipos_proceso[NAME_VALIDACION], self.estados_ejecucion[ESTADO_EXITO], get_token_acces())
            return  h_r
        except Exception as e:
            update_process_state(self.tipos_proceso[NAME_VALIDACION], self.estados_ejecucion[ESTADO_ERROR], get_token_acces())
            msg_error = "No se completo la validacion de datos para el modelo de rinas de seguridad."   
            logging.error(msg_error)
            raise Exception(msg_error + " / " + str(e))
