__name__ = "Class for the SEPP Model"

#%matplotlib inline
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import gmaps
import gmaps.datasets
import scipy.stats
import utils
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


class ModeloBase:
    
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
    :param poligonos_df: dataframe with the polygons of each cell of grid ant its resepctive covariate values
    :return beta: optimization of the parameter related with the constant background rate
    :return omega: optimization of the parameter related with ocurrence of triggering events
    :return sigma2: optimization of the parameter related with the spread of triggering events
    :return puntos_gdf: dataframe with the simulated events
    :return array_cells_events_sim: two dimensional array with the number of cell and the number of events ocurred on it
    :return h_r: the validation process is make with the hit rate metric for the simulated events
    """

    def __init__(self, setting, events_gdf, cov_norm_cell_m, cov_norm_eventos_m, temp_factor, sp_factor, number_hours, poligonos_df):
        self.setting = setting
        self.events_gdf = events_gdf
        self.cov_norm_cell_m = cov_norm_cell_m
        self.cov_norm_eventos_m = cov_norm_eventos_m
        self.temp_factor = temp_factor
        self.sp_factor = sp_factor
        self.number_hours = number_hours
        self.poligonos_df = poligonos_df
        
        
    #########################################
    # Training Process (Parameter fitting) #
    ########################################
    
    def train_model(self):
        """
        Calculate the optimized values for the conditional intensity for the SEPP model
     
        :param setting: 0 if the data are real data, 1 if the data are simulated data
        :param events_gdf: dataframe with the events data for the training model
        :param cov_norm_cell_m: array of covariates of each cell of the grid 
        :param cov_norm_eventos_m: array of covariates of each cell where the events ocurred
        :param temp_factor: temporal scale for the time data of the dataframe
        :param sp_factor: spatial scale for the spatial time as polygon area and distance between events
        :return beta: optimization of the parameter related with the constant background rate
        :return omega: optimization of the parameter related with ocurrence of triggering events
        :return sigma2: optimization of the parameter related with the spread of triggering events
        """
        x_eventos = self.events_gdf.X.to_numpy()
        y_eventos = self.events_gdf.Y.to_numpy()
        if self.setting == 0:          #siedco data
            tiempo_eventos = self.events_gdf.TimeStamp.to_numpy()
            tiempo_eventos = tiempo_eventos/(self.temp_factor*3600)
        if self.setting == 1:          #simulation data
            tiempo_eventos = self.events_gdf.TimeStamp.to_numpy()

        beta = np.ones(78)[0:len(self.cov_norm_cell_m[0])]
        omega = 1
        sigma2 =  1
        T = tiempo_eventos[-1]
        sc = 0.09 

        #===========================
        #== Parameter Optmization ==
        #===========================
        
        #==============
        #== beta opt == 
        #==============

        def syst(beta):
            """
            Generates the equation set to optimized beta
            :param beta: initial value of beta (array)
            :return equations: the equations set
            """
            equations = np.array([])
            for i in range(0, len(beta)):
                equations = np.append(equations,np.dot(pbe,self.cov_norm_eventos_m[:,i].astype(float))-
                                      T*sc*self.sp_factor*np.dot(self.cov_norm_cell_m[:,i].astype(float),
                                      np.exp((beta * self.cov_norm_cell_m.astype(float)).sum(axis=1))))
            return equations
        
        def root_beta(beta):
            """
            Solves the equations set
            :param beta: solution of the equations set
            """
            return fsolve(syst, beta, xtol=10e-10, maxfev=1000000)

        #================
        #== sigma2 opt == 
        #================

        def update_sigma2(sigma2):
            """
            Solves the equation for squared sigma
            :param sigma2: initial value of sigma2
            :return sigma2: solution of the equation for sigma2
            """
            num_sigma2 = 0
            for i in range(0, len(self.events_gdf)):
                for j in range (0, len(self.events_gdf)):
                    num_sigma2 +=  p_ij[i,j] * dist2_eventos[i,j] 
            sigma2 = 0.5 * num_sigma2 * self.sp_factor/sum(pte)
            return sigma2
    
        #===============
        #== omega opt ==
        #===============
        
        def funct_omega(omega):
            """
            Equation to solve for omega
            :param omega: initial value of omega
            :return __: equation to solve
            """
            term1, suma = 0, 0
            for i in range(0, len(self.events_gdf)):
                for j in range(0, len(self.events_gdf)):
                    term1 += p_ij[i,j]*(tiempo_eventos[i]-tiempo_eventos[j])
                suma += np.exp(-omega*(T-tiempo_eventos[i])) * (T-tiempo_eventos[i])
            return sum(pte)/(suma+term1)

        def new_omega(omega):
            """
            Solves the equation for omega 
            :param omega: initial value of the omega
            :return omega: solution of the equation for omega
            """
            while abs(omega-funct_omega(omega)) > 1e-5:
                omega = funct_omega(omega)
            return omega 

        # distance between events
        dist2_eventos = np.empty((len(self.events_gdf), len(self.events_gdf)))
        for i in range(0, len(self.events_gdf)):
            for j in range(0, len(self.events_gdf)):
                dist2_eventos[i][j] = ((x_eventos[i]-x_eventos[j])**2 + (y_eventos[i]-y_eventos[j])**2)/1e+6

        n_iter = 700
        p = 0
        while True:
            cte1 = 1/(2 * sigma2)
            cte2 =  omega /(2 * np.pi *  sigma2)
            
            # kernel matrix between the events j and j
            # the value [i][j] measure the correlation between the events i and j
            g_ij = np.empty((len(self.events_gdf), len(self.events_gdf)))
            for i in range(0, len(self.events_gdf)):
                for j in range(0, len(self.events_gdf)):
                    if (tiempo_eventos[j] < tiempo_eventos[i]):
                        g_ij[i][j] = cte2 * np.exp(-omega*(tiempo_eventos[i]-tiempo_eventos[j]))*np.exp(-self.sp_factor*dist2_eventos[i][j]*cte1)  
                    else:
                        g_ij[i][j] = 0
            # sum of kernel for each event
            sum_kernels = np.array([sum(g_ij[i]) for i in range(0, len(self.events_gdf))]) 
            # ocurrence rate for background events
            mu = np.exp((beta * self.cov_norm_eventos_m.astype(float)).sum(axis=1))
            # conditional intensity of the events
            Lambda = mu + sum_kernels
            # prob for event i was triggering for the event j
            p_ij = g_ij/Lambda
            # prob for event i was triggering for some event ocurred before i
            pte = sum_kernels/Lambda
            # prob of background event
            pbe = mu/Lambda 
            if (abs(sigma2-update_sigma2(sigma2)) >  1e-5) and (abs(beta-root_beta(beta)).all() > 1e-5) and (abs(omega-new_omega(omega)) > 1e-5):
                p += 1
                nbeta = root_beta(beta)
                nomega = new_omega(omega)
                nsigma2 = update_sigma2(sigma2)
                beta = nbeta
                omega = nomega
                sigma2 = nsigma2
            else:
                break
               
        return beta, omega, sigma2

    ##############
    # Simulation #
    ##############

    def predict_model(self):
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
        window_size = self.number_hours/self.temp_factor
        lens = np.array([])
        data_simul = np.array([])
        Event = namedtuple("Event", ["t", "loc_x", "loc_y"])
        parameters = self.train_model()
        beta = parameters[0]
        omega = parameters[1]
        sigma2 = parameters[2]
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
            :param poly: goemetry of the polygon
            :return p: point inside of polygon
            """
            minx, miny, maxx, maxy = poly.bounds
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
                    minx, miny, maxx, maxy = poly.bounds
                    while True:
                        loc_x = np.random.normal(loc=parent.loc_x, scale=sigma)
                        loc_y = np.random.normal(loc=parent.loc_y, scale=sigma)
                        p = Point(loc_x, loc_y)
                        if poly.contains(p):
                            return p
                p = get_random_point_in_polygon_trig(self.poligonos_df.geometry[k])
                loc_x = p.x
                loc_y = p.y
                points.append(Event(parent.t + np.log(alpha/abs(alpha-t*omega))/omega , loc_x, loc_y))

        def _add_point(points, mu):
            """
            Generates time deltas between consecutive events and adds it to a list
            :param points: list of points
            :param mu: events rate for the cell
            """
            wait_time = np.random.exponential(1/mu)
            if len(points) == 0:
                last = 0
            else:
                last = points[-1]
            points.append(last + wait_time)
        
        def sample_poisson_process_hom(window_size, mu):
            """
            Generates the background events according to the background events rate in some window temporal size 
            :param window_size: window temporal size
            :param mu: background events rate
            :return points: simulated background points
            """
            points = []
            _add_point(points, mu)
            while points[-1] < window_size:
                _add_point(points, mu)
            return points

        def simulate(window_size,k):
            """
            Generates the background events with their descendant events in some window temporal size in one cell
            :param window_size: window temporal size
            :param k: number of cell
            :return puntos_gdf: dataframe with the simulated events
            :return array_cells_events_sim: two dimensional array with cell number and its corresponding number of events
            """
            backgrounds = sample_poisson_process_hom(window_size, 1/mu[k])
            backgrounds = backgrounds[:-1]
            points = []
            for i in range(0, len(backgrounds)):
                t = backgrounds[i]
                p = get_random_point_in_polygon_back(self.poligonos_df.geometry[k])
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
            return points,  backgrounds, caused_by

        events_sim_on_cells = np.array([])
        all_events_sim = np.array([])

        mu = np.exp((beta*self.cov_norm_cell_m).sum(axis=1).astype(float))
        
        
        for k in range(0, len(self.cov_norm_cell_m)):
            ev = simulate(window_size,k)[0]
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
        #creation of the elements [cell, number of events in this cell] and
        #  # save it in array_cells_events_sim
        #    array_cells_events_sim.append([i, int(events_sim_on_cells[i])])
        array_cells_events_sim = list(map(list,zip(array_cells_events_sim,events_sim_on_cells)))
        
        return puntos_gdf, array_cells_events_sim

    
    ##############
    # Validation #
    ##############

    def validation_model(self, sim_points_filtered, sim_points_without_filtered):
        """
        Calculate the hit rate for the coveraged area for the simulated events
        :param sim_points_filtered: two dimensional array with filtered events according to the porcentage area covered
        :param sim_points_without_filtered: two dimensional array with events without filtered
        :return h_r: hit rate metric for the simulated events
        """
        h_r = len(sim_points_filtered)/len(sim_points_without_filtered)
        
        return h_r

    


            
        
        
    


    