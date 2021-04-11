import math
import numpy as np
import random as rd
from random import randint
from scipy.optimize import fsolve
import scipy.optimize as optimize
import pandas as pd
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
import timeit
import datetime

start = timeit.default_timer()
######################################################################

############################################################
####### IMPLEMENTACIÓN DEL MODELO MATEMATICO DE SEPP #######
############################################################
e = math.e
pi = math.pi

data = pd.read_csv('union_dataframes.csv')
data = data[(data['FECHA'] >= '2018-01-01 00:00:00') & (data['FECHA'] <= '2018-01-31 23:59:59')]
data['TimeStamp'] = ' '
for i in range(0, len(data)):
    data.TimeStamp.iloc[i] = datetime.datetime.fromisoformat(str(data.FECHA.iloc[i])).timestamp()
data['geometry'] = data['geometry'].apply(wkt.loads)
data.TimeStamp = (data.TimeStamp-data.TimeStamp.iloc[0])/3600+0.0001
data = gpd.GeoDataFrame(data, geometry = 'geometry')
data = data.rename(columns={'LONGITUD': 'Long', 'LATITUD': 'Lat'})
data.cov1 = (data.cov1-data.cov1.min())/(data.cov1.max()-data.cov1.min())
data.cov2 = (data.cov2-data.cov2.min())/(data.cov2.max()-data.cov2.min())

data['X'] = data.geometry.x
data['Y'] = data.geometry.y
                          
                               ####################################
                               ###### DIVISION DEL DATAFRAME ######  
                               ####################################

initial_date = np.array(['2018-01-01 00:00:00', '2018-01-22 00:00:00', '2018-01-24 00:00:00', '2018-01-26 00:00:00',
                      '2018-01-28 00:00:00','2018-01-30 00:00:00']) 

final_date = np.array(['2018-01-21 23:59:59', '2018-01-23 23:59:59', '2018-01-25 23:59:59', '2018-01-27 23:59:59',
                      '2018-01-29 23:59:59','2018-01-31 23:59:59']) 

def date(i_date, f_date):
    return data.loc[(data['FECHA'] >= i_date) & (data['FECHA'] <= f_date)]
                
data = date(initial_date[0], final_date[0])

###########################################################################################
stop = timeit.default_timer()
execution_time = stop - start
print("1ra Parte: Program Executed in "+str(execution_time)) # It returns time in seconds  


                        ##################################################
                        ###### IMPLEMENTACION DEL MODELO MATEMATICO ######  
                        ##################################################


start = timeit.default_timer()
######################################################################

########################
###### Parameters ######            
########################

#number_events = len(data)         # number of events      
omega = 1000                         # temporary decay
iomega = 1/omega
sigma2 = 10                        # spatial decay
T = data.TimeStamp.iloc[-1]        # temporal window of the experiment
sc = 10000                          # size cell
beta = np.array([10,10])             # weight of covariates


cte1 = 1/(2*sigma2)
cte2 = 1/(2*pi*omega*sigma2)
cte3 = T*sc


#############################
###### Event Positions ######
#############################
lon = data.X.to_numpy()
lat = data.Y.to_numpy()

##########################################
###### Time of occurrence of events ######
##########################################
t = data.TimeStamp.to_numpy()


###############################
#   Declarations to improve   #
#       execution speed       #
###############################
ncovs = 2                      # number of covariates    

#cov_ev = data[['cov1','cov2']].to_numpy()
data_ar = data.to_numpy()
covs_ar = data[['cov1','cov2']].to_numpy()

#####################################
###### Distance between events ######
#####################################

dist2_ij = np.array([[((data.X[i]-data.X[j])**2+(data.Y[i]-data.Y[j])**2)/1e6 for j in range(0,len(data))] 
                     for i in range(0,len(data))])

                                   #####################################
                                   ###### Parameters optimization ######
                                   #####################################

############# beta optimization #############
def syst(beta):
  beta1 = beta[0]
  beta2 = beta[1]
  f1 = sum(pbe*data_ar[:,5])-cte3*sum(covs_ar[:,0]*e**(covs_ar*beta).sum(axis=1))
  f2 = sum(pbe*data_ar[:,6])-cte3*sum(covs_ar[:,1]*e**(covs_ar*beta).sum(axis=1))
  return (f1, f2)

def root_beta(beta):
  return fsolve(syst, beta)

############# sigma2 optimization #############
def update_sigma2(sigma2):
  """ function that updates the value of sigma2 """
  num_sigma2 = 0
  for i in range(0, len(data)):
    for j in range (0, len(data)):
      num_sigma2 +=  p_ij[i,j]*dist2_ij[i,j]
  sigma2 = 0.5*num_sigma2/sum(pte)
  return sigma2

############# omega optimization #############
def funct_omega(new_omega):
  term1, suma = 0, 0
  inew_omega = -1/new_omega
  """ function that calculates the right side of the omega optimization equation """
  for i in range(0, len(data)):
    for j in range(0, len(data)):
      term1 += p_ij[i,j]*(t[i]-t[j])
    suma += e**(inew_omega*(T-t[i])) * (T-t[i])
  return (suma+term1)/sum(pte)

def new_omega(omega):
  while abs(omega-funct_omega(omega)) > 1e-5:
    omega = funct_omega(omega);
  return omega
#################################################################################
stop = timeit.default_timer()
execution_time = stop - start
print("Program Executed in "+str(execution_time)) # It returns time in seconds  


                       #########################################################
                       ###### OPTIMIZACION Y CALCULO DE LAS PROBABILIDADES #####
                       #########################################################

start = timeit.default_timer()
######################################################################

n_iter = 200

for n in range(0, n_iter): 
  cte1 = 1/(2*sigma2)
  cte2 = 1/(2*pi*omega*sigma2)
  cte3 = T*sc
  # kernel to know if event i is triggered by event j
  iomega = 1/omega
  # correlation function between events i and j
  g_ij = np.array([[e**(-(t[i]-t[j])*iomega) * e**(-dist2_ij[i][j]*cte1) * cte2 
                    if t[j] < t[i] else 0 for j in range(0,len(data))] for i in range(0,len(data))]) 
  # sum of kernels for each event i
  sum_kernels = np.array([sum(g_ij[i]) for i in range(0,len(data))]) 
  # background for the events i
  mu = np.array([e**((covs_ar * beta).sum(axis=1)[i]) for i in range(0, len(data))])
  # conditional intensity of the events i
  Lambda = mu + sum_kernels
  # probability of event i to be triggered by event j
  p_ij = np.array([[g_ij[i,j]/Lambda[i] for j in range(0, len(data))] for i in range(0, len(data))])
  # probability of event i to be triggered for some event j
  pte = sum_kernels/Lambda
  # probability of background event
  pbe = mu/Lambda
  if (abs(omega-new_omega(omega)) > 1e-5):
    print('diferencia omega = {}'.format(abs(omega-new_omega(omega))))
    nomega = new_omega(omega)
    print('new omega = {}'.format(omega))
    print(n)
    print('------------') 
  else:
    pass
  if (abs(beta-root_beta(beta)).all()) > 1e-5:
    print('diferencia betas = {}'.format(abs(beta-root_beta(beta))))
    nbeta = root_beta(beta)
    print('new beta = {}'.format(beta))
    print(n)
    print('------------')
  else:
    pass
  if (abs(sigma2-update_sigma2(sigma2))) > 1e-5:
    print('diferencia sigma2 = {}'.format(abs(sigma2-update_sigma2(sigma2))))
    nsigma2 = update_sigma2(sigma2)
    print('new sigma2 = {}'.format(sigma2))
    print(n)
    print('------------') 
  else:
    pass
  omega = nomega
  beta = nbeta
  sigma2 = nsigma2
  
print('The beta value is:{}'.format(beta))
print('The sigma2 value is:{}'.format(sigma2))
print('The omega value is:{}'.format(omega))
#################################################################################
stop = timeit.default_timer()
execution_time = stop - start
print("Program Executed in "+str(execution_time)) # It returns time in seconds
            

