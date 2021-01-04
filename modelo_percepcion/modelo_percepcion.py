import seaborn as sns
import numpy as np
import pandas as pd
from scipy.optimize import leastsq
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta
import pickle

def kernel_zhao_vec(s, s0=300/3600, theta=0.242):
    """
    Calculates Zhao kernel for given value.
    Optimized using nd-arrays and vectorization.

    :param s: time points to evaluate, should be a nd-array
    :param s0: initial reaction time
    :param theta: empirically determined constant
    :return: values at given time points
    """
    c0=1.0/(s0*(1.0+1.0/theta)) # normalization constant
    res = np.copy(s)
    res[s < 0] = 0
    res[(s <= s0) & (s >= 0)] = c0
    res[s > s0] = c0 * (res[s > s0] / s0) ** (-(1.0+theta))
    return res

def kernel_primitive_zhao_vec(x, s0=300/3600, theta=0.242):
    """
    Calculates the primitive of the Zhao kernel for given values.
    Optimized using nd-arrays and vectorization.

    :param x: points to evaluate, should be a nd-array
    :param s0: initial reaction time
    :param theta: empirically determined constant
    :param c0: normalization constant
    :return: primitives evaluated at given points
    """
    c0 = 1.0/(s0*(1.0+1.0/theta)) # normalization constant
    res = np.copy(x)
    res[x < 0] = 0
    res[(x <= s0) & (x >= 0)] = c0 * res[(x <= s0) & (x >= 0)]
    res[x > s0] = s0*c0*(1.0+1.0/theta*(1-(res[x > s0]/s0)**-theta))
    return res




def integral_zhao(x1, x2, s0=300/3600, theta=0.242):
    """
    Calculates definite integral of Zhao function.

    :param x1: start
    :param x2: end
    :param s0: initial reaction time
    :param theta: empirically determined constant
    :return: integral of Zhao function
    """
    return kernel_primitive_zhao_vec(x2, s0, theta) - kernel_primitive_zhao_vec(x1, s0, theta)

    
def compare_vectors(a,b):
    """
    Compara dos vectores para verificar si son el mismo o no

    :param a: vector uno
    :param a: vector dos
    :return: veracidad igualdad entre los vectores
    """
    if np.linalg.norm(abs(a-b)) == 0:
        return True
    else:
        return False
    
def get_particion(inicio,fin,T_c,num=50):

    """
    Obtiene una particion temporal a partir de un t-inicial y t-final y sus valores dada la funcion T_c

    :param inicio: tiempo inicial
    :param fin: tiempo final
    :param num: cantidad valores particion inicial
    :return: vector de longitudes particion, vector valores intermedios particion
    """
    particion=[]
    particion.append(inicio)
    valor_actual=T_c(inicio)
    for i in np.linspace(inicio,fin,num=num):
        valor=T_c(i)
        if compare_vectors(valor,valor_actual)== False:
            particion.append(i)
            valor_actual=valor
    particion=np.array(particion)
    if len(particion) == 1:
        return np.array([fin-inicio]),np.array([(inicio+fin)/2])
    else:
        pares=np.array(list(zip(particion[:-1],particion[1:])))
    return pares[:,1]-pares[:,0],pares.sum(axis=1)/2





## calculo de p_i gorro

def get_event_count(event_times, start, end,*args):
    """
    Count of events in given interval.

    :param event_times: nd-array of event times
    :param start: interval start
    :param end: interval end
    :return: count of events in interval
    """
    mask = (event_times >= start) & (event_times <= end)
    return mask.sum(*args)

def estimate_infectious_rate_constant_vec(event_times, follower, t_start, t_end, kernel_integral, count_events=None):
    """
    Returns estimation of infectious rate for given event time and followers on defined interval.
    Optimized using numpy.

    :param event_times: nd-array of event times
    :param follower: nd-array of follower counts
    :param t_start: time interval start
    :param t_end: time interval end
    :param kernel_integral: integral function of kernel function
    :param count_events: count of observed events in interval (used for time window approach)
    :return: estimated values for infectious rate
    """
#     print(event_times,follower)
    
    if (len(follower) == 1) & (sum(follower) == 0):
        return 0
    
    kernel_int = follower * kernel_integral(t_start - event_times, t_end - event_times)
#     print(count_events)
    if count_events is not None:
        return count_events / kernel_int.sum()
    else:
        return event_times.size / kernel_int.sum()
    

def estimate_infectious_rate_vec(event_times, follower, kernel_integral=integral_zhao, obs_time=24,num=20):
    """
    Estimates infectious rate using moving time window approach.
    Optimized using numpy and vectorized approach.

    :param event_times: nd-array of event times
    :param follower: nd-array of follower counts
    :param kernel_integral: function for calculating the integral of the kernel
    :param obs_time: observation time
    :param window_size: bin width for estimation (in hours)
    :param window_stride: interval for moving windows (in hours)
    :return: 3-tuple holding list of estimated infectious rate for every moving time window, event counts for every
    window and the time in the middle of every window
    """
    t0=event_times[0]
    time_x=np.linspace(t0, obs_time,num=num)
    values=[]
    for i in range(len(time_x)-1):
        
        start=time_x[i]
        end=time_x[i+1]
        mask = event_times < end 
        count_current = get_event_count(event_times, start, end)
        est = estimate_infectious_rate_constant_vec(event_times[mask],
                                                    follower[mask],
                                                    start,
                                                    end,
                                                    kernel_integral,
                                                    count_current)
        values.append(est)
#     print(values)
    if (len(follower) == 1) & (sum(follower) == 0):
        values=values[:1]
    else:
        values=np.trim_zeros(np.array(values))
    return values,time_x[:len(values)]



def infectious_rate_tweets_vec(t, r0=0.424, phi0=0.125, taum=2., tm=24., t0=0,p0=3):
    """
    Alternative form of infectious rate from paper. Supports bounds for r0 and taum. Bound should be passed as an array
    in the form of [(lower r0, lower taum), (upper r0, upper taum)].
    Converted to hours.
    Vectorized version.

    :param t: points to evaluate function at, should be a nd-array (in hours)
    :param p0: base rate
    :param r0: amplitude
    :param phi0: shift (in days)
    :param taum: decay/freshness (in days)
    :param t0: start time of observation (in hours)
    :param tm: cyclic property (after what time a full circle passed, in hours)
    :return: intensities for given points
    """
    return p0*(1-r0*np.sin(2*np.pi/tm*(t+phi0)))*np.exp(-(t-t0)/taum)

def loss_function(params, estimates, fun,p0_vec):
    """
    Loss function used by least squares.

    :param params: current values of parameters to fit
    :param estimates: estimates of function to fit
    :param fun: function to fit
    :return: array of loss values for every estimation
    """
    diffs=[]
    for i in estimates:
        est=estimates[i][0]
        xval=estimates[i][1] 
        t0=xval[0]
        diffs.append(abs(fun(xval,t0=t0,p0=p0_vec[i],*params)-est))
    return np.concatenate(diffs)


def fit_parameter(estimates, fun, p0_vec,start_values=None):
    """
    Fitting any numbers of given infectious rate function using least squares.
    Used count of observed events in observation window as weights.

    :param estimates: estimated values of function
    :param fun: function to fit
    :param start_values: initial guesses of parameters, should be a ndarray
    :return: fitted parameters
    """
    from scipy.optimize import leastsq
    if start_values is None:
        start_values = np.array([0, 0, 1.])

    return leastsq(func=loss_function, x0=start_values, args=(estimates, fun,p0_vec))[0]

def error(estimated,param_fitted,fun,p0_vec):
    total_estimated=[]
    total_fitted=[]
    for i in estimated:
        total_estimated.append(estimated[i][0])
        xval=estimated[i][1] 
        total_fitted.append(fun(xval,t0=xval[0],p0=p0_vec[i],*param_fitted))
    total_estimated=np.concatenate(total_estimated)
    total_fitted=np.concatenate(total_fitted)
    
    return abs(total_estimated-total_fitted).sum()/sum(abs(total_estimated))





def get_predictions(data_tweets,inicio,time_observed,time_pred):
    '''
    Funcion que permite crear predicciones sobre la cantidad de tweets a partir de los datos de los tweets el tiempo inicial, tiempo observado y tiempo a predecir
    
    entradas: 
    
        data_tweets: diccionario con la informacion centralizada de los tweets.
        inicio: tiempo inicial en horas
        time_observed: tiempo de obserevacion
        time_pred: time_pred
        
    ########
    
    Salida:
        beta: parametros covariados
        param_fit: parametros funcion influencia
        predict: prediccion cantidad tweets
        x_time: tiempo de prediccion
    
    
    
    '''
    
#     with open('data_tweets.pickle', 'rb') as handle:
#         data = pickle.load(handle)
    data=data_tweets
    
    keys_tweets=[]
    for i in data['Tweets']:
        if ((data['Tweets'][i]['times']>=inicio) & (data['Tweets'][i]['times']<=time_pred)).sum() > 0:
            keys_tweets.append(i)
            
     ## covariados
    
    
    def T_c(t,f_inicio=data['Inicio']):
        """
        Calcula los covariados para un tiempo especifico t

        :param t: tiempo a evaluar
        :param f_inicio: fecha inicial datos completos, formato datetime
        :return: vector covariados para el valor t
        """
        t=f_inicio+timedelta(hours=t)
        return np.array([t.weekday(),(t.hour > 12)*1])
            
     ## Calculo de Beta
    
     ## Calculo de Beta
    def to_solve(beta,particion,left_hand):
        l_particion,v_particion=particion
        right_hand=(l_particion.reshape(len(v_particion),1)*np.array([T_c(i)*np.exp(np.dot(beta,T_c(i))) for i in v_particion])).sum(axis=0)
        return right_hand-left_hand

    def get_beta(beta_0,particion,left_hand):
        return minimize(lambda x: np.linalg.norm(to_solve(x,particion,left_hand)),
                        x0=beta_0,
                        method='Nelder-Mead').x


    ### left_hand
    left_hand=np.zeros_like(T_c(0))
    for i in keys_tweets:
        t_0=data['Tweets'][i]['times'][0]
        if t_0 <= time_observed:
            left_hand+=T_c(t_0)
    ### right_hand     
    l_particion,v_particion=get_particion(inicio,time_observed,T_c)

    
    beta=get_beta(np.random.rand(len(left_hand)),
                  (l_particion,v_particion),
                  left_hand)

    ## calculo de p_i gorro
    p_est={}
    for i in keys_tweets:
        tweet=data['Tweets'][i]
        event_times=tweet['times']
        t0=event_times[0]
        S=tweet['sentiment']
        followers=tweet['followers']
        if (t0<=time_observed) & (t0>=inicio):
            mask= event_times <= time_observed
            p_i_est,t_points=estimate_infectious_rate_vec(event_times[mask], followers[mask], integral_zhao, time_observed,num=25)
            p_est[i]=[p_i_est,t_points]
            
    p0_vec={}
    for i in p_est:
        p0_vec[i]=data['Tweets'][i]['sentiment']
        
    param_fit=fit_parameter(p_est,infectious_rate_tweets_vec,p0_vec)
    error_=error(p_est,param_fit,infectious_rate_tweets_vec,p0_vec)
    
    x_time=np.linspace(time_observed,time_pred)
    back_g=[np.exp(np.dot(beta,T_c(i))) for i in x_time]
    
    replicas=np.zeros_like(x_time)
    for i in keys_tweets:
        tweet=data['Tweets'][i]
        event_times=tweet['times']
        t0=event_times[0]
        S=tweet['sentiment']
        followers=tweet['followers']
        if (t0<=time_observed) & (t0>=inicio):
            sum_int=followers*kernel_zhao_vec(x_time.reshape((len(x_time),1))-event_times)
            sum_int=sum_int.sum(axis=1)    
            p_t=infectious_rate_tweets_vec(x_time,t0=t0,p0=S,*param_fit)
            sum_ext=np.nan_to_num(sum_int*p_t)
            replicas+=sum_ext
    
    L=back_g+replicas
    dt=(x_time[-1]-x_time[0])/50
        
        
    return beta, param_fit, L*dt, x_time