import seaborn as sns
import numpy as np
import os
import pandas as pd
from scipy.optimize import leastsq
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta
import pickle
from scipy import stats
import logging


#----------------------------------------------------------------------------------------

'''
Funciones relacionadas al kernel
'''
def kernel_zhao_vec(s, s0=300/3600, theta=0.242):
    """
    Calculates Zhao kernel for given value.
    Optimized using nd-arrays and vectorization.

    :param s: time points to evaluate, should be a nd-array
    :param s0: initial reaction time
    :param theta: empirically determined constant
    :return: values at given time points
    """
    try:
        c0=1.0/(s0*(1.0+1.0/theta)) # normalization constant
        res = np.copy(s)
        res[s < 0] = 0
        res[(s <= s0) & (s >= 0)] = c0
        res[s > s0] = c0 * (res[s > s0] / s0) ** (-(1.0+theta))
        return res
    except:
        msg_error="No se completo operación en la funcion  kernel_zhao_vec"
        logging.error(msg_error)
        raise Exception(msg_error)

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
    try:
        c0 = 1.0/(s0*(1.0+1.0/theta)) # normalization constant
        res = np.copy(x)
        res[x < 0] = 0
        res[(x <= s0) & (x >= 0)] = c0 * res[(x <= s0) & (x >= 0)]
        res[x > s0] = s0*c0*(1.0+1.0/theta*(1-(res[x > s0]/s0)**-theta))
        return res
    except:
        msg_error="No se completo operación en la funcion  kernel_primitive_zhao_vec"
        logging.error(msg_error)
        raise Exception(msg_error)

def integral_zhao(x1, x2, s0=300/3600, theta=0.242):
    """
    Calculates definite integral of Zhao function.

    :param x1: start
    :param x2: end
    :param s0: initial reaction time
    :param theta: empirically determined constant
    :return: integral of Zhao function
    """
    try:
        return kernel_primitive_zhao_vec(x2, s0, theta) - kernel_primitive_zhao_vec(x1, s0, theta)
    except:
        msg_error="No se completo operación en la funcion  integral_zhao"
        logging.error(msg_error)
        raise Exception(msg_error)

#----------------------------------------------------------------------------------------

def T_c(t):
    """
    Funcion ejemplo covariados
    Dada una fecha en datetime devuelve un vector con sus valores de 
    covariados.
    """    
    try:
        return np.array([t.weekday()/6.0,(t.hour > 12)*1,1])
    except:
        msg_error="No se completo operación en la funcion  T_c"
        logging.error(msg_error)
        raise Exception(msg_error)


def date_to_hours(date,f_inicio):
    """
    Calcula la cantidad de horas que hay entre una fecha dada y una fecha inicial.
    
    :param date: fecha en formato datetime
    :param f_inicio: fecha inicial en datetime
    :return: horas entre las fechas formato int
    """    
    try:
        date = datetime.fromisoformat(date)
        hours = (date-f_inicio).total_seconds()/3600
        return hours
    except:
        msg_error="No se completo operación en la funcion  date_to_hours"
        logging.error(msg_error)
        raise Exception(msg_error)

def restore_date(t,f_inicio):
    """
    Restaura la fecha original de un evento a partir de una fecha inicial y 
    el tiempo en horas transcurrido.
    
    :param t: tiempo en horas int
    :param f_inicio: fecha inicial en datetime
    :return: fecha original
    """    
    try:
        return f_inicio+timedelta(hours=t)
    except:
        msg_error="No se completo operación en la funcion  restore_date"
        logging.error(msg_error)
        raise Exception(msg_error)

def compare_vectors(a,b):
    """
    Establece si dos vectores (np.array) son iguales
    
    :param a: primer vector
    :param b: segundo vector
    :return: True si son iguales y False en caso contrario
    """
    try:
        if np.linalg.norm(abs(a-b)) == 0:
            return True
        else:
            return False
    except:
        msg_error="No se completo operación en la funcion  compare_vectors"
        logging.error(msg_error)
        raise Exception(msg_error)
def get_particion(inicio,fin,f_covariados,win_size=1):
    
    """
    Calcula los tamanos y puntos medios de la particion generada 
    en un intervalo por los valores de sus covariados.
    
    :param inicio: tiempo en horas del inicio del intervalo a evaluar
    :param fin: tiempo en horas del fin del intervalo a evaluar
    :param f_covariados: funcion de covariados
    :param win_size: saltos para evaluacion nuevos intervalos
    :return: lista de longitud intervalos de la particion y 
             puntos medios de esta (np.array)
    """
    try:
        particion=[]
        particion.append(inicio)
        valor_actual=f_covariados(inicio)
        for i in np.linspace(inicio,fin,num=int((fin-inicio)/win_size)):
            valor=f_covariados(i)
            if compare_vectors(valor,valor_actual) == False:
                particion.append(i)
                valor_actual=valor
        particion=np.array(particion)
        if len(particion) == 1:
            return np.array([fin-inicio]),np.array([(inicio+fin)/2])
        else:
            pares=np.array(list(zip(particion[:-1],particion[1:])))
        return pares[:,1]-pares[:,0],pares.sum(axis=1)/2
    except:
        msg_error="No se completo operación en la funcion  get_particion"
        logging.error(msg_error)
        raise Exception(msg_error)


#----------------------------------------------------------------------------------------
## Calculo de Beta

def beta_left_hand(keys_train,Tweets,f_covariados):
    """
    Calcula la parte izquierda de la ecuacion a resover de beta a partir de
    los tweets originales y sus covariados temporales
    
    :param keys_train: lista de keys de los tweets que se tienen en cuenta
    :param Tweets: diccionario con la informacion de los tweets
    :param f_covariados: funcion de covariados
    :return: vector resultante 
    """    
    try:
        left_hand=np.zeros_like(f_covariados(0))
        for i in keys_train:
            t_0=Tweets[i]['times'][0]        
            left_hand+=f_covariados(t_0)
        return left_hand
    except:
        msg_error="No se completo operación en la funcion  beta_left_hand"
        logging.error(msg_error)
        raise Exception(msg_error)


def to_solve(beta,particion,left_hand,f_covariados):
    """
    Calcula la diferencia entre la parte izquierda y derecha de la ecuacion
    para establecer Beta
    
    :param beta: valor inicial de beta (np.array)
    :param particion: particion precalculada
    :param left_hand: valor parte izquierda precalculada
    :param f_covariados: funcion de covariados
    :return: vector resultante diferencia
    """       
    try:
        l_particion,v_particion=particion
        T_i=np.array([f_covariados(b) for b in v_particion])
        Ti=l_particion*np.exp(np.dot(T_i,beta))

        right_hand=(Ti.reshape((len(Ti),1))*T_i).sum(axis=0)
        return right_hand-left_hand
    except:
        msg_error="No se completo operación en la funcion  to_solve"
        logging.error(msg_error)
        raise Exception(msg_error)

    
def get_beta(beta_0,particion,left_hand,f_covariados):
    """
    Determina el valor de beta optimo para los datos
    
    :param beta_0: valor inicial de beta (np.array)
    :param particion: particion precalculada
    :param left_hand: valor parte izquierda precalculada
    :param f_covariados: funcion de covariados
    :return: vector Beta optimo
    """      
    try:
        result= minimize(lambda x: np.linalg.norm(to_solve(x,particion,left_hand,f_covariados)),
                        x0=beta_0,
                        method='Powell')
        #print(result.fun)
        T_i=np.array([f_covariados(b) for b in particion[1]]).sum(axis=0)
        #print(T_i)
        result=result.x
        result[T_i == 0] = 0
        #print(np.linalg.norm(to_solve(result,particion,left_hand,f_covariados)))
        return result
    except:
        msg_error="No se completo operación en la funcion  get_beta"
        logging.error(msg_error)
        raise Exception(msg_error)

def Beta(inicio,time_observed,keys_start_in,Tweets,f_covariados,win_size=1,beta_0=np.array([1,1,1])):
    """
    Determina el valor de beta optimo para los datos datos sin hacer precalculos de
    los valores intermedios
    
    :param inicio: tiempo en horas donde se quiere iniciar el calculo de beta
    :param time_observed: tiempo en horas donde se quiere terminar el calculo de beta
    :param keys_start_in: lista de keys de tweets que se tienen en consideracion
    :param Tweets: diccionario con la informacion de los tweets
    :param f_covariados: funcion de covariados
    :param win_size: saltos para evaluacion nuevos intervalos para funcion particion
    :param beta_0: valor inicial de beta (np.array)
    :return: vector Beta optimo
    """      
    
    try:
        left_hand=beta_left_hand(keys_start_in,Tweets,f_covariados)
        particion=get_particion(inicio,time_observed,f_covariados,win_size=win_size)
        result = get_beta(beta_0,particion,left_hand,f_covariados)
        error = np.linalg.norm(to_solve(result,particion,left_hand,f_covariados))
        return result,error
    except Exception as e:
        msg_error="No se completo operación en la funcion  Beta"
        logging.error(msg_error)
        raise Exception(msg_error + " / " +str(e))

def back_ground(beta,t,f_covariados):
    """
    Calcula los valores de background para un beta, intervalo de tiempo y funcion
    de covariados dado
    
    :param beta: vector Beta dado
    :param t: intervalo de tiempo
    :param f_covariados: funcion de covariados
    :return: vector valores background
    """       
    try:
        return [np.exp(np.dot(beta,f_covariados(i))) for i in t]
    except:
        msg_error="No se completo operación en la funcion  back_ground"
        logging.error(msg_error)
        raise Exception(msg_error)
    
#----------------------------------------------------------------------------------------
## calculo de p_i gorro

def get_event_count(event_times, start, end,*args):
    """
    Count of events in given interval.

    :param event_times: nd-array of event times
    :param start: interval start
    :param end: interval end
    :return: count of events in interval
    """
    try:
        mask = (event_times >= start) & (event_times <= end)
        return mask.sum(*args)
    except:
        msg_error="No se completo operación en la funcion  get_event_count"
        logging.error(msg_error)
        raise Exception(msg_error)

def get_followers_sum(event_times,followers,start,end):
    """
    Count of followers in given interval.

    :param event_times: nd-array of event times
    :param followers: nd-array of followers
    :param start: interval start
    :param end: interval end
    :return: count of events in interval
    """    
    try:
        mask = (event_times >= start) & (event_times <= end)
        return followers[mask].sum()
    except:
        msg_error="No se completo operación en la funcion  get_followers_sum"
        logging.error(msg_error)
        raise Exception(msg_error)

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
    try:
        if sum(follower) == 0:
            return 0

        kernel_int = follower * kernel_integral(t_start - event_times, t_end - event_times)

        if count_events is not None:
            return count_events / kernel_int.sum()
        else:
            return event_times.size / kernel_int.sum()
    except Exception as e:
        msg_error="No se completo operación en la funcion  estimate_infectious_rate_constant_vec"
        logging.error(msg_error)
        raise Exception(msg_error + " / " +str(e))
    

def estimate_infectious_rate_vec(event_times, follower, kernel_integral=integral_zhao, obs_time=24,win_size=4):
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
    try:
        t0=event_times[0]
        num_=int((obs_time-t0)/win_size)
        time_x=np.linspace(t0, obs_time,num=num_)
        if len(time_x) < 2:
            time_x=np.array([t0,obs_time])



        values=[]
        foll=[]
        for i in range(len(time_x)-1):

            start=time_x[i]
            end=time_x[i+1]
            mask = event_times < end 
            count_current = get_event_count(event_times, start, end)
            foll+=[get_followers_sum(event_times,follower,time_x[i],time_x[i+1])]
            est = estimate_infectious_rate_constant_vec(event_times[mask],
                                                        follower[mask],
                                                        start,
                                                        end,
                                                        kernel_integral,
                                                        count_current)
            values.append(est)
    #     print(values)
        if sum(follower) == 0:
            values=values[:1]
        else:
            values=np.trim_zeros(np.array(values))
            if len(values) > 1:
                values=np.trim_zeros(values[:-1])


    #     return np.array(foll[:len(values)])*values,time_x[:len(values)]
        return values,time_x[:len(values)]
    except Exception as e:
        msg_error="No se completo operación en la funcion  estimate_infectious_rate_vec"
        logging.error(msg_error)
        raise Exception(msg_error + " / " +str(e))

def sigmoid_foll(foll,exp=3):
    """
    Normaliza valores de seguidores 
    
    :param foll: vector seguidores
    :param exp: umbral de normalizacion
    :return: vector seguidores normalizado
    """     
    #2/(1+np.exp(-10**(-exp)*foll))-1
    try:
        f=foll.copy()
        f[f>10**exp]=10**exp
        return f
    except:
        msg_error="No se completo operación en la funcion  sigmoid_foll"
        logging.error(msg_error)
        raise Exception(msg_error)

def compute_p_est(time_observed,keys_tweets,Tweets,kernel_integral=integral_zhao,followers_rate=2,win_size=4):
    """
    Calcula los valores estimados de la tasa de infeccion para los tweets en el tiempo observado
    
    
    :param time_observed: tiempo observado en horas
    :param keys_tweets: lista de keys de tweets que estan en el tiempo observado
    :param Tweets: diccionario de tweets
    :param kernel_integral: funcion de la integral del kernel utilizado
    :param followers_rate: funcion de la integral del kernel utilizado
    :param win_size: longitud ventanas temporales supuesto constantes
    :return: diccionario indexado por las keys de los tweets que contiene los valores estimados y los valores de los tiempos
    """        
    try:
        p_est={}
        for i in keys_tweets:
            tweet=Tweets[i]
            event_times=tweet['times']
            t0=event_times[0]
            S=tweet['sentiment']
            followers=sigmoid_foll(tweet['followers'],followers_rate)
            mask= event_times <= time_observed
            p_i_est,t_points=estimate_infectious_rate_vec(event_times[mask], followers[mask], kernel_integral, time_observed,win_size)
            if len(p_i_est) == 0:
                print(i)
                print(mask)
                print(event_times)
                print(event_times[mask])
                print(followers[mask])
            p_est[i]=[p_i_est,t_points]
        return p_est
    except Exception as e:
        msg_error="No se completo operación en la funcion  compute_p_est"
        logging.error(msg_error)
        raise Exception(msg_error + " / " +str(e))
#----------------------------------------------------------------------------------------
def infectious_rate_tweets_vec(t, r0=0.424, phi0=0.125, taum=2., tm=24., t0=0,p0=0.01,S=3):
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
#     print(r0,phi0,taum)
    try:
        if r0>1 or r0 < -1 :
            return 10**10*np.ones_like(t)
        if taum>20 or taum < 0.5 :
            return 10**10*np.ones_like(t)
        else:
            R= p0*(1-(S*r0)*np.sin(2*np.pi/tm*(t+phi0)))*np.exp(-(t-t0)/taum)
            R[t<t0]=0
            return R
    except:
        msg_error="No se completo operación en la funcion  infectious_rate_tweets_vec"
        logging.error(msg_error)
        raise Exception(msg_error)
    

def loss_function(params, estimates, fun,s_vec):
    """
    Loss function used by least squares.

    :param params: current values of parameters to fit
    :param estimates: estimates of function to fit
    :param fun: function to fit
    :return: array of loss values for every estimation
    """
    try:
        diffs=[]
        for i in estimates:
            est=estimates[i][0]
            xval=estimates[i][1] 
            t0=xval[0]
            diffs.append(abs(fun(xval,t0=t0,p0=est[0],S=s_vec[i],*params)-est))
        return np.concatenate(diffs)
    except:
        msg_error="No se completo operación en la funcion  loss_function"
        logging.error(msg_error)
        raise Exception(msg_error)


def fit_parameter(estimates, fun, s_vec,start_values=None):
    """
    Fitting any numbers of given infectious rate function using least squares.
    Used count of observed events in observation window as weights.

    :param estimates: estimated values of function
    :param fun: function to fit
    :param start_values: initial guesses of parameters, should be a ndarray
    :return: fitted parameters
    """
    from scipy.optimize import leastsq
    try:
        if start_values is None:
            start_values = np.array([0, 0, 1.])

        return leastsq(func=loss_function, x0=start_values, args=(estimates, fun,s_vec))[0]
    except:
        msg_error="No se completo operación en la funcion  fit_parameter"
        logging.error(msg_error)
        raise Exception(msg_error)

def error_infectious_rate(estimated,param_fitted,fun,s_vec):
    '''
    WMAPE  --- MAE
    '''
    """
    Calculo error entre los valores estimados y los resultantes usando la formula ajustada de la tasa de infeccion

    :param estimates: estimated values of function
    :param param_fitted: fitted parameters of fuction
    :param fun: infectious rate function
    :param p0_vec: p0 vector of each tweet
    :return: WMAPE calculus
    """
    try:
        total_estimated=[]
        total_fitted=[]
        for i in estimated:
            total_estimated.append(estimated[i][0])
            xval=estimated[i][1] 
            total_fitted.append(fun(xval,t0=xval[0],p0=estimated[i][0][0],S=s_vec[i],*param_fitted))
        total_estimated=np.concatenate(total_estimated)
        total_fitted=np.concatenate(total_fitted)

        return abs(total_estimated-total_fitted).sum()/sum(abs(total_estimated))
    except Exception as e:
        msg_error="No se completo operación en la funcion  error_infectious_rate"
        logging.error(msg_error)
        raise Exception(msg_error + " / " +str(e))

def get_infectious_rate_fitted(time_observed,
                               keys_tweets,
                               Tweets,
                               kernel_integral,
                               fun_infectious,
                               followers_rate=3,
                               win_size=4
                              ):
    """
    Calculo tasa de infeccion por la formula y los parametros ajustados

    :return: lista parametros ajustados, funcion resultante , error de ajuste
    """    
    try:
        p_est=compute_p_est(time_observed,keys_tweets,Tweets,kernel_integral,followers_rate,win_size)
        s_vec={}
        for i in p_est:
            s_vec[i]=(5.0-Tweets[i]['sentiment'])/5.0
            p0=p_est[i][0][0]
        param_fit=fit_parameter(p_est,fun_infectious,s_vec)
        error_infectious=error_infectious_rate(p_est,param_fit,fun_infectious,s_vec)
        fun_fit= lambda t,t0,p0,s : fun_infectious(t,t0=t0,p0=p0,S=s,*param_fit)
        return p_est,param_fit,fun_fit,error_infectious
    except Exception as e:
        msg_error="No se completo operación en la funcion  get_infectious_rate_fitted"
        logging.error(msg_error)
        raise Exception(msg_error + " / " +str(e))
#----------------------------------------------------------------------------------------
def input_replicas(t,keys_tweets,Tweets,kernel,fun,followers_rate,estimated):
    """
    Calculo aporte de los valores de las replicas en la funcion lambda

    :return: lista valores replicas para cada instante t
    """    
    try:
        replicas=np.zeros_like(t)
        for i in keys_tweets:
            tweet=Tweets[i]
            event_times=tweet['times']
            t0=event_times[0]
            S=(5.0-tweet['sentiment'])/5.0
            followers=sigmoid_foll(tweet['followers'],followers_rate)
            sum_int=(followers*kernel(t.reshape((len(t),1))-event_times)).sum(axis=1)
            p_t=fun(t,t0=t0,p0=estimated[i][0][0],s=S)
            sum_ext=np.nan_to_num(sum_int*p_t)
            replicas+=sum_ext
        return replicas
    except Exception as e:
        msg_error="No se completo operación en la funcion  input_replicas"
        logging.error(msg_error)
        raise Exception(msg_error + " / " +str(e))

def lambda_t(t,beta,f_cov,keys_tweets,Tweets,kernel,fun,followers_rate,estimated):
    """
    Calculo valores de la funcion intensidad dado sus parametros y los instantes de tiempo a calcular

    :return: lista valores funcion intensidad
    """    
    try:
        back_g=back_ground(beta,t,f_cov)
        replicas=input_replicas(t,keys_tweets,Tweets,kernel,fun,followers_rate,estimated)
        return back_g+replicas
    except Exception as e:
        msg_error="No se completo operación en la funcion  lambda_t"
        logging.error(msg_error)
        raise Exception(msg_error + " / " +str(e))


def get_tweets_from_lambda(Lt,t,f_inicio=None):
    """
    Calculo cantidad tweets a partir de la intensidad

    :return: tabla tiempos iniciales y finales de los intervalos y cantidad de tweets en cada uno
    """    
    try:
        dt=(t[-1]-t[0])/len(t)
        df=pd.DataFrame({'start':t[:-1],'end':t[1:],'Tweets':(Lt*dt)[:-1]}) 
        if f_inicio!=None:
            restore = lambda a: restore_date(a,f_inicio)
            df["start"]=df["start"].apply(restore)
            df["end"]=df["end"].apply(restore)

        return df


    except Exception as e:
        msg_error="No se completo operación en la funcion  get_tweets_from_lambda"
        logging.error(msg_error)
        raise Exception(msg_error + " / " +str(e))

def interpolate_T(T,pre_lamda,t):
    """
    interpolacion para calcular valores de lamda que no estan en la lista de tiempo predefinida

    :return: valor de lamda hallado por la interpolacion
    """    
    try:
        index_low=np.where((t <= T) == True)[0][0]
        index_up=np.where((t >= T) == True)[0][0]
        if index_low == index_up:
            lamda=pre_lamda[index_low]
        else:
            lamda=np.interp(T,[t[index_low],t[index_up]],
                      [pre_lamda[index_low],pre_lamda[index_up]])
        return lamda
    except:
        msg_error="No se completo operación en la funcion  interpolate_T"
        logging.error(msg_error)
        raise Exception(msg_error)

def thinning_pred(pre_lamda,t,return_samples=False,f_inicio=None):
    """
    metodo de prediccion basado en el calculo de probabilidad de la existencia de eventos

    :return: tabla tiempos iniciales y finales de los intervalos y cantidad de tweets predichos en cada uno
    """    
    try:
        samples=[]
        T=t[0]
        while T <= t[-1]:
            lamda=interpolate_T(T,pre_lamda,t)
            mu = np.random.uniform()
            Tau = -np.log(mu)/lamda
            T=T+Tau
            if T >= t[-1]:
                break
            s = np.random.uniform()
            if s <= interpolate_T(T,pre_lamda,t)/lamda:
                samples.append(T)

        if return_samples == True :
            return samples
        else:    
            CT = count_tweets(samples,t)
            df = pd.DataFrame({'start':t[:-1],'end':t[1:],'Tweets':CT})  
            if f_inicio!=None:
                restore = lambda a: restore_date(a,f_inicio)
                df["start"]=df["start"].apply(restore)
                df["end"]=df["end"].apply(restore)
            return df
    except:
        msg_error="No se completo operación en la funcion  thinning_pred"
        logging.error(msg_error)
        raise Exception(msg_error)

def count_tweets(times,intervals):
    """
    cuenta los tweets en intervalos dados
    :return: lista tweets
    """    
    try:
        tweets = []
        for i in range(len(intervals)-1):
            tweets+=[get_event_count(times, intervals[i], intervals[i+1])]
        return tweets
    except Exception as e:
        msg_error="No se completo operación en la funcion  count_tweets"
        logging.error(msg_error)
        raise Exception(msg_error + " / " +str(e))
        
def real_tweets(keys_tweets,Tweets,t):
    """
    cuenta los tweets reales que aparecen en los datos para un intervalo de tiempo dado
    :return: tabla tiempos iniciales y finales de los intervalos y cantidad de tweets encontrados en cada uno
    """    
    try:
        Total_et=[]
        for i in keys_tweets:
            tweet=Tweets[i]
            event_times=tweet['times']
            Total_et+=list(event_times)
        Total_et=np.array(Total_et)
        real_values=count_tweets(Total_et,t)
        return pd.DataFrame({'start':t[:-1],'end':t[1:],'Tweets':real_values})  
    except Exception as e:
        msg_error="No se completo operación en la funcion  real_tweets"
        logging.error(msg_error)
        raise Exception(msg_error + " / " +str(e))


def lambda_pred(t_pred,keys_tweets,Tweets,beta,f_covariados,kernel,kernel_integral,fun,followers_rate,time_observed,estimated):
    """
    Calculo prediccion intensidad tweets en un intervalo dado
    :return: lista de valores de la intensidad predicha para los valores temporales del intervalo dado
    """    
    try:
        back_g=back_ground(beta,t_pred,f_covariados)
        BG_samples=thinning_pred(back_g,t_pred,return_samples=True)

        p0_mean=np.mean([estimated[i][0][0] for i in estimated])

        f_mean=[]
        for i in keys_tweets:
            f_mean+=[sigmoid_foll(Tweets[i]['followers'],followers_rate).mean()]
        hist,bins=np.histogram(np.array(f_mean),density=False,bins=10) 
        hist=hist/hist.sum()
        bins=bins[1:]/225+5/9


        sentiments, percent_sen = np.unique([Tweets[i]['sentiment'] for i in keys_tweets],return_counts=True)
        sentiments = (5-sentiments)/5
        percent_sen = percent_sen/percent_sen.sum()

        sum_= []
        for t0 in BG_samples:
            p0=p0_mean
            s=np.random.choice(sentiments,p=percent_sen)
            d=np.random.choice(bins,p=hist)
            sum_.append(d*fun(t_pred,t0,p0,s))
        sum_ = np.array(sum_).sum(axis=0)

        int_=kernel_integral(0, t_pred-time_observed)

        LT=lambda_t(t_pred,beta,f_covariados,keys_tweets,Tweets,kernel,fun,followers_rate,estimated)
        LP=LT+sum_*int_


    #     replicas=input_replicas(t_pred,keys_tweets,Tweets,kernel,fun,followers_rate)
    #     numerador=back_g+replicas
    #     #denominador
    #     sum_int=np.zeros_like(t_pred)
    #     for i in keys_tweets:
    #         tweet=Tweets[i]
    #         event_times=tweet['times']
    #         t0=event_times[0]
    #         S=(5.0-tweet['sentiment'])/5.0
    #         followers=sigmoid_foll(tweet['followers'],followers_rate)
    #         p_t=fun(t_pred,t0=t0,p0=S)
    #         sum_int+=p_t*followers[event_times<=time_observed].mean()/(followers_rate*2)
    #     denomin=1-kernel_integral(0, t_pred-time_observed)*sum_int
    #     return numerador/denomin

        return LP
    except Exception as e:
        msg_error="No se completo operación en la funcion  lambda_pred"
        logging.error(msg_error)
        raise Exception(msg_error + " / " +str(e))

def tweets_for_interval(interval,Tweets,t_start,t_end):
    try:
        t1, t2 = interval
        total=Tweets[(Tweets.start >= t1) & (Tweets.end <= t2)].Tweets.sum()
        if (t1 >= t_start) & (t2 <= t_end):
            return total
        else:
            print("El intervalo de tiempo dado no esta contenido en los Tweets")
            return total
    except:
        msg_error="No se completo operación en la funcion  tweets_for_interval"
        logging.error(msg_error)
        raise Exception(msg_error)
    

#----------------------------------------------------------------------------------------
class modelTweets:
    """
    Clase del modelo de tweets que contiene todas las funciones anteriormente definidas
    de tal forma que se definen los parametros globales al crear una instancia y las funciones
    son llamadas sin necesidad de poner explicitamente los parametros adicionales.
    """    
    def __init__(self, data,
                 train_period,
                 kernel=lambda x: kernel_zhao_vec(x,s0=300/3600,theta=0.242),
                 kernel_primitive=lambda x: kernel_primitive_zhao_vec(x,s0=300/3600,theta=0.242),
                 kernel_integral=lambda x1,x2: integral_zhao(x1,x2,s0=300/3600,theta=0.242),
                 f_covariates=(T_c,restore_date),
                 win_size_for_partition_cov=1,
                 followers_rate=2,
                 infectious_rate_base = infectious_rate_tweets_vec,
                 win_size_infectious_rate = 3,
                 win_size_train_period = 1,
                 win_size_pred_period = 1,
                 method_pred = 'integral',
                ):
        logging.debug('Empezo inicialización modelo de Tweets.')
        try:
            self.f_inicio = data['Inicio']
            self.Tweets = data['Tweets']
            self.train_period = train_period
            self.train_start = date_to_hours(train_period[0],self.f_inicio)
            self.train_end = date_to_hours(train_period[1],self.f_inicio)   
            
            self.kernel = kernel
            self.kernel_primitive = kernel_primitive
            self.kernel_integral = kernel_integral
            self.f_covariates = lambda a : f_covariates[0](f_covariates[1](a,self.f_inicio))
            self.win_size_for_partition_cov=win_size_for_partition_cov
            self.followers_rate = followers_rate
            self.infectious_rate_base = infectious_rate_base
            self.win_size_infectious_rate  = win_size_infectious_rate
            self.win_size_train_period  = win_size_train_period
            self.win_size_pred_period  = win_size_pred_period
            self.method_pred  = method_pred
            
            keys_train=[]
            for i in self.Tweets:
                times=self.Tweets[i]['times']
                if sum((times >= self.train_start) & (times < self.train_end)) > 0:
                    keys_train.append(i)
            self.keys_train = keys_train
            if len(keys_train)==0:
                msg_error="No se encontraron datos en el periodo de entrenamiento establecido."
                logging.error(msg_error)
                raise Exception(msg_error)
            logging.debug('Lista de tweets de entrenamiento creado.')   
            
            keys_train_in=[]
            for i in keys_train:
                t0=self.Tweets[i]['times'][0]
                if (t0 >= self.train_start) &  (t0 < self.train_end):
                    keys_train_in.append(i)
            self.keys_train_in = keys_train_in
            self.t_train=np.linspace(self.train_start,self.train_end,num=int((self.train_end-self.train_start)/self.win_size_train_period))
            self.real_tweets_train=real_tweets(self.keys_train,self.Tweets,self.t_train)

            
        except Exception as e:
            msg_error = "No se pudo completar con la inicialización del modelo de tweets."
            logging.error(msg_error)
            raise Exception(msg_error + " / " +str(e))
        
    def compute_Beta(self,beta_0=np.array([1,1,1])):
        logging.debug('Empezo operación calculo Beta.')
        try:
            beta_0=np.ones_like(self.f_covariates(0))
            self.Beta,self.error_Beta=Beta(self.train_start,self.train_end,
                                           self.keys_train_in,self.Tweets,
                                           self.f_covariates,self.win_size_for_partition_cov,
                                           beta_0
                                           )
            logging.debug('Termino operación calculo Beta.')
            return self.Beta
        except Exception as e:
            msg_error = "No se completo operación en la funcion  compute_Beta en la clase."
            logging.error(msg_error)
            raise Exception(msg_error + " / " +str(e))
            

    def compute_p_est(self):
        logging.debug('Empezo operación calculo estimación de intensidades.')
        try:
            self.p_est = compute_p_est(self.train_end,
                                       self.keys_train,
                                       self.Tweets,
                                       self.kernel_integral,
                                       self.followers_rate)
            logging.debug('Termino operación calculo estimación de intensidades.')
            return self.p_est
        except Exception as e:
            msg_error = "No se completo operación en la funcion  compute_p_est en la clase."
            logging.error(msg_error)
            raise Exception(msg_error + " / " +str(e))
    
    def infectious_rate_fit(self):
        logging.debug('Empezo operación ajuste de intensidades.')
        try:
            p_est,param_fit,fun_fit,error_infectious = get_infectious_rate_fitted(self.train_end,
                                                                                  self.keys_train,
                                                                                  self.Tweets,
                                                                                  self.kernel_integral,
                                                                                  self.infectious_rate_base,
                                                                                  self.followers_rate,
                                                                                  self.win_size_infectious_rate  
                                                                                 )
            self.p_est = p_est
            self.param_infectious_fit=param_fit
            self.infectious_rate=fun_fit
            self.error_infectious=error_infectious
            logging.debug('Termino operación ajuste de intensidades.')
            return self.param_infectious_fit,self.infectious_rate,self.error_infectious
        except Exception as e:
            msg_error = "No se completo operación en la funcion  infectious_rate_fit en la clase."
            logging.error(msg_error)
            raise Exception(msg_error + " / " +str(e))
    
    def compute_lamda_train_point(self,t):
        try:
            return lambda_t(np.array([t]),
                            self.Beta,
                            self.f_covariates,
                            self.keys_train,
                            self.Tweets,
                            self.kernel,
                            self.infectious_rate,
                            self.followers_rate,
                            self.p_est)[0]
        except Exception as e:
            msg_error = "No se completo operación en la funcion  compute_lamda_train_point en la clase."
            logging.error(msg_error)
            raise Exception(msg_error + " / " +str(e))
    
    def compute_lambda_train(self):
        logging.debug('Empezo calculo lamda en el periodo de entrenamiento.')
        try:
            self.lambda_train = lambda_t(self.t_train,
                                        self.Beta,
                                        self.f_covariates,
                                        self.keys_train,
                                        self.Tweets,
                                        self.kernel,
                                        self.infectious_rate,
                                        self.followers_rate,
                                        self.p_est)
            if self.method_pred == 'integral':
                self.Tweets_est_train=get_tweets_from_lambda(self.lambda_train, self.t_train,self.f_inicio)
            elif self.method_pred == 'thinning':
                self.Tweets_est_train=thinning_pred(self.lambda_train,self.t_train)
            else:
                raise Exception("Metodo de predicción invalido")
            logging.debug('Termino calculo lamda en el periodo de entrenamiento.')
            return self.lambda_train, self.Tweets_est_train
        except Exception as e:
            msg_error = "No se completo operación en la funcion  compute_lambda_train en la clase."
            logging.error(msg_error)
            raise Exception(msg_error + " / " +str(e))
    
    def compute_lamda_predict_point(self,t):
        try:
            return lambda_pred(np.array([t]),
                               self.keys_train,
                               self.Tweets,
                               self.Beta,
                               self.f_covariates,
                               self.kernel,
                               self.kernel_integral,
                               self.infectious_rate,
                               self.followers_rate,
                               self.train_end,self.p_est)[0]
        except:
            print("Error calculo lambda_predict puntual en la clase")

    def compute_lambda_predict(self,dates):
        logging.debug('Empezo calculo lamda en el periodo de predicción.')
        try:
            start = date_to_hours(dates[0],self.f_inicio)
            end = date_to_hours(dates[1],self.f_inicio)   
            t_pred=np.linspace(start,end,num=int((end-start)/self.win_size_pred_period))

            self.lambda_predict = lambda_pred(t_pred,
                                              self.keys_train,
                                              self.Tweets,
                                              self.Beta,
                                              self.f_covariates,
                                              self.kernel,
                                              self.kernel_integral,
                                              self.infectious_rate,
                                              self.followers_rate,
                                              self.train_end,
                                              self.p_est)

            if self.method_pred == 'integral':
                self.Tweets_pred=get_tweets_from_lambda(self.lambda_predict, t_pred,self.f_inicio)
            elif self.method_pred == 'thinning':
                self.Tweets_pred=thinning_pred(self.lambda_predict,t_pred)
            else:
                raise Exception("Metodo de predicción invalido")

            logging.debug('Termino calculo lamda en el periodo de predicción.')
            return self.lambda_predict,self.Tweets_pred
        except Exception as e:
            msg_error = "No se completo operación en la funcion  compute_lambda_predict en la clase."
            logging.error(msg_error)
            raise Exception(msg_error + " / " +str(e))


    def poisson_method(self,dates):
        logging.debug('Empezo proceso de predicción metodo de poisson.')
        import statsmodels.api as sm
        try:            
            X=np.array([self.f_covariates(i) for i in self.real_tweets_train.start])
            y=self.real_tweets_train.Tweets
            poisson_training_results = sm.GLM(y, X, family=sm.families.Poisson()).fit()
            X_pred=np.array([self.f_covariates(i) for i in self.real_tweets_validate.start])
            poisson_predictions = poisson_training_results.get_prediction(X_pred).summary_frame()['mean'].values

            start = date_to_hours(dates[0],self.f_inicio)
            end = date_to_hours(dates[1],self.f_inicio)   
            t_pred=np.linspace(start,end,num=int((end-start)/self.win_size_pred_period))

            dt=(t_pred[-1]-t_pred[0])/len(t_pred)
            self.poisson_predictions=pd.DataFrame({'start':t_pred[:-1],'end':t_pred[1:],'Tweets':poisson_predictions}) 
            logging.debug('Termino proceso de predicción metodo de poisson.')
            return self.poisson_predictions
        except:
            logging.error("No termino proceso de predicción metodo de poisson.")

    

    def linear_reg_method(self,dates):
        logging.debug('Empezo proceso de predicción metodo de regresión lineal.')
        try:
            def to_mini(alpha):
                return np.sum(times-alpha,axis=0)

            from scipy.optimize import leastsq

            start = date_to_hours(dates[0],self.f_inicio)
            end = date_to_hours(dates[1],self.f_inicio)   
            t_pred=np.linspace(start,end,num=int((end-start)/self.win_size_pred_period))

            back_g=back_ground(self.Beta,t_pred,self.f_covariates)
            BG_samples=thinning_pred(back_g,t_pred,return_samples=True)
            BG_samples=np.array(count_tweets(BG_samples,t_pred))

            values=[]
            dist_RT=[]
            max_=0
            for i in self.keys_train_in:
                CT=count_tweets(self.Tweets[i]['times'],self.t_train)
                CT=np.cumsum(np.trim_zeros(CT))
                dist_RT.append(CT[0])    
                CT=CT/CT[-1]
                if len(CT)>max_:
                    max_=len(CT)
                values.append(np.log(CT))

            times=[]
            for i in values:
                l=np.zeros(max_)
                l[:len(i)]=i
                times.append(l)
            times=np.array(times)
            Root = leastsq(to_mini, x0=np.ones(max_))[0]
            S2=Root/len(times)
            exp_=np.exp(Root+S2/2)
            hist,bins=np.histogram(dist_RT,density=False,bins=30) 
            hist=hist/hist.sum()
            BG_cum=np.zeros_like(BG_samples)
            for i,j in enumerate(BG_samples):
                RT_dist=np.random.choice(bins[1:],size=j,p=hist)
                A=(RT_dist.reshape((len(RT_dist),1))*exp_).astype(int)
                A=(A[:,1:]-A[:,:-1]).sum(axis=0)
                lim_sup=i+len(A)
                if  lim_sup >= len(BG_samples):
                    lim_sup=len(BG_samples)
                BG_cum[i:lim_sup]+=A[:lim_sup-i]
            dt=(t_pred[-1]-t_pred[0])/len(t_pred)
            self.linear_predictions=pd.DataFrame({'start':t_pred[:-1],'end':t_pred[1:],'Tweets':BG_cum+BG_samples}) 
            logging.debug('Termino proceso de predicción metodo de regresión lineal.')
            return self.linear_predictions
        except:
            logging.error('No termino proceso de predicción metodo de regresión lineal.')
    
    def compute_errors(self,real,predict):
        logging.debug('Empezo proceso de calculo errores en predicción.')
        try:
            real=real.Tweets.values
            predict=predict.Tweets.values
            diff=abs(real-predict)

            self.errors_predict={}
            ##
            self.errors_predict['APE']=diff/real
            self.errors_predict['MAPE']=self.errors_predict['APE'].mean()
            self.errors_predict['MSE']=(diff**2).mean()
            self.errors_predict['MAE']=diff.mean()
            self.errors_predict['RMSE']=np.sqrt((diff**2).mean())
            self.errors_predict['Pearson']=abs(stats.pearsonr(real,predict)[0])
            self.errors_predict['kendall']=abs(stats.kendalltau(real,predict)[0])

            real=np.cumsum(real)
            predict=np.cumsum(predict)
            diff=abs(real-predict)

            self.errors_predict_cum={}
            ##
            self.errors_predict_cum['APE']=diff/real
            self.errors_predict_cum['MAPE']=self.errors_predict_cum['APE'].mean()
            self.errors_predict_cum['MSE']=(diff**2).mean()
            self.errors_predict_cum['MAE']=diff.mean()
            self.errors_predict_cum['RMSE']=np.sqrt((diff**2).mean())
            self.errors_predict_cum['Pearson']=abs(stats.pearsonr(real,predict)[0])        
            self.errors_predict_cum['kendall']=abs(stats.kendalltau(real,predict)[0])
            logging.debug('Termino proceso de calculo errores en predicción.')
            return self.errors_predict,self.errors_predict_cum
        except:
            logging.error("No termino proceso de calculo errores en predicción.")
        
        
        
    def train(self,beta_0=np.array([1,1,1])):
        logging.debug('Empezo proceso de entrenamiento modelo tweets.')
        try:
            self.compute_Beta(beta_0)
            self.infectious_rate_fit()
            self.compute_lambda_train()
            logging.debug('Termino proceso de entrenamiento modelo tweets.')
            return self.Beta, self.error_Beta,self.param_infectious_fit, self.error_infectious, self.lambda_train
        except Exception as e:
            msg_error = "No se completo operación en la funcion  train en la clase."
            logging.error(msg_error)
            raise Exception(msg_error + " / " +str(e))

    def validate(self,valid_period):
        logging.debug('Empezo proceso de validación modelo tweets.')
        try:
            validate_start = date_to_hours(valid_period[0],self.f_inicio)
            validate_end = date_to_hours(valid_period[1],self.f_inicio)
            keys_validation=[]
            for i in self.Tweets:
                times=self.Tweets[i]['times']
                if sum((times >= validate_start) & (times < validate_end)) > 0:
                    keys_validation.append(i)
            if len(keys_validation)==0:
                msg_error="No se encontraron datos en el periodo de validacion establecido"
                logging.debug(msg_error)
                raise Exception(msg_error)
            logging.debug('Lista de tweets de validación creado.')
            t_pred=np.linspace(validate_start,validate_end,num=int((validate_end-validate_start)/self.win_size_pred_period))
            real_tweets_validate=real_tweets(keys_validation,self.Tweets,t_pred)
            predict_tweets=self.compute_lambda_predict(valid_period)[1]
            errors,errors_cum=self.compute_errors(real_tweets_validate,predict_tweets)
            logging.debug('Termino proceso de validación modelo tweets.')
            return errors,errors_cum

        except Exception as e:
            msg_error = "No se completo operación en la funcion  validate en la clase."
            logging.error(msg_error)
            raise Exception(msg_error + " / " +str(e))

    def save_model(self,path):
        """
        Funcion para guardar el objeto de la clase como archivo pkl
        """
        try:
            with open(path, 'wb') as outp:
                pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        except:
            msg_error="No se pudo guardar el modelo con la direccion establecida"
            logging.error(msg_error)
            raise Exception(msg_error)
        
#----------------------------------------------------------------------------------------