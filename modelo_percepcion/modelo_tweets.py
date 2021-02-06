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

#----------------------------------------------------------------------------------------

def T_c(t):
    return np.array([t.weekday()/6.0,(t.hour > 12)*1,1])


def date_to_hours(date,f_inicio):
    date = datetime.fromisoformat(date)
    hours = (date-f_inicio).total_seconds()/3600
    return hours

def restore_date(t,f_inicio):
    return f_inicio+timedelta(hours=t)

def compare_vectors(a,b):
    if np.linalg.norm(abs(a-b)) == 0:
        return True
    else:
        return False

def get_particion(inicio,fin,f_covariados,win_size=1):

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


#----------------------------------------------------------------------------------------
## Calculo de Beta

def beta_left_hand(keys_train,Tweets,f_covariados):
    left_hand=np.zeros_like(f_covariados(0))
    for i in keys_train:
        t_0=Tweets[i]['times'][0]        
        left_hand+=f_covariados(t_0)
    return left_hand


def to_solve(beta,particion,left_hand,f_covariados):
    l_particion,v_particion=particion
    right_hand=(l_particion.reshape((len(l_particion),1))*np.array([f_covariados(t)*np.exp(beta*f_covariados(t)) for t in v_particion])).sum(axis=0)
    return right_hand-left_hand

    
def get_beta(beta_0,particion,left_hand,f_covariados):
    return minimize(lambda x: np.linalg.norm(to_solve(x,particion,left_hand,f_covariados)),
                    x0=beta_0,
                    method='Nelder-Mead').x

def Beta(inicio,time_observed,keys_start_in,Tweets,f_covariados,win_size=1,beta_0=np.array([1,1,1])):
    left_hand=beta_left_hand(keys_start_in,Tweets,f_covariados)
    particion=get_particion(inicio,time_observed,f_covariados,win_size=win_size)
    return get_beta(beta_0,particion,left_hand,f_covariados)

def back_ground(beta,t,f_covariados):
    return [np.exp(np.dot(beta,f_covariados(i))) for i in t]
    
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
    if count_events is not None:
        return count_events / kernel_int.sum()
    else:
        return event_times.size / kernel_int.sum()
    

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
    t0=event_times[0]
    num_=int((obs_time-t0)/win_size)
    time_x=np.linspace(t0, obs_time,num=num_)
    if len(time_x) < 2:
        time_x=np.array([t0,obs_time])
    
    
    
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

def sigmoid_foll(foll,exp=2):
    return 2/(1+np.exp(-10**(-exp)*foll))-1

def compute_p_est(time_observed,keys_tweets,Tweets,kernel_integral=integral_zhao,followers_rate=2,win_size=4):
    p_est={}
    for i in keys_tweets:
        tweet=Tweets[i]
        event_times=tweet['times']
        t0=event_times[0]
        S=tweet['sentiment']
        followers=sigmoid_foll(tweet['followers'],followers_rate)
        mask= event_times <= time_observed
        p_i_est,t_points=estimate_infectious_rate_vec(event_times[mask], followers[mask], integral_zhao, time_observed,win_size)
        p_est[i]=[p_i_est,t_points]
    return p_est
#----------------------------------------------------------------------------------------
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
#     print(r0,phi0,taum)
    if r0>1 or r0 < -1 :
        return 10**10*np.ones_like(t)
    if taum>20 or taum < 0.5 :
        return 10**10*np.ones_like(t)
    else:
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

def error_infectious_rate(estimated,param_fitted,fun,p0_vec):
    '''
    WMAPE  --- MAE
    '''
    total_estimated=[]
    total_fitted=[]
    for i in estimated:
        total_estimated.append(estimated[i][0])
        xval=estimated[i][1] 
        total_fitted.append(fun(xval,t0=xval[0],p0=p0_vec[i],*param_fitted))
    total_estimated=np.concatenate(total_estimated)
    total_fitted=np.concatenate(total_fitted)
    
    return abs(total_estimated-total_fitted).sum()/sum(abs(total_estimated))

def get_infectious_rate_fitted(time_observed,
                               keys_tweets,
                               Tweets,
                               kernel_integral,
                               fun_infectious,
                               followers_rate=2,
                               win_size=4
                              ):
    p_est=compute_p_est(time_observed,keys_tweets,Tweets,kernel_integral,followers_rate,win_size)
    p0_vec={}
    for i in p_est:
        p0_vec[i]=(5.0-Tweets[i]['sentiment'])/5.0
    param_fit=fit_parameter(p_est,fun_infectious,p0_vec)
    error_infectious=error_infectious_rate(p_est,param_fit,fun_infectious,p0_vec)
    fun_fit= lambda t,t0,p0 : fun_infectious(t,t0=t0,p0=p0,*param_fit)
    return param_fit,fun_fit,error_infectious
#----------------------------------------------------------------------------------------
def input_replicas(t,keys_tweets,Tweets,kernel,fun,followers_rate):
    replicas=np.zeros_like(t)
    for i in keys_tweets:
        tweet=Tweets[i]
        event_times=tweet['times']
        t0=event_times[0]
        S=(5.0-tweet['sentiment'])/5.0
        followers=sigmoid_foll(tweet['followers'],followers_rate)

        sum_int=followers*kernel(t.reshape((len(t),1))-event_times)
        sum_int=sum_int.sum(axis=1)    
        p_t=fun(t,t0=t0,p0=S)
        sum_ext=np.nan_to_num(sum_int*p_t)
#         if sum(sum_ext) > 1000:
#             print(i)
        replicas+=sum_ext
    return replicas

def lambda_t(t,beta,f_cov,keys_tweets,Tweets,kernel,fun,followers_rate):
    back_g=back_ground(beta,t,f_cov)
    replicas=input_replicas(t,keys_tweets,Tweets,kernel,fun,followers_rate)
    return back_g+replicas


def get_tweets_from_lambda(Lt,t):
    dt=(t[-1]-t[0])/len(t)
    return pd.DataFrame({'start':t[:-1],'end':t[1:],'Tweets':(Lt*dt)[:-1]}) 

def interpolate_T(T,pre_lamda,t):
    index_low=np.where((t <= T) == True)[0][0]
    index_up=np.where((t >= T) == True)[0][0]
    if index_low == index_up:
        lamda=pre_lamda[index_low]
    else:
        lamda=np.interp(T,[t[index_low],t[index_up]],
                  [pre_lamda[index_low],pre_lamda[index_up]])
    return lamda

def thinning_pred(pre_lamda,t):
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
            
    CT = count_tweets(samples,t)
    return pd.DataFrame({'start':t[:-1],'end':t[1:],'Tweets':CT})  

def count_tweets(times,intervals):
    tweets = []
    for i in range(len(intervals)-1):
        tweets+=[get_event_count(times, intervals[i], intervals[i+1])]
    return tweets
        
def real_tweets(keys_tweets,Tweets,t):
    Total_et=[]
    for i in keys_tweets:
        tweet=Tweets[i]
        event_times=tweet['times']
        Total_et+=list(event_times)
    Total_et=np.array(Total_et)
    real_values=count_tweets(Total_et,t)
    return pd.DataFrame({'start':t[:-1],'end':t[1:],'Tweets':real_values})  


def lambda_pred(t_pred,keys_tweets,Tweets,beta,
                f_covariados,kernel,kernel_integral,fun,followers_rate,time_observed):
    back_g=back_ground(beta,t_pred,f_covariados)
    replicas=input_replicas(t_pred,keys_tweets,Tweets,kernel,fun,followers_rate)
    numerador=back_g+replicas
    #denominador
    sum_int=np.zeros_like(t_pred)
    for i in keys_tweets:
        tweet=Tweets[i]
        event_times=tweet['times']
        t0=event_times[0]
        S=(5.0-tweet['sentiment'])/5.0
        followers=sigmoid_foll(tweet['followers'],followers_rate)
        p_t=fun(t_pred,t0=t0,p0=S)
        sum_int+=p_t*followers[event_times<=time_observed].mean()/(followers_rate*2)
    denomin=1-kernel_integral(0, t_pred-time_observed)*sum_int
    return numerador/denomin

def tweets_for_interval(interval,Tweets,t_start,t_end):
    t1, t2 = interval
    total=Tweets[(Tweets.start >= t1) & (Tweets.end <= t2)].Tweets.sum()
    if (t1 >= t_start) & (t2 <= t_end):
        return total
    else:
        print("El intervalo de tiempo dado no esta contenido en los Tweets")
        return total
    

#----------------------------------------------------------------------------------------
class modelTweets:
    def __init__(self, data,
                 train_period,
                 val_period,
                 kernel=lambda x: kernel_zhao_vec(x,s0=300/3600,theta=0.242),
                 kernel_primitive=lambda x: kernel_primitive_zhao_vec(x,s0=300/3600,theta=0.242),
                 kernel_integral=lambda x1,x2: integral_zhao(x1,x2,s0=300/3600,theta=0.242),
                 f_covariates=(T_c,restore_date),
                 win_size_for_partition_cov=1,
                 followers_rate=2,
                 infectious_rate_base = infectious_rate_tweets_vec,
                 win_size_infectious_rate = 4,
                 win_size_train_period = 1,
                 win_size_pred_period = 1,
                 method_pred = 'integral',
                ):
        
        self.data = data
        self.f_inicio = data['Inicio']
        self.Tweets = data['Tweets']
        self.train_period = train_period
        self.train_start = date_to_hours(train_period[0],self.f_inicio)
        self.train_end = date_to_hours(train_period[1],self.f_inicio)   
        self.validate_period = val_period
        self.validate_start = date_to_hours(val_period[0],self.f_inicio)
        self.validate_end = date_to_hours(val_period[1],self.f_inicio)
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
        
        keys_train_in=[]
        for i in keys_train:
            t0=self.Tweets[i]['times'][0]
            if (t0 >= self.train_start) &  (t0 < self.train_end):
                keys_train_in.append(i)
        self.keys_train_in = keys_train_in
        
        
        keys_validation=[]
        for i in self.Tweets:
            times=self.Tweets[i]['times']
            if sum((times >= self.validate_start) & (times < self.validate_end)) > 0:
                keys_validation.append(i)
        self.keys_validation = keys_validation
        
        self.t_train=np.linspace(self.train_start,self.train_end,num=int((self.train_end-self.train_start)/self.win_size_train_period))
        self.t_pred=np.linspace(self.validate_start,self.validate_end,num=int((self.validate_end-self.validate_start)/self.win_size_pred_period))
        
        self.real_tweets_train=real_tweets(self.keys_train,self.Tweets,self.t_train)
        self.real_tweets_validate=real_tweets(self.keys_validation,self.Tweets,self.t_pred)
        
    def compute_Beta(self,beta_0=np.array([1,1,1])):
        beta_0=np.zeros_like(self.f_covariates(0))
        self.Beta=Beta(self.train_start,self.train_end,
                        self.keys_train_in,self.Tweets,
                        self.f_covariates,self.win_size_for_partition_cov,
                        beta_0
                        )
        return self.Beta

    def compute_p_est(self):
        self.p_est = compute_p_est(self.train_end,
                                   self.keys_train,
                                   self.Tweets,
                                   self.kernel_integral,
                                   self.followers_rate)
        return self.p_est
    
    def infectious_rate_fit(self):
        param_fit,fun_fit,error_infectious = get_infectious_rate_fitted(self.train_end,
                                                                        self.keys_train,
                                                                        self.Tweets,
                                                                        self.kernel_integral,
                                                                        self.infectious_rate_base,
                                                                        self.followers_rate,
                                                                        self.win_size_infectious_rate                                                               
                                                                        )
        self.param_infectious_fit=param_fit
        self.infectious_rate=fun_fit
        self.error_infectious=error_infectious
        return self.param_infectious_fit,self.infectious_rate,self.error_infectious
    
    def compute_lamda_train_point(self,t):
        return lambda_t(np.array([t]),
                        self.Beta,
                        self.f_covariates,
                        self.keys_train,
                        self.Tweets,
                        self.kernel,
                        self.infectious_rate,
                        self.followers_rate)[0]
    
    def compute_lambda_train(self):
        self.lambda_train = lambda_t(self.t_train,
                                    self.Beta,
                                    self.f_covariates,
                                    self.keys_train,
                                    self.Tweets,
                                    self.kernel,
                                    self.infectious_rate,
                                    self.followers_rate)
        if self.method_pred == 'integral':
            self.Tweets_est_train=get_tweets_from_lambda(self.lambda_train, self.t_train)
        elif self.method_pred == 'thinning':
            self.Tweets_est_train=thinning_pred(self.lambda_train,self.t_train)
        else:
            return "Metodo de predicción invalido"
        
        return self.lambda_train, self.Tweets_est_train
    
    def compute_lamda_predict_point(self,t):
        return lambda_pred(np.array([t]),
                           self.keys_train,
                           self.Tweets,
                           self.Beta,
                           self.f_covariates,
                           self.kernel,
                           self.kernel_integral,
                           self.infectious_rate,
                           self.followers_rate,
                           self.train_end)[0]

    def compute_lambda_predict(self):
        self.lambda_predict = lambda_pred(self.t_pred,
                                          self.keys_train,
                                          self.Tweets,
                                          self.Beta,
                                          self.f_covariates,
                                          self.kernel,
                                          self.kernel_integral,
                                          self.infectious_rate,
                                          self.followers_rate,
                                          self.train_end)
        
        if self.method_pred == 'integral':
            self.Tweets_pred=get_tweets_from_lambda(self.lambda_predict, self.t_pred)
        elif self.method_pred == 'thinning':
            self.Tweets_pred=thinning_pred(self.lambda_predict,self.t_pred)
        else:
            return "Metodo de predicción invalido"
        
        
        return self.lambda_predict,self.Tweets_pred
    
    def compute_errors(self):
        ###
        real=self.real_tweets_validate.Tweets.values
        predict=self.Tweets_pred.Tweets.values
        diff=abs(real-predict)
        
        self.errors_predict={}
        ##
        self.errors_predict['APE']=diff/real
        self.errors_predict['MAPE']=self.errors_predict['APE'].mean()
        self.errors_predict['MSE']=(diff**2).mean()
        self.errors_predict['MAE']=diff.mean()
        self.errors_predict['RMSE']=np.sqrt((diff**2).mean())
        self.errors_predict['Pearson']=stats.pearsonr(real,predict)        
        self.errors_predict['kendall']=stats.kendalltau(real,predict)
        
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
        self.errors_predict_cum['Pearson']=stats.pearsonr(real,predict)        
        self.errors_predict_cum['kendall']=stats.kendalltau(real,predict)
        
        return self.errors_predict,self.errors_predict_cum
        
        
        
    def train(self,beta_0=np.array([1,1,1])):
        self.compute_Beta(beta_0)
        self.compute_p_est()
        self.infectious_rate_fit()
        self.compute_lambda_train()
        
        return self.Beta, self.param_infectious_fit, self.error_infectious, self.lambda_train
#----------------------------------------------------------------------------------------