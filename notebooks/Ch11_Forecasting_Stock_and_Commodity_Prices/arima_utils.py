# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 19:21:24 2017

@author: RAGHAV
"""

import itertools
import numpy as np
import pandas as pd


import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.model_selection import TimeSeriesSplit

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sns.set_style('whitegrid')
sns.set_context('talk')



# Dickey Fuller test for Stationarity
def ad_fuller_test(ts):
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic',
                                             'p-value',
                                             '#Lags Used',
                                             'Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

# Plot rolling stats for a time series
def plot_rolling_stats(ts):
    rolling_mean = ts.rolling(window=12,center=False).mean()
    rolling_std = ts.rolling(window=12,center=False).std()

    #Plot rolling statistics:
    orig = plt.plot(ts, color='blue',label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    
def plot_acf_pacf(series):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(series.squeeze(), lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(series, lags=40, ax=ax2)    
    

def auto_arima(param_max=1,series=pd.Series(),verbose=True):
    # Define the p, d and q parameters to take any value 
    # between 0 and param_max
    p = d = q = range(0, param_max+1)

    # Generate all different combinations of seasonal p, d and q triplets
    pdq = [(x[0], x[1], x[2]) for x in list(itertools.product(p, d, q))]
    
    model_resuls = []
    best_model = {}
    min_aic = 10000000
    for param in pdq:
        try:
            mod = sm.tsa.ARIMA(series, order=param)

            results = mod.fit()
            
            if verbose:
                print('ARIMA{}- AIC:{}'.format(param, results.aic))
            model_resuls.append({'aic':results.aic,
                                 'params':param,
                                 'model_obj':results})
            if min_aic>results.aic:
                best_model={'aic':results.aic,
                            'params':param,
                            'model_obj':results}
                min_aic = results.aic
        except Exception as ex:
            print(ex)
    if verbose:
        print("Best Model params:{} AIC:{}".format(best_model['params'],
              best_model['aic']))  
        
    return best_model, model_resuls


def arima_gridsearch_cv(series, cv_splits=2,verbose=True,show_plots=True):
    # prepare train-test split object
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    # initialize variables
    splits = []
    best_models = []
    all_models = []
    i = 1
    
    # loop through each CV split
    for train_index, test_index in tscv.split(series):
        print("*"*20)
        print("Iteration {} of {}".format(i,cv_splits))
        i = i + 1
        
        # print train and test indices
        if verbose:
            print("TRAIN:", train_index, "TEST:", test_index)
        splits.append({'train':train_index,'test':test_index})
        
        # split train and test sets
        train_series = series.ix[train_index]
        test_series = series.ix[test_index]
        
        print("Train shape:{}, Test shape:{}".format(train_series.shape,
              test_series.shape))
        
        # perform auto arima
        _best_model, _all_models = auto_arima(series=train_series)
        best_models.append(_best_model)
        all_models.append(_all_models)
        
        # display summary for best fitting model
        if verbose:
            print(_best_model['model_obj'].summary())
        results = _best_model['model_obj']
        
        if show_plots:
            # show residual plots
            residuals = pd.DataFrame(results.resid)
            residuals.plot()
            plt.title('Residual Plot')
            plt.show()
            residuals.plot(kind='kde')
            plt.title('KDE Plot')
            plt.show()
            print(residuals.describe())
        
            # show forecast plot
            fig, ax = plt.subplots(figsize=(18, 4))
            fig.autofmt_xdate()
            ax = train_series.plot(ax=ax)
            test_series.plot(ax=ax)
            fig = results.plot_predict(test_series.index.min(), 
                                       test_series.index.max(), 
                                       dynamic=True,ax=ax,
                                       plot_insample=False)
            plt.title('Forecast Plot ')
            plt.legend()
            plt.show()

            # show error plot
            insample_fit = list(results.predict(train_series.index.min()+1, 
                                                train_series.index.max(),
                                                typ='levels')) 
            plt.plot((np.exp(train_series.ix[1:].tolist())-\
                             np.exp(insample_fit)))
            plt.title('Error Plot')
            plt.show()
    return {'cv_split_index':splits,
            'all_models':all_models,
            'best_models':best_models}
    
# results.predict(test_series.index.min(), test_series.index.max(),typ='levels')
def plot_on_original(train_series,test_series,forecast_series):
    # show forecast plot on original series
    fig, ax = plt.subplots(figsize=(18, 4))
    fig.autofmt_xdate()
    plt.plot(train_series,c='black')
    plt.plot(test_series,c='blue')
    plt.plot(np.exp(forecast_series),c='g')
    plt.title('Forecast Plot with Original Series')
    plt.legend()
    plt.show()    