# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 17:17:11 2017

@author: RAGHAV
"""

import pandas as pd
from fbprophet import Prophet
from lstm_utils import get_raw_data

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
sns.set_context('talk')

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)


if __name__=='__main__':
    
    STOCK_INDEX = '^GSPC'

    sp_df = get_raw_data(STOCK_INDEX) 

    print("Data Retrieved")
    
    # reset index to get date_time as a column
    prophet_df = sp_df.reset_index()
    
    # prepare the required dataframe
    prophet_df.rename(columns={'index':'ds','Close':'y'},inplace=True)
    prophet_df = prophet_df[['ds','y']]
    
    # prepare train and test sets
    train_size = int(prophet_df.shape[0]*0.9)
    train_df = prophet_df.iloc[:train_size]
    test_df = prophet_df.iloc[train_size+1:]
    
    # build a prophet model
    pro_model = Prophet()
    
    # fit the model
    pro_model.fit(train_df)
    
    # prepare a future dataframe
    test_dates = pro_model.make_future_dataframe(periods=test_df.shape[0])
    
    # forecast values
    forecast_df = pro_model.predict(test_dates)
    
    # plot the forecast
    pro_model.plot(forecast_df)
    plt.show()
    
    # plot against true data
    plt.plot(forecast_df.yhat,c='r',label='Forecast')
    plt.plot(forecast_df.yhat_lower.iloc[train_size+1:],
             linestyle='--',c='b',alpha=0.3,
             label='Confidence Interval')
    plt.plot(forecast_df.yhat_upper.iloc[train_size+1:],
             linestyle='--',c='b',alpha=0.3,
             label='Confidence Interval')
    plt.plot(prophet_df.y,c='g',label='True Data')
    plt.legend()
    plt.title('Prophet Model Forecast Against True Data')
    plt.show()
    
    
    