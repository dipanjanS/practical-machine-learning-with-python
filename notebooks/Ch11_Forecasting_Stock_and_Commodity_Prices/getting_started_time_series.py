# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 15:33:10 2017

@author: RAGHAV
"""

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('talk')


if __name__=='__main__':
    #load data
    input_df = pd.read_csv(r'website-traffic.csv')
    
    input_df['date_of_visit'] = pd.to_datetime(input_df.MonthDay.\
                                                        str.cat(
                                                                input_df.Year.astype(str), 
                                                                sep=' '))
    
    print(input_df[['date_of_visit','Visits']].head(10))
    
    # plot time series
    input_df.plot(x='date_of_visit',
                  y='Visits', 
                  title= "Website Visits per Day")
    
    # extract visits as series from the dataframe
    ts_visits = pd.Series(input_df.Visits.values
                          ,index=pd.date_range(
                                                input_df.date_of_visit.min()
                                                , input_df.date_of_visit.max()
                                                , freq='D')
                         )
                          
    
    deompose = seasonal_decompose(ts_visits.interpolate(),
                                    freq=24)
    deompose.plot()
    
    # moving average  
    input_df['moving_average'] = input_df['Visits'].rolling(window=3,
                                                            center=False).mean()
    
    print(input_df[['Visits','moving_average']].head(10)) 
    
    plt.plot(input_df.Visits,'-',color='black',alpha=0.3)
    plt.plot(input_df.moving_average,color='b')
    plt.title('Website Visit and Moving Average Smoothening')
    plt.legend()
    plt.show()
    
    
    # exponentially weighted moving average
    input_df['ewma'] = input_df['Visits'].ewm(halflife=3,
                                                ignore_na=False,
                                                min_periods=0,
                                                adjust=True).mean()
    
    plt.plot(input_df.Visits,'-',color='black',alpha=0.3)
    plt.plot(input_df.ewma,color='g')
    plt.title('Website Visit and Exponential Smoothening')
    plt.legend()
    plt.show()
                      