# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 10:59:03 2017

@author: RAGHAV
"""

import quandl
import numpy as np
import pandas as pd

from arima_utils import ad_fuller_test, plot_rolling_stats
from arima_utils import plot_acf_pacf, arima_gridsearch_cv

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
sns.set_context('talk')



if __name__ == '__main__':
    commodity_dataset_name = "BUNDESBANK/BBK01_WT5511"
    gold_df = quandl.get(commodity_dataset_name, end_date="2017-07-31")
    
    # handle missing time stamps
    gold_df = gold_df.reindex(pd.date_range(gold_df.index.min(), 
                                      gold_df.index.max(), 
                                      freq='D')).fillna(method='ffill')
    
    gold_df.plot(figsize=(15, 6))
    plt.show()
    
    # log series
    log_series = np.log(gold_df.Value)
    
    ad_fuller_test(log_series)
    plot_rolling_stats(log_series)
    
    # Using log series with a shift to make it stationary
    log_series_shift = log_series - log_series.shift()
    log_series_shift = log_series_shift[~np.isnan(log_series_shift)]
    
    ad_fuller_test(log_series_shift)
    plot_rolling_stats(log_series_shift)
    
    # determining p and q
    plot_acf_pacf(log_series_shift)
    
    gold_df['log_series'] = log_series
    gold_df['log_series_shift'] = log_series_shift
    
    # cross validate 
    results_dict = arima_gridsearch_cv(gold_df.log_series,cv_splits=2)