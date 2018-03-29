# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 12:22:37 2017

@author: RAGHAV
"""

import math
import numpy as np

import keras
from lstm_utils import get_raw_data
from lstm_utils import get_reg_model
from lstm_utils import plot_reg_results
from lstm_utils import get_reg_train_test
from lstm_utils import predict_reg_multiple

from sklearn.metrics import mean_squared_error


if __name__=='__main__':
    
    WINDOW = 6
    PRED_LENGTH = int(WINDOW/2)  
    STOCK_INDEX = '^GSPC'

    sp_df = get_raw_data(STOCK_INDEX)
    sp_close_series = sp_df.Close 

    print("Data Retrieved")
    
    x_train,y_train,x_test,y_test,scaler = get_reg_train_test(sp_close_series,
                                                      sequence_length=WINDOW+1,
                                                      roll_mean_window=None,
                                                      normalize=True,
                                                      scale=False)
    
    print("Data Split Complete")
    
    print("x_train shape={}".format(x_train.shape))
    print("y_train shape={}".format(y_train.shape))
    print("x_test shape={}".format(x_test.shape))
    print("y_test shape={}".format(y_test.shape))
    
    lstm_model=None
    try:
        lstm_model = get_reg_model(layer_units=[50,100],
                               window_size=WINDOW)   
    except:
        print("Model Build Failed. Trying Again")
        lstm_model = get_reg_model(layer_units=[50,100],
                               window_size=WINDOW) 
        
    # use eatrly stopping to avoid overfitting     
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=2,
                                               verbose=0)]
    lstm_model.fit(x_train, y_train, 
                   epochs=20, batch_size=16,
                   verbose=1,validation_split=0.05,
                   callbacks=callbacks)
    print("Model Fit Complete")
    
    train_pred_seqs = predict_reg_multiple(lstm_model,
                                                 x_train,
                                                 window_size=WINDOW,
                                                 prediction_len=PRED_LENGTH)
    
    train_offset = y_train.shape[0] - np.array(train_pred_seqs).flatten().shape[0]
    train_rmse = math.sqrt(mean_squared_error(y_train[train_offset:], 
                                              np.array(train_pred_seqs).\
                                              flatten()))
    print('Train Score: %.2f RMSE' % (train_rmse))
    
    
    test_pred_seqs = predict_reg_multiple(lstm_model,
                                          x_test,
                                          window_size=WINDOW,
                                          prediction_len=PRED_LENGTH)
    
    test_offset = y_test.shape[0] - np.array(test_pred_seqs).flatten().shape[0]
    test_rmse = math.sqrt(mean_squared_error(y_test[test_offset:], 
                                              np.array(test_pred_seqs).\
                                              flatten()))
    print('Test Score: %.2f RMSE' % (test_rmse))
    
    #pred_seqs = predict_point_by_point(lstm_model,x_test)
    print("Prediction Complete")
    plot_reg_results(test_pred_seqs,y_test,prediction_len=PRED_LENGTH)
