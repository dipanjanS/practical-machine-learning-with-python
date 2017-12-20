# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 12:22:37 2017

@author: RAGHAV
"""
import math
import warnings
import numpy as np

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

# specify to ignore warning messages
warnings.filterwarnings("ignore") 

from lstm_utils import get_raw_data
from lstm_utils import get_seq_model
from lstm_utils import get_seq_train_test
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import mean_squared_error


if __name__=='__main__':
    
    TRAIN_PERCENT = 0.7
    STOCK_INDEX = '^GSPC'
    VERBOSE=True
    
    # load data
    sp_df = get_raw_data(STOCK_INDEX)
    sp_close_series = sp_df.Close 

    print("Data Retrieved")
    
    # split train and test datasets
    train,test,scaler = get_seq_train_test(sp_close_series,
                                       scaling=True,
                                       train_size=TRAIN_PERCENT)

    train = np.reshape(train,(1,train.shape[0],1))
    test = np.reshape(test,(1,test.shape[0],1))
    
    train_x = train[:,:-1,:]
    train_y = train[:,1:,:]
    
    test_x = test[:,:-1,:]
    test_y = test[:,1:,:]
        
    print("Data Split Complete")
    
    print("train_x shape={}".format(train_x.shape))
    print("train_y shape={}".format(train_y.shape))
    print("test_x shape={}".format(test_x.shape))
    print("test_y shape={}".format(test_y.shape))
    
    # build RNN model
    seq_lstm_model=None
    try:
        seq_lstm_model = get_seq_model(input_shape=(train_x.shape[1],1),
                                                    verbose=VERBOSE)   
    except:
        print("Model Build Failed. Trying Again")
        seq_lstm_model = get_seq_model(input_shape=(train_x.shape[1],1),
                                                    verbose=VERBOSE)
        
    # train the model
    seq_lstm_model.fit(train_x, train_y, 
                   epochs=150, batch_size=1, 
                   verbose=2)
    print("Model Fit Complete")
    
    # train fit performance
    trainPredict = seq_lstm_model.predict(train_x)
    trainScore = math.sqrt(mean_squared_error(train_y[0], trainPredict[0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    
    
    # Pad input sequence
    testPredict = pad_sequences(test_x,
                                    maxlen=train_x.shape[1],
                                    padding='post',
                                    dtype='float64')
    
    # forecast values
    testPredict = seq_lstm_model.predict(testPredict)
    
    # evaluate performances
    testScore = math.sqrt(mean_squared_error(test_y[0], 
                                             testPredict[0][:test_x.shape[1]]))
    
    # inverse transformation
    trainPredict = scaler.inverse_transform(trainPredict.\
                                            reshape(trainPredict.shape[1]))
    testPredict = scaler.inverse_transform(testPredict.\
                                           reshape(testPredict.shape[1]))
    
    # plot the true and forecasted values
    train_size = len(trainPredict)+1

    plt.plot(sp_close_series.index,
             sp_close_series.values,c='black',
             alpha=0.3,label='True Data')
    plt.plot(sp_close_series.index[1:train_size],
             trainPredict,label='Training Fit',c='g')
    plt.plot(sp_close_series.index[train_size+1:],
             testPredict[:test_x.shape[1]],label='Testing Forecast')
    plt.title('Forecast Plot')
    plt.legend()
    plt.show()
