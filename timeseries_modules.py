# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:08:29 2022

This class file contains the functions used in the development of the
covid cases predictor model.

@author: LeongKY
"""
#%% Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_absolute_error

#%% Classes
class TimeSeriesModeller():
    def __init__(self):
        pass
    
    def clean_fill_series(self, data, method):
        '''
        This function is used to clean any non-numeric data from time-series
        data and perform imputation using the specified method for NaN data.

        Parameters
        ----------
        data : DataFrame/series
            time-series data to be processed.
        method : stf
            method to impute data (ffill/bfill).

        Returns
        -------
        data : DataFrame/series
            cleaned data.

        '''
        data = pd.to_numeric(data, errors='coerce').fillna(method=method)
        
        return data
    
    def range_selector(self, data, window_size):
        '''
        This function is used to select the range of features and labels for
        time-series data based on the provided window size.

        Parameters
        ----------
        data : DataFrame/Array
            time-series data to select features and target.
        window_size : int
            size of window to capture as features.

        Returns
        -------
        X : Array
            time-series data features.
        y : Array
            time-series data target.

        '''
        X = []
        y = []
        
        # generate range
        for i in range(window_size,len(data)):
            X.append(data[i-window_size:i,0])
            y.append(data[i,0])
        
        # array conversion
        X = np.expand_dims(np.array(X), -1)
        y = np.array(y)
            
        return X, y
    
    def create_model(self, input_shape, nb_lstm, dropout):
        '''
        This function is used to generate a model with 2 LSTM layers with
        dropout for time-series prediction.

        Parameters
        ----------
        input_shape : tuple
            input shape of features.
        nb_lstm : int
            number of LSTM nodes.
        dropout : float
            dropout parameter.

        Returns
        -------
        model : model
            model created.

        '''
        model = Sequential()
        model.add(LSTM(nb_lstm, activation='tanh',
                       return_sequences=(True),
                       input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(LSTM(nb_lstm))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        plot_model(model, os.path.join(os.getcwd(), 'results', 'model.png'))
        model.summary()
        
        return model
    
    def predict_score_model(self, model, X_test, y_test):
        '''
        This function is used to perform prediction using the defined model
        and score the model using the mean absolute percentage error (MAPE).

        Parameters
        ----------
        model : model
            trained model.
        X_test : Array
            test features.
        y_test : Array
            test target.

        Returns
        -------
        y_pred : Array
            predicted target.

        '''
        # predict labels
        predicted = []
        for test in X_test:
            predicted.append(model.predict(np.expand_dims(test, axis=0)))
            
        predicted = np.array(predicted)
        
        # model performance
        y_pred = predicted.reshape(len(predicted),1)
        
        mape = (mean_absolute_error(y_test, y_pred)/sum(abs(y_test)))*100
        print('\nThe mean absolute percentage error (MAPE) is ' 
              + str('{:.3f}'.format(mape)) +'%')
        
        return y_pred
    
    def compare_results(self, y_true, y_pred, legend):
        '''
        This function is used to plot and compare the actual target and the
        predicted target for result presentation and visualization.

        Parameters
        ----------
        y_true : Array
            actual target values.
        y_pred : Array
            predicted target values.
        legend : list
            list containing labels of respective data.

        Returns
        -------
        None.

        '''
        plt.figure()
        plt.plot(y_true, color='r')
        plt.plot(y_pred, color='b')
        plt.legend(legend)
        plt.show()
        