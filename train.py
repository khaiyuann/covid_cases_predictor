# -*- coding: utf-8 -*-
"""
Created on Fri May 20 09:06:51 2022

This script is used to train an LSTM model to predict the number of new
COVID-19 cases based on the previous 30 days new cases.

@author: LeongKY
"""
#%% Imports
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.preprocessing import MinMaxScaler
from timeseries_modules import TimeSeriesModeller

#%% Statics
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'saved_model', 'model.h5')
TRAIN_PATH = os.path.join(os.getcwd(), 'datasets', 'cases_malaysia_train.csv')
TEST_PATH = os.path.join(os.getcwd(), 'datasets', 'cases_malaysia_test.csv')
SCALER_PATH = os.path.join(os.getcwd(), 'saved_model', 'scaler.pkl')
LOG_PATH = os.path.join(os.getcwd(), 'logs')

#%% 1. Load data
train_df = pd.read_csv(TRAIN_PATH, parse_dates=['date'])
test_df = pd.read_csv(TEST_PATH, parse_dates=['date'])

#%% 2. Inspect data
print(train_df.describe())
print(train_df.head())

print(train_df.info())
print(test_df.info())
# observed object dtype for train set, and 1 null value for test set

#%% 3. Clean data
#extract train and test columns
train = train_df['cases_new']
test = test_df['cases_new']

# use to_numeric and ffill to clean and impute missing data
ts = TimeSeriesModeller()
train = ts.clean_fill_series(train, 'ffill')
test = ts.clean_fill_series(test, 'ffill')

# visualize data after cleaning
plt.figure()
plt.plot(train.values)
plt.show()

#%% 4. Preprocess data
# scaling data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(np.expand_dims(train, -1))
X_test_scaled = scaler.transform(np.expand_dims(test, -1))
pickle.dump(scaler, open(SCALER_PATH, 'wb'))

# declare window size
window_size = 30

# generate train range
X_train, y_train = ts.range_selector(X_train_scaled, window_size)

# generate test range
joined = np.concatenate((X_train_scaled, X_test_scaled))   
len_test = len(X_test_scaled) + window_size
testset = joined[-len_test:,:]

X_test, y_test = ts.range_selector(testset, window_size)
    
#%% 5. Create model
model = ts.create_model(input_shape=(X_train.shape[1:]), 
                        nb_lstm=64, 
                        dropout=0.2)

# callbacks
log_files = os.path.join(LOG_PATH, datetime.now().strftime('%Y%m%d-%H%M%S')) 
es_callback = EarlyStopping(monitor='loss', patience=3)
tb_callback = TensorBoard(log_dir=log_files)

# compile & train model
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

hist = model.fit(X_train, y_train, epochs=50, batch_size=128,
                 callbacks=[es_callback, tb_callback])

#%% 6. Evaluate model
# predict test data and score model
y_pred = ts.predict_score_model(model, X_test, y_test)

# obtain actual values from scaler
y_true = scaler.inverse_transform(np.expand_dims(y_test, axis=-1))
y_pred = scaler.inverse_transform(y_pred)

# plot predicted and true values
legend = ['Actual new cases', 'Predicted new cases']
ts.compare_results(y_true, y_pred, legend)

# save model for deployment
model.save(MODEL_SAVE_PATH)