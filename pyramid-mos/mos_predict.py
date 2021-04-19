#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:25:04 2019

@author: xuandong
"""

from __future__ import print_function

import numpy as np
np.random.seed(13579)
import random as rn
rn.seed(13579)
import tensorflow as tf
tf.set_random_seed(13579)
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import itertools
import math
import matplotlib.pyplot as plt
import h5py
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, KFold
from glob import glob
from keras.models import Sequential, Model, load_model 
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, LSTM, BatchNormalization, Bidirectional, multiply, Reshape, Permute
from keras.utils import multi_gpu_model
from keras import metrics
from keras.optimizers import Adam, SGD, Nadam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras.backend as K

import pandas as pd


algo = 'bilstm_regr_attn'
score_type = 'pesq'
work_dir = '/nobackup/anakuzne/models/mos-predict'
freq_dim = 321
time_dim = 166
#filenames = glob(work_dir + '/../../Matlab_Project/data_place/mixDataMultilabels_5000tr_0te_Babble_13trSNR/**/*.mat')
csv_all = pd.read_csv('/nobackup/anakuzne/data/COSINE-orig/csv/all.csv')
filenames = csv_all['path']
mos_scores = csv_all['MOS']
X_data = np.empty([len(filenames), freq_dim, time_dim]) # (batch_size, timesteps, input_dim)
Y_data = np.empty([len(filenames)])
    
'''
for idx, file in enumerate(filenames):
    with h5py.File(file, 'r') as mat_data:  # use mat_data.keys() to check stored variables
        X_data[idx, :, :] = np.transpose(mat_data['feats'])  # need to transpose HDF data
        Y_data[idx] = np.array(mat_data[score_type + '_score'])
    mat_data.close()
'''
    
train_num = 7000
test_num = 2178
batch_num = 64
epochs_num = 500

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, train_size = train_num, test_size = test_num, random_state = 2019, shuffle = True)
del X_data, Y_data


#print('Scale input features to [0, 1] range...')
#minx = np.min(X_train)
#maxx = np.max(X_train)
#print ('Train min, max:', minx, maxx)
#X_train = (X_train-minx)/(maxx-minx) 
#X_test = (X_test-minx)/(maxx-minx)  
print('Normalize input features to zero-mean and unit-variance...')
meanx = X_train.mean()
stdx = X_train.std()
print ('Train mean, std:', meanx, stdx)
X_train = (X_train - meanx) / stdx 
X_test = (X_test - meanx) / stdx


#tr_num, tm, freq= X_train.shape # input (samples��timesteps��input_dim); if return_sequences=True, then output (samples��timesteps��output_dim), else return (samples��output_dim)
#te_num = X_test.shape[0] 

#def root_mean_squared_error(y_true, y_pred):
#        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
    
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    output_dim1 = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(freq_dim, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    
    return output_attention_mul
    
# create and compile the BiLSTM network
def create_bilstm_regr_attn_model(IN_shape):        
    output_dim1 = 100 #64
    output_dim2 = 50 #32
    model_signature = str(output_dim1) + '#' + str(output_dim2)
    
    inputs = Input(shape=IN_shape)
#    lstm_out = LSTM(output_dim1, return_sequences=True)(inputs)
    bilstm_out = Bidirectional(LSTM(output_dim1, return_sequences=True), merge_mode='concat')(inputs)
    attention_mul = attention_3d_block(bilstm_out)
    attention_flatten = Flatten()(attention_mul)
    outputs = Dense(output_dim2, activation = 'elu')(attention_flatten)
    outputs = Dense(output_dim2, activation = 'elu')(outputs)
    outputs = Dense(1, activation='linear')(outputs)
    model = Model(input=[inputs], output=[outputs])
    model.summary()

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #adam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0)
    #rmsp = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(loss='mean_squared_error',
                           optimizer=Nadam(), # Adadelta(), Adam()
                           metrics=['mae']) # 'mse','mape','msle'
    
    return parallel_model, model_signature
    

# Fit the BiLSTM network, and save the best model
bilstm_regr_attn_model, model_signature = create_bilstm_regr_attn_model(X_train.shape[1:])

filename_suffix = '{0}_{1}tr_{2}te_{3}epoch_{4}batch_{5}arch'.format(score_type, train_num, test_num, epochs_num, batch_num, model_signature)

best_model_save_path = work_dir + '/trainedModels/bilstm_regr_attn_best_model_' + filename_suffix + '.h5' 
mcp = ModelCheckpoint(best_model_save_path, monitor='val_mean_absolute_error', verbose=1, save_best_only=True, mode='min', period=1)
rlr = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.5, patience=100, verbose=1, min_lr=1e-4, mode='min')
es = EarlyStopping(monitor='val_mean_absolute_error', patience=200, verbose=1, mode='min', restore_best_weights=False)
callbacks_list = [mcp, rlr, es]

history = bilstm_regr_attn_model.fit(X_train, Y_train,
                  batch_size = batch_num, epochs = epochs_num,
                  callbacks = callbacks_list, 
                  verbose = 1, validation_split = 0.2) # ...holding out last 20% of the data for validation

fig_save_path = work_dir+'/fig_tb_res/'


# Save the whole model
model_save_path = work_dir + '/trainedModels/bilstm_regr_attn_model_' + filename_suffix + '.h5'
#bilstm_regr_attn_model.save(model_save_path)
print('Saved trained model at %s ' % model_save_path)

# Save weights as HDF5 (h5)
#weights_save_path = work_dir+'/trainedModels/bilstm_weights_'+filename_suffix+'.h5'
#bilstm_model.save_weights(weights_save_path)
#print('Saved model weights at %s ' % weights_save_path)

# Save weights as numpy array (npy)
#weights_arr = bilstm_model.get_weights()
#np.save(weights_save_path[:-3], weights_arr)

## Loading the whole model (architecture + weights + training configuration + optimizer state)
#bilstm_model = load_model(best_model_save_path, custom_objects={'acc_top3': acc_top3})

## Loading only the weights into a model with the same architecture
#bilstm_model.load_weights(weights_save_path)

## Loading upon restarting the kernel, and setting weights of the new model
#trained_weights = np.load(weights_save_path[:-3]+'.npy')
#bilstm_model.set_weights(trained_weights)


# Score trained model.
scores = bilstm_regr_attn_model.evaluate(X_test, Y_test, batch_size=2*batch_num, verbose=1)  # Returns the loss value & metrics values for the model in test mode
print('Test mse:', scores[0])
print('Test mae:', scores[1])


pred_y = bilstm_regr_attn_model.predict(X_test, batch_size=2*batch_num, verbose=1) # Generates output predictions for the input samples

# scatter plot
font = {'family': 'sans-serif',
        'color':  'darkblue',
        'weight': 'normal',
        'size': 12 }

plt.figure(figsize=[10, 10])
plt.scatter(Y_test, pred_y)
plt.xlabel('True score', fontdict=font)
plt.ylabel('Predicted score', fontdict=font)
plt.title('Scatter of the true and predicted scores (regr)', fontdict=font)
plt.savefig(fig_save_path + '/bilstm_regr_attn_pred_scatter_' + filename_suffix + '.png')

print('Pearson correlation coefficient: ', np.corrcoef(pred_y[:,0], Y_test)[0][1]) # pesq: 0.896800086188109; stoi: 0.9710058764935302