import numpy as np
import librosa
import pandas as pd
import math

from tensorflow import keras
import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, LSTM, BatchNormalization, Bidirectional, Reshape, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.compat.v1.keras.backend import set_session
tf.config.list_physical_devices('GPU')


class DataGenerator(keras.utils.Sequence):
    def __init__(self, df, batch_size, shuffle=True):
        self.df = df.reset_index()
        self.x, self.y = df['path'], df['MOS']
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)
    
    def replaceZeroes(self, data):
        min_nonzero = np.min(data[np.nonzero(data)])
        data[data == 0] = min_nonzero
        return data

    
    def collate(self, fpath, maxlen=751):
        audio, _ = librosa.core.load(fpath, sr=16000)
        audio = self.replaceZeroes(librosa.stft(audio, n_fft=512))
        audio = 10*np.log10(audio)
        audio = np.pad(audio, ((0, 0), (0, maxlen-audio.shape[1])))
        return audio
    
    def on_epoch_end(self):
        if self.shuffle == True:
            self.df = df.sample(frac=1, random_state=42)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        
        X = []
        for f in batch_x:
            x = self.collate(f)
            X.append(x)
        return np.array(X), np.array(batch_y)

def create_model(
        IN_shape,
        nb_bilstm_layer=3, 
        nb_units=[128, 64, 32],
        nb_dense_layer=1,
        nb_hiddens=[32],
        ):  
    
    time_dim = 751
    
    inputs = Input(shape=IN_shape)
    bilstm_out = Bidirectional(LSTM(nb_units[0], return_sequences=True), input_shape=(time_dim, freq_dim), merge_mode='concat')(inputs)
    bilstm_out = Lambda(lambda x: x[:, :-1, :])(bilstm_out) 
    time_dim //= 2
    bilstm_out = Reshape((time_dim, 4 * nb_units[0]))(bilstm_out)
   
    for bilstm_idx in range(1, nb_bilstm_layer - 1):
        bilstm_out = Bidirectional(LSTM(nb_units[bilstm_idx], return_sequences=True), input_shape=(time_dim, 2 * nb_units[bilstm_idx - 1]), merge_mode='concat')(bilstm_out)        
        bilstm_out = Lambda(lambda x: x[:, :-1, :])(bilstm_out)
        time_dim //= 2
        bilstm_out = Reshape((time_dim, 4 * nb_units[bilstm_idx]))(bilstm_out)

    bilstm_out = Bidirectional(LSTM(nb_units[-1], return_sequences=False), merge_mode='concat')(bilstm_out)
    attention_mul = tf.keras.layers.Attention()([bilstm_out, bilstm_out])
    outputs = Flatten()(attention_mul)
    
    for dense_idx in range(nb_dense_layer):
        outputs = Dense(nb_hiddens[dense_idx], kernel_initializer='normal', use_bias=False)(outputs)
        outputs = BatchNormalization()(outputs)
        outputs = Activation('relu')(outputs)
        
    outputs = Dense(1, activation='linear')(outputs)
    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(loss='mean_squared_error',
                           optimizer=Adam(), 
                           metrics=['mae']) 
    
    return model

def train_model(
        df_path, 
        model, 
        batch_num, 
        epochs_num
        ):
    model_save_path = '/nobackup/anakuzne/models/mos-predict/'
    best_model_save_path = model_save_path + '/best_model.h5' 

    mcp = ModelCheckpoint(
            filepath=best_model_save_path, monitor='val_loss', 
            verbose=1, save_best_only=True, 
            mode='min', period=1
            )
    rlr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, 
            patience=10, verbose=1, 
            min_lr=1e-6, mode='min',
            )
    
    callbacks_list = [mcp, rlr]

    df = pd.load_csv(df_path)

    train = df[:int(len(df)*0.8)]
    test = df[int(len(df)*0.8):].reset_index()
    dev = train[:int(len(train)*0.1)].reset_index()
    train = train[int(len(train)*0.1):].reset_index()

    training_gen = DataGenerator(train, batch_size=batch_num)
    validation_gen = DataGenerator(dev, batch_size=batch_num)

    history = model.fit_generator(generator=training_gen,
                                 validation_data=validation_gen, 
                                 epochs=epochs_num, verbose=1,
                                 callbacks = callbacks_list)

    return model, history


epochs_num = 100 
batch_num = 24

freq_dim = 257 
time_dim = 751
IN_shape = (time_dim, freq_dim) 

model = create_model(IN_shape)
train_model('/nobackup/anakuzne/data/COSINE-orig/csv/all.csv', model, batch_num, epochs_num)