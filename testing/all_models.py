import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten, Conv1D
from tensorflow.keras.layers import  CuDNNGRU,  CuDNNLSTM
from tensorflow.python.keras.layers.recurrent import LSTM,GRU

import pandas as pd

def get_cudnngru(shape,dropout):
        model = Sequential()
        with tf.variable_scope("CuDNNGRU1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNGRU(64,input_shape=(shape),return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("CuDNNGRU2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNGRU(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("CuDNNGRU3" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNGRU(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("CuDNNGRU4" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNGRU(64,return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
        
        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
            
        return model
       
        
def get_cudnnlstm(shape,dropout):
        model = Sequential()
        with tf.variable_scope("CuDNNLSTM1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,input_shape=(shape),return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("CuDNNLSTM2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("CuDNNLSTM3" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("CuDNNLSTM4" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
        return model


def get_cudnn3lstm(shape,dropout):
        model = Sequential()
        with tf.variable_scope("CuDNNLSTM1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,input_shape=(shape),return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("CuDNNLSTM2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("CuDNNLSTM3" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("CuDNNLSTM4" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("CuDNNLSTM5" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("CuDNNLSTM6" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
        return model
        
        
        

def get_cudnncnnlstm(shape,dropout):
        model = Sequential()
        with tf.variable_scope("Conv1D1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Conv1D(128, input_shape=(train_X.shape[1:]), kernel_size=3, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
        with tf.variable_scope("Conv1D2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Conv1D(128,kernel_size=3, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

        with tf.variable_scope("CuDNNLSTM1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("CuDNNLSTM2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("CuDNNLSTM3" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("CuDNNLSTM4" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
        return model
        
 
def get_fastrnnlstm(shape,dropout):
        model = Sequential()
        with tf.variable_scope("LSTM1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(64,input_shape=(shape),return_sequences=True, celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("LSTM2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(64,return_sequences=True,celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("LSTM3" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(64,return_sequences=True,celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("LSTM4" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(64,return_sequences=False,celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
        return model
        
        
def get_fastgrnnlstm(shape,dropout):
        model = Sequential()
        with tf.variable_scope("LSTM1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(64,input_shape=(shape),return_sequences=True, celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("LSTM2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(64,return_sequences=True,celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("LSTM3" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(64,return_sequences=True,celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("LSTM4" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(64,return_sequences=False,celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
        return model
        
        
def get_optfastgrnnlstm(shape,dropout):
        model = Sequential()
        with tf.variable_scope("LSTM1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(128,input_shape=(shape),return_sequences=True, celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("LSTM2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(64,return_sequences=False,celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(32, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()

        return model
        
def get_optfastgrnnlstm_single_layer(shape,dropout,first_layer_neurons):
        model = Sequential()
        with tf.variable_scope("LSTM1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(first_layer_neurons,input_shape=(shape),return_sequences=True, celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(first_layer_neurons, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()

        return model

def get_optfastrnnlstm(shape,dropout,first_layer_neurons,second_layer_neurons):
        model = Sequential()
        with tf.variable_scope("LSTM1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(first_layer_neurons,input_shape=(shape),return_sequences=True, celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("LSTM2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(second_layer_neurons,return_sequences=False,celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(second_layer_neurons, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
        return model


def get_optfastrnnlstm_single_layer(shape,dropout,first_layer_neurons):
        model = Sequential()
        with tf.variable_scope("LSTM1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(first_layer_neurons,input_shape=(shape),return_sequences=True, celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(first_layer_neurons, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
        return model
 
def get_fastrnngru(shape,dropout):
        model = Sequential()
        with tf.variable_scope("GRU1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(64,input_shape=(shape),return_sequences=True, celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("GRU2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(64,return_sequences=True,celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("GRU3" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(64,return_sequences=True,celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("GRU4" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(64,return_sequences=False,celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
        return model
        
        
def get_fastgrnngru(shape,dropout):
        model = Sequential()
        with tf.variable_scope("GRU1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(64,input_shape=(shape),return_sequences=True, celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("GRU2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(64,return_sequences=True,celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("GRU3" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(64,return_sequences=True,celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("GRU4" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(64,return_sequences=False,celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
        return model
        
        
def get_optfastgrnngru(shape,dropout,first_layer_neurons,second_layer_neurons):
        model = Sequential()
        with tf.variable_scope("GRU1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(first_layer_neurons,input_shape=(shape),return_sequences=True, celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("GRU2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(second_layer_neurons,return_sequences=False,celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(second_layer_neurons, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()

        return model
        
def get_optfastrnngru(shape,dropout):
        model = Sequential()
        with tf.variable_scope("GRU1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(128,input_shape=(shape),return_sequences=True, celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("GRU2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(64,return_sequences=False,celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(32, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
        return model
