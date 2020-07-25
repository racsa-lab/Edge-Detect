#!/usr/bin/python3

from math import sqrt
from numpy import concatenate
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten, Conv1D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import tensorflow as tf
import pandas as pd
import sys
 
 

import numpy as np
import sklearn.metrics as sklm
import all_models
print("Please make necessary code changes as per the dataset")
# change shape if selected feature dataset is used
#test_X=np.empty((0,53),float)
#test_y=np.empty((0,1),int)

window_size=sys.argv[2]

test_X=np.load("../SEL_TESTING_X_"+window_size+".npy")
test_y=np.load("../SEL_TESTING_Y_"+window_size+".npy")

window_index=0
num_windows=np.size(test_X,0)
window_size=np.size(test_X,1)
num_features=np.size(test_X,2)

first_layer_num_neurons=int(sys.argv[3])
if (len(sys.argv)) > 4 :
    if sys.argv[4] != 'grnn':
        second_layer_num_neurons=int(sys.argv[4])
        # choose model from all_models
        model=all_models.get_optfastrnnlstm([window_size,num_features],0.1,first_layer_num_neurons,second_layer_num_neurons) #(shape,dropout) in accordance to dataset
    else:
        model=all_models.get_optfastgrnnlstm_single_layer([window_size,num_features],0.1,first_layer_num_neurons) #(shape,dropout) in accordance to dataset

else  :
    model=all_models.get_optfastrnnlstm_single_layer([window_size,num_features],0.1,first_layer_num_neurons) #(shape,dropout) in accordance to dataset

model.load_weights(sys.argv[1])

while window_index < num_windows :
    test_x = test_X[window_index:window_index+1,:,:]
    y=model.predict(test_x)
    window_index=window_index+1
    print("SAMPLE #:", window_index)

