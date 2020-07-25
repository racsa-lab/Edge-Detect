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
from tensorflow.keras.layers import CuDNNLSTM
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import minmax_scale
import sys
 
 
import keras
import numpy as np
import sklearn.metrics as sklm





# convert series to supervised learning
timestep=int(sys.argv[1])
# load dataset
dataset = pd.read_csv('training.csv', header=0)
dataset=dataset.dropna()
dataset=dataset.fillna(method="ffill")
dataset = dataset.apply(pd.to_numeric,errors="coerce")
train_y=dataset.values[:,-1]
orgshape=dataset.shape[1]
columns = [dataset.shift(i) for i in range(timestep,0,-1)]
dataset = pd.concat(columns, axis=1)
values = dataset.values.reshape(dataset.shape[0],timestep,orgshape)

for i in range(timestep):
	 values=np.delete(values,0,axis=0)
	 train_y=np.delete(train_y,0,axis=0)


# split into train and test sets

train = values[:, :,:]
# split into input and outputs
train_X=train[:,:,:-1]


dataset = pd.read_csv('testing.csv', header=0)
dataset=dataset.dropna()
dataset=dataset.fillna(method="ffill")
dataset = dataset.apply(pd.to_numeric,errors="coerce")
test_y=dataset.values[:,-1]
columns = [dataset.shift(i) for i in range(timestep,0,-1)]
dataset = pd.concat(columns, axis=1)
values = dataset.values.reshape(dataset.shape[0],timestep,orgshape)
for i in range(timestep):
	 values=np.delete(values,0,axis=0)
	 test_y=np.delete(test_y,0,axis=0)
# integer encode direction

test = values[:, :,:]
# split into input and outputs



test_X=test[:, :,:-1]
# reshape input to be 3D [samples, timesteps, features]

print (train_X.shape,train_y.shape,test_X.shape,test_y.shape)

np.save("TRAINING_X_"+str(timestep)+".npy",train_X)
np.save("TRAINING_Y_"+str(timestep)+".npy",train_y)

np.save("TESTING_X_"+str(timestep)+".npy",test_X)
np.save("TESTING_Y_"+str(timestep)+".npy",test_y)
