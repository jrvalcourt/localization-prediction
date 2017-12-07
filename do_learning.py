import pickle
import os
import gzip
import numpy as np
import keras
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv1D,MaxPooling1D,Flatten
from keras.layers import Dropout,LSTM,GlobalAveragePooling1D
import tensorflow as tf

np.random.seed(7)

pickle_dir = 'data/'
x_data_filename     = os.path.join(pickle_dir, 'X.p.gz')
y_data_filename     = os.path.join(pickle_dir, 'Y.p.gz')
names_data_filename = os.path.join(pickle_dir, 'names.p.gz')
features_filename   = os.path.join(pickle_dir, 'features.p.gz')

# load data
x_data = pickle.load(gzip.open(x_data_filename))
y_data = pickle.load(gzip.open(y_data_filename))

# split into training and test sets
training_indices = np.random.choice(np.arange(0, 
    x_data.shape[0]), size=int(x_data.shape[0] * 0.8), replace=False)
training_filter = np.isin(np.arange(0, x_data.shape[0]), training_indices)
test_filter = np.logical_not(training_filter)

x_train = x_data[training_filter, :, :]
x_test  = x_data[test_filter,     :, :]
y_train = y_data[training_filter, :]
y_test  = y_data[test_filter,     :]

# store the datasets
pickle.dump(x_train, gzip.open(os.path.join(pickle_dir, 'x_train.p.gz'), 'wb'))
pickle.dump(y_train, gzip.open(os.path.join(pickle_dir, 'y_train.p.gz'), 'wb'))
pickle.dump(x_test,  gzip.open(os.path.join(pickle_dir, 'x_test.p.gz'),  'wb'))
pickle.dump(y_test,  gzip.open(os.path.join(pickle_dir, 'y_test.p.gz'),  'wb'))

# set up the model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=8, activation='relu', 
    input_shape=(x_data.shape[1], x_data.shape[2])))
model.add(Dropout(0.3))
model.add(MaxPooling1D(pool_size=4))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(5)))
model.add(Dropout(0.3))
model.add(Dense(y_data.shape[1], activation='sigmoid'))
opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=opt)

# an LSTM model
#model = Sequential()
#model.add(LSTM(128, input_shape=(x_data.shape[1], x_data.shape[2])))
#model.add(Dropout(0.2))
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(y_data.shape[1], activation='sigmoid'))
#opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#model.compile(loss='binary_crossentropy', optimizer=opt)

# fit and save the model
model.fit(x_train, y_train, batch_size=32, epochs=30, 
        validation_data=(x_test, y_test))
model.save('model.h5')
scores = model.evaluate(x_test, y_test, batch_size=32)
print(scores)
