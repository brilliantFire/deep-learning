"""
~*~* Simple Multilayer Perceptron using Keras/Tensorflow *~*~

Code for a simple multilayer perceptron classifier for the MNIST handwritten 
digit data in the context of Kaggle's 'Digit Recognizer' competition.
"""
import pandas as pd
import numpy as np
import os

from tensorflow.python.client import device_lib
from keras.utils.np_utils import to_categorical # for labels
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(237)

# check processors 
device_lib.list_local_devices()

# Set the working directory
os.chdir('C:\\Users\\rlvis\\work_MSPA\\MNIST') # desktop

train_orig = pd.read_csv('data/train.csv')
X_test = pd.read_csv('data/test.csv')

# Hold out 4200 random images (10%) as a validation set
valid = train_orig.sample(n = 4200, random_state = 555)
train = train_orig.loc[~train_orig.index.isin(valid.index)]

# delete original train set to clear space
del train_orig

X_train = train.drop(['label'], axis=1)
labels_train = train['label']

X_valid = valid.drop(['label'], axis=1)
labels_valid = valid['label']

# clear more space
del train, valid

# Normalize
X_train = X_train.astype('float32') / 255.
X_valid = X_valid.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# one hot encoding of digit labels
labels_train = to_categorical(labels_train)
labels_valid = to_categorical(labels_valid)

# Take values out of pandas dataframes
X_train = X_train.values
X_valid = X_valid.values
X_test = X_test.values

# K = 10 digit classes; 784 px images
K_classes = 10
px = X_train.shape[1]

# construct sequential model
simple_mlp = Sequential()
# Input and hidden layers
simple_mlp.add(Dense(px, input_dim=px, 
                     kernel_initializer='normal', 
                     activation='relu'))
# output layer
simple_mlp.add(Dense(K_classes, 
                     kernel_initializer='normal', 
                     activation='softmax'))
# Compile model
simple_mlp.compile(loss='categorical_crossentropy', 
                   optimizer='adam', 
                   metrics=['accuracy'])
# fit the model
simple_mlp_fit = simple_mlp.fit(X_train, labels_train, 
                                validation_data=(X_valid, labels_valid), 
                                epochs=30, 
                                batch_size=100, 
                                verbose=1)

# make predictions on test
simpleMLP_test_preds = simple_mlp.predict(X_test)

# predict as the class with the highest probability
simpleMLP_test_preds = np.argmax(simpleMLP_test_preds,axis = 1)

# put test set predictions into pandas series
simpleMLP_test_preds = pd.Series(simpleMLP_test_preds,name='label')

# add 'ImageId' column
simpleMLP_for_csv = pd.concat([pd.Series(X_test.index,name = 'ImageId'), 
                               simpleMLP_test_preds],axis = 1)

simpleMLP_for_csv.to_csv('output/csv/simple_mlp_keras.csv',index=False)
