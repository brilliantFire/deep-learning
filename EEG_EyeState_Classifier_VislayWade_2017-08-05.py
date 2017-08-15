# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 18:31:55 2017

"""
# A second network using the keras library - EEG Eye State data
# Rebecca Vislay Wade
# PREDICT 413 | Summer 2017

# This program builds a deep neural network model of eye state
# (eyes open or closed) from EEG signals recorded on 14 different scalp 
# electrodes. More info on this dataset can be found here:
# https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State#

# UPDATED 15 Aug 2017: Added data scaling and computation of test set
# confusion matrix and classification accuracy.

from keras.models import Sequential
from keras.layers import Dense
import numpy
from pandas import DataFrame, Series, crosstab
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# EEG Eye State dataset - (freely available at 
# https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff )
# Copy-paste data into text editor and save as "EEGeyebehavior.csv" in working
# directory.
# Response (column 14) coded 1 for eyes closed and 0 for eyes open
eeg=numpy.loadtxt("EEGeyebehavior.csv", delimiter = ",")

# convert numpy array to pandas Series to do train/test split and look at 
# actual values of TARGET
eegDF=DataFrame(eeg)

# Random number seed
numpy.random.seed(420)

# 70/30 training/test split
X_train, X_test, y_train, y_test = train_test_split(eegDF, eegDF.iloc[:,14], 
                                                    test_size=0.3)
# Drop the response variable from the X's
X_train = X_train.drop(14,1)
X_test = X_test.drop(14,1)

# Normalize predictors
X_trainNorm = preprocessing.scale(X_train)
X_testNorm = preprocessing.scale(X_test)

# Actual values for TARGET from EEG dataset (frequency table)
crosstab(index=eegDF.iloc[:,14], columns="count") 

# Checking the split (also frequency tables)
crosstab(index=y_train, columns="count")
crosstab(index=y_test, columns="count")

# Convert splits back to numpy arrays for Keras functions
X_trainArr = numpy.array(X_trainNorm)
y_trainArr = numpy.array(y_train)
X_testArr = numpy.array(X_testNorm)
y_testArr = numpy.array(y_test)

# Random number seed
numpy.random.seed(420)

# Create a sequential model
model = Sequential()
# Add model layers
    # "Dense" indicates a fully connected network
    #
    # input_dim - the number of predictors.
    #
    # Sigmoid activations for all neurons (could change in subsequent versions)

model.add(Dense(14, input_dim = 14, activation = 'sigmoid'))
model.add(Dense(20, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(10, activation = 'sigmoid'))
model.add(Dense(1, activation = 'sigmoid'))

# Compile model
    # This is where the tensor calculations are being done by the library of
    # fast numerical functions in TensorFlow (or Theano).
    #
    # Some things about the compile function:
        # loss argument - specifies the loss function used to check the 
        # weights; Here, it is the log-loss function ('binary_crossentropy'). 
        # This is the function the model is minimizing.
        #
        # optimizer argument - specifies the search algorithm for finding the 
        # best set of weights; 'adam' specifies an efficient default for
        # the method of finding the best set of weights
        #
        # metrics argument - the metric(s) reported after training; here, we're
        # reporting classification accuracy.
model.compile(loss='binary_crossentropy', 
              optimizer='adam', metrics=['accuracy'])

# Fit the model
    # Lots and lots of matrix operations are performed. This step can be 
    # distributed to graphics cards.
    # epochs - number of training iterations
    #
    # batch_size - number of samples per gradiant update
    #
    # validation_split - percentage of training samples to use for weight set
    # validation
model.fit(X_trainArr, y_trainArr, epochs=10000, batch_size=100)

# Training accuracy - how well does the model fit the training data?
scores = model.evaluate(X_trainArr, y_trainArr)
print("\nTraining Accuracy: %.2f%%" % (scores[1]*100))

# Test accuracy - how well does the model predict data it has not been trained
# on?
# First, calculate predictions and round them (0.5 decision rule)
preds = model.predict(X_testArr, batch_size = 10, verbose = 1)
roundPreds = Series([round(x[0]) for x in preds])

# Confusion matrix for predictions
confMat = crosstab(index=roundPreds, columns=y_test, margins = True)

# Test set classification accuracy
print("\nTest Accuracy: %.2f%%" % ((confMat.loc[0,0] + confMat.loc[1,1])/
                                       (confMat.loc["All","All"])*100))
