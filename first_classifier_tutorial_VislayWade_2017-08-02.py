# A first network using the keras library
# Rebecca Vislay Wade
# PREDICT 490 | Spring 2017

# This program classifies patients based on whether or not they exhibit signs of
# diabetes.

# This code largely adapted from 
# http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed 
numpy.random.seed(123)

# Load pima indians dataset (freely available at 
# http://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes )
# Copy-paste data into text editor and save as "pimaDiabetes.csv" in wd.
# All 768 patients are females at least 21 years old of Pima Indian heritage.
pima = numpy.loadtxt("pimaDiabetes.csv", delimiter=",")

# Isolate TARGET variable...
TARGET = pima[:,8]    # this is 0 if not diagnosed, 1 if diagnosed
# ...from the input variables...
inputs = pima[:,0:8]

# Create a sequential model
model = Sequential()
# Model layers
model.add(Dense(25, input_dim=25, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(5, activation='sigmoid'))




