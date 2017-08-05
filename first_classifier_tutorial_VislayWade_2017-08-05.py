# A first network using the keras library
# Rebecca Vislay Wade
# PREDICT 413 | Summer 2017

# This program classifies diabetes patients based on physiological
# measurements.

# This code adapted from 
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
# Add model layers
    # "Dense" indicates a fully connected network
    #
    # input_dim argument is equal to the number of predictors.
    #
    # Activation functions for neurons in the first 3 layers is the rectifier
    # function (see https://en.wikipedia.org/wiki/Rectifier_(neural_networks) 
    # for details). 
    #
    # Activation function for the final layer is the sigmoid function (ensures
    # our output is between 0 and 1).

model.add(Dense(12, input_dim=8, activation='relu')) # Input layer
model.add(Dense(8, activation='relu'))               # Hidden Layer 1
model.add(Dense(8, activation='relu'))               # Hidden Layer 2
model.add(Dense(1, activation='sigmoid'))            # Output layer

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
    # Lots and lots of matrix operations are performed. This step that can be 
    # distributed to graphics cards.
    # epochs - number of training iterations
    #
    # batch_size - number of samples per gradiant update

model.fit(inputs, TARGET, epochs=1000, batch_size=10)

# Training accuracy - how well does the model fit
scores = model.evaluate(inputs, TARGET)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

