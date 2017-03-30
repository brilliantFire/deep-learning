# Rebecca Vislay Wade
# PREDICT 490 | Spring 2017 | Northwestern MSPA Program
# Created on 29 March 2017
 
# Simple Multilayer Perceptron
# Code largely modified from Github user @iamtrask (original code can be found
# here: http://iamtrask.github.io/2015/07/12/basic-python-network/
###############################################################################

import numpy as np

def nonlinWeightingFunc(x,deriv=False):
    if(deriv==True):
        return x*(1-x)             # Derivative of the transfer function
    return 1/(1+np.exp(-x))        # Sigmoid transfer function
    
inputs = np.array([[0,0,1],        # CHANGE LATER to user input??
                   [0,1,0],
                   [1,0,0],
                   [1,1,1]])
              
# Added user input for output pattern:
desiredOutput = np.array([int(num) for num in input("Enter a string (\"\") of four (4) 1s and 0s, separated by spaces: ").split()]).reshape(4,1)
print desiredOutput


np.random.seed(123)    # So we all have the same random numbers

# Initialize synaptic weights with random values
synapse0 = 1 - 2*np.random.random((3,4))        # inputLayer -> hiddenLayer
synapse1 = 1 - 2*np.random.random((4,1))        # hiddenLayer -> outputLayer

for trainingRun in xrange(10000):
    if (trainingRun% 1000) == 0:
        print "Training Run # %s " %trainingRun
        
    # FEEDFORWARD: inputLayer -> hiddenLayer -> outputLayer 
    inputLayer = inputs
    hiddenLayer = nonlinWeightingFunc(np.dot(inputLayer,synapse0))
    outputLayer = nonlinWeightingFunc(np.dot(hiddenLayer,synapse1))
    
    if (trainingRun% 1000) == 0:
        print "Output Guess: \n" + str(outputLayer)

    # How close did the outputLayer get to the desiredOutput?
    outputLayer_error = desiredOutput - outputLayer
    
    if (trainingRun% 1000) == 0:
        print "Output Layer Error: " + str(np.mean(np.abs(outputLayer_error)))
        
    # By how much should we change the weights on synapse1 (hiddenLayer -> outputLayer)?
    outputLayer_delta = outputLayer_error * nonlinWeightingFunc(outputLayer,deriv=True)

    # How much did each hiddenLayer value contribute to outputLayer error (according to the weights)?
    hiddenLayer_error = outputLayer_delta.dot(synapse1.T)
    
    if (trainingRun% 1000) == 0:
        print "Hidden Layer Error: " + str(np.mean(np.abs(hiddenLayer_error))) + "\n"
    
    # By how much should we change the weights on synapse0 (inputLayer -> outputLayer)?
    hiddenLayer_delta = hiddenLayer_error * nonlinWeightingFunc(hiddenLayer,deriv=True)
    
    # Update synaptic weights
    synapse1 += hiddenLayer.T.dot(outputLayer_delta)    # Update weights on synapse1 (hiddenLayer -> outputLayer)
    synapse0 += inputLayer.T.dot(hiddenLayer_delta)     # Update weights on synapse0 (inputLayer -> hiddenLayer)
    
print "\nOutput After %s Training Runs:" %trainingRun
print outputLayer
