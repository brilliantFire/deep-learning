"""
~*~* Simple Convolutional Neural Net (CNN) using Keras/Tensorflow *~*~

Code for a simple CNN with one Conv2D -> MaxPool as a classifier for the 
MNIST handwritten digit data in the context of Kaggle's 'Digit Recognizer' 
competition.
"""
import pandas as pd
import numpy as np
import os


from keras.utils.np_utils import to_categorical # for labels
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


from keras import backend as K
K.set_image_dim_ordering('th')

np.random.seed(237)

# Set the working directory
os.chdir('C:\\Users\\rlvis\\work_MSPA\\MNIST') # desktop

train_orig = pd.read_csv('data/train.csv')
X_test = pd.read_csv('data/test.csv')

# Hold out 4200 random images (10%) as a validation set
valid = train_orig.sample(n = 4200, random_state = 555)
train = train_orig.loc[~train_orig.index.isin(valid.index)]

# delete original train set
del train_orig

# separate images & labels
X_train = train.drop(['label'], axis=1)
labels_train = train['label']

X_valid = valid.drop(['label'], axis=1)
labels_valid = valid['label']

# clear more space
del train, valid

# Normalize and reshape
X_train = X_train.astype('float32') / 255.
X_train = X_train.values.reshape(X_train.shape[0], 1, 28, 28).astype('float32')

X_valid = X_valid.astype('float32') / 255.
X_valid = X_valid.values.reshape(X_valid.shape[0], 1, 28, 28).astype('float32')

X_test = X_test.astype('float32') / 255.
X_test = X_test.values.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# one hot encoding of digit labels
labels_train = to_categorical(labels_train)
labels_valid = to_categorical(labels_valid)

# K = 10 digits classes; 784 px images as input
K_classes = 10
px = X_train.shape[1]    # 784 pixels/inputs

convnet1 = Sequential()
 # Input layer is a convolution layer with 32 filters
convnet1.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
# Take the max over 2x2 pixel units
convnet1.add(MaxPooling2D(pool_size=(2, 2)))
# Exclude random 20% of nodes (regularization layer)
convnet1.add(Dropout(0.2))
# Go from 28x28 images to vector of 784 pixels
convnet1.add(Flatten())
# Have to flatten before going through a dense layer
convnet1.add(Dense(128, activation='relu'))
 # Output layer with 10 nodes for 10 classes
convnet1.add(Dense(K_classes, activation='softmax'))
# Compile model
convnet1.compile(loss='categorical_crossentropy', 
                 optimizer='adam', 
                 metrics=['accuracy'])

# Fit the model
convnet1_fit = convnet1.fit(X_train, labels_train, 
                            validation_data=(X_valid, labels_valid), 
                            epochs=30, 
                            batch_size=100, 
                            verbose=2)

# make predictions on test
convnet1_test_preds = convnet1.predict(X_test)

# predict as the class with highest probability
convnet1_test_preds = np.argmax(convnet1_test_preds,axis = 1)

# put predictions in pandas series
convnet1_test_preds = pd.Series(convnet1_test_preds,name='label')

# Add 'ImageId' column
convnet1_for_csv = pd.concat([pd.Series(range(1,28001),name = 'ImageId'), 
                              convnet1_test_preds],axis = 1)

# write dataframe to csv for submission
convnet1_for_csv.to_csv('..output/csv/convnet1_keras.csv',index=False)
