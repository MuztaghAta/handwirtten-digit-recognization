"""This script defines a convolutional neural network for the project of
handwritten digits recognition using Keras.
"""
from keras import Sequential
from keras.optimizers import SGD, Adam
from keras.initializers import random_normal
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten


def cnn_model():
    """A convolutional network model"""
    model = Sequential()
    model.add(Conv2D(32, 5, activation='sigmoid', input_shape=(28, 28, 1)))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, 5, activation='sigmoid'))  # , input_shape=(12, 12, 20)
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))
    # compile setting
    model.compile(optimizer=SGD(lr=0.1), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )  # sparse_
    return model

