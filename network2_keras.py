"""Define a fully-connected vanilla feedforward neural network using Keras.
"""
from keras import Sequential
from keras.initializers import random_normal
from keras.regularizers import l2
from keras.layers import Dense


def vanilla_model(sample, net_sizes, activations, l2rs, opt, loss):
    """A vanilla fully connected feedforward network.

    net_sizes - a list of scalar indicating the size of each Dense layer
    activations - a list of string indicating the activation function for each
        Dense layer
    l2rs - a list of scalar indicating l2 rate for each Dense layer
    sample - an input sample
    opt - optimizer
    loss - loss function
    """
    model = Sequential()
    seed = net_sizes[:-1]
    seed.insert(0, len(sample))
    for ns, ac, s, r in zip(net_sizes, activations, seed, l2rs):
        model.add(Dense(ns, input_shape=sample.shape, activation=ac,
                  kernel_initializer=random_normal(0, 1 / s ** 0.5),
                  kernel_regularizer=l2(r)))
    # specify learning algorithm and cost/loss function ...
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    return model
