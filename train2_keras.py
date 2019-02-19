"""This script provides an environment to train the neuron model for the
MNIST handwriting digit classification project.
"""
import numpy as np
import mnist_loader as loader
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.models import save_model
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam
from network2_keras import vanilla_model
from sklearn.model_selection import StratifiedKFold


# import data set #
# option 1: from local directory
path = './mnist.pkl.gz'
train_data, validation_data, test_data = loader.load_data(path)
x_train, y_train = train_data
x_val, y_val = validation_data
x_test, y_test = test_data
x_train = np.concatenate([x_train, x_val], axis=0)
y_train = np.concatenate([y_train, y_val], axis=0)
train_data = (x_train, y_train)
# option 2: download from keras dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# check data shape and whether data is normalized
sample = x_train[0]
# print('data shape: ', x_train.shape, y_train.shape)
# print(sample.shape)
# plt.imshow(sample.reshape(28, 28), cmap='Greys')
# plt.show()

# hyper parameters #
net_sizes = [100, 10]  # input layer not included
activations = ['sigmoid', 'sigmoid']  # sigmoid, softmax, relu
l2rs = [0., 0.]
opt = SGD(lr=0.1)
loss = 'sparse_categorical_crossentropy'  # sparse_
mb_size = 10  # mini-batch size
epochs = 30  # number of epochs to train

# train #
validation = 'classic'  # classic, skfold
# train the model with classic 80/20 validation
if validation == 'classic':
    # create a vanilla model #
    model_v = vanilla_model(sample, net_sizes, activations, l2rs, opt, loss)
    history = model_v.fit(x_train, y_train, batch_size=mb_size, epochs=epochs,
                          verbose=1, validation_split=0.18, shuffle=True)
    results = history.history
    print(results)
# train the model with stratified k-fold cross validation
elif validation == 'skfold':
    cv_scores = []
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    for tr, val in kf.split(x_train, y_train):
        x_tr, x_val = x_train[tr], x_train[val]
        y_tr, y_val = y_train[tr], y_train[val]
        # y_tr = to_categorical(y_train[tr])
        # y_val = to_categorical(y_train[val])
        # check data shape
        print('input data shape: ', x_tr.shape, x_val.shape)
        print('label data shape: ', y_tr.shape, y_val.shape)
        # build network model
        model_v = vanilla_model(sample, net_sizes, activations, l2rs, opt, loss)
        # train the model
        history = model_v.fit(x_tr, y_tr, batch_size=mb_size, epochs=epochs,
                              verbose=1, validation_data=(x_val, y_val))
        val_results = model_v.evaluate(x_val, y_val)
        cv_scores.append(val_results[1])
    # print cross evaluation results
    print(cv_scores, np.mean(cv_scores), np.std(cv_scores))

# finally train the model if cross validation passes #
print('input data shape: ', x_train.shape)
print('label data shape: ', y_train.shape)
# y_train = to_categorical(y_train)
f_model = vanilla_model(sample, net_sizes, activations, l2rs, opt, loss)
f_model.fit(x_train, y_train, batch_size=10, epochs=30, verbose=1)
# final test the trained model using test data #
# y_test = to_categorical(y_test)
f_results = f_model.evaluate(x_test, y_test)
print(f_results)
# save trained model #
save_model(f_model, 'vanilla_model.h5')
