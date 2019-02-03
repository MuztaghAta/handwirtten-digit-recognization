"""This script provides an environment to train the neuron network for the
MNIST handwriting digit classification project.

To train a network, you need to:
1. load the MNIST data set
2. shuffle the training data set
3. set hyper parameters and performance monitor
4. initiate and train network with setting in 3
5. save the performance results and trained network for future use

Depends on which parameter concerns, change the training code accordingly.
"""
import mnist_loader as loader
import activation_cost as ac
import analysis2 as a2
import network2
import numpy as np
import json
import os


# import data set
path = './mnist.pkl.gz'
train_data, validation_data, test_data = loader.load_data_wrapper(path)
# shuffle data before use for training
np.random.shuffle(train_data)

# hyper parameters
train_data = train_data[1:1000]  # number of data to use for training
net_size = [784, 30, 10]  # network size
mb_size = 10  # mini-batch size
epochs = 30  # number of epochs to train
eta = 1  # learning rate
l2lbd = 0  # penalty for L2 regularization
dropout = 0  # the proportion of neurons to be dropped in hidden layers

# monitor and evaluation setting
monitor_runtime = False  # False or True
monitor_train_accuracy = True
monitor_train_cost = True
# evaluation
evaluation_data = test_data  # test_data, validation_data, or None
monitor_evaluation_accuracy = True
monitor_evaluation_cost = True
# create a tuple which is convenient to pass to learning methods
monitor = (monitor_runtime, monitor_train_accuracy, monitor_train_cost,
           evaluation_data, monitor_evaluation_accuracy,
           monitor_evaluation_cost)

# create neuron network with a size and an activation_cost combination
net = network2.Network(net_size, ac.SigmoidCrossEntropy)

# train with a purpose:
results_net = net.learning_sgd(
    train_data, mb_size, epochs, eta, l2lbd, dropout, *monitor)

# create a label for each network to summarized its main property
net_label = {'net': 'learning rate 1'}

print('Training finished')

# save all training results to json file
results = {'net': results_net}  # keys have to be consistent with net_label
results_filename = 'results2.json'
results_file_path = os.getcwd() + os.sep + results_filename
with open(results_file_path, 'w') as data_out:
    json.dump(results, data_out)

# plot training results for a visual comparison
# specify which dimension(s) is interested, full list:
# ['runtime', 'accuracy_trd', 'accuracy_evd', 'cost_trd', 'cost_evd']
dimension1 = ['accuracy_trd', 'accuracy_evd']
# 'show' should set to True only in last call of the performance_plot method
# so that all the figures can show up
a2.performance_plot(results, dimension1, net_label, show=False, fig_name=None)
dimension2 = ['cost_trd', 'cost_evd']
a2.performance_plot(results, dimension2, net_label, show=True, fig_name=None)
