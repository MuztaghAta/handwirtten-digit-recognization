"""This program plots the training results to help analyze the performance
(runtime, accuracy, cost, etc.) of the neural network model by using the
training results data. The analysis helps detect problems (overfitting,
overtraining, slow earning, etc.) and select the right network architecture
and hyper parameters.

It's trivial to define this method/function that can plot any dimension of
the results. However, this would compromise the flexibility of plotting
customization, since different plots are likely to have different settings
like plot scale (normal or log ...), legend location, label ticks, aspect
ratio, and so on.

Overfitting can be detected by comparing the prediction accuracies of the
trained model on the training data and on the test data. Larger gap between
the two indicates more severe overfitting.

Overtraining can be detected using linear curve fitting, i.e., if the last
part of the accuracy/cost curve can be fit to a straight line with zero or
negative slope then it can be said that overtraining is happening.
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def golden_ratio(ax):
    """Return a ratio so that the plot has a golden ratio!!!
    """
    y_min, y_max = ax.get_ylim()
    x_min, x_max = ax.get_xlim()
    return (x_max-x_min) / (y_max-y_min) / 1.618


def performance_plot(results, dim, net_label, show=True, fig_name=None):
    """Plot the results of chosen dimensions for all trained networks. The
    aspect ratio is set to golden ratio!!!

    dim - a list [dim1, dim2, ...] that contains the names of the interested
    dimensions, e.g. ['cost_trd', 'cost_evd'] can be plotted in the same
    figure.

    measure_label - a dictionary {dim1: label1, ...]. Each label shows the
    measure name/unit of a dimension, e.g. the label of 'accuracy_trd' is
    'Prediction accuracy'.

    net_label - a dictionary {net1: label1, ...]. Each label characterizes a
    network model.

    dim_label - a dictionary {dim1: label1, ...]. Each label tells more about
    a dimension than its measure label, e.g. the label of 'accuracy_trd' is
    'on training data' which means this dimension is 'Prediction accuracy on
    training data'.
    """
    measure_label = {'runtime': 'Runtime (s)',
                     'accuracy_trd': 'Prediction accuracy',
                     'cost_trd': 'Cost',
                     'accuracy_evd': 'Prediction accuracy',
                     'cost_evd': 'Cost'}
    dim_label = {'runtime': '',
                 'accuracy_trd': 'on training data',
                 'cost_trd': 'on training data',
                 'accuracy_evd': 'on evaluation data',
                 'cost_evd': 'on evaluation data'}
    sns.set()
    fig = plt.figure()
    # get the length of the dim(s) from a random-chosen network. If multiple
    # dimensions are interested, just use the first dimension since all
    # dimensions are supposed to have the same length given that they are
    # going to be plotted in the same figure.
    len_dim = len(results[random.choice(list(results.keys()))][dim[0]])
    x_m = len_dim + 1
    x = np.arange(1, x_m)
    # plot the dim results for all trained networks
    for net in results.keys():
        net_lb = net_label[net]
        num_dims = len(dim)
        for d in dim:
            if num_dims > 1:
                joint = ' for model with '
                # if there is results for only one network, no need to
                # show its label in the plot box
                if len(results.keys()) < 2:
                    joint, net_lb = '', ''
            else:
                joint = ''
            label = ''.join([dim_label[d], joint, net_lb])
            plt.plot(x, results[net][d], label=label)
    plt.xlim([0, x_m])
    plt.xlabel('Epochs')
    plt.ylabel(measure_label[dim[0]])
    if measure_label[dim[0]] == 'Runtime (s)':
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1), frameon=True)
    if measure_label[dim[0]] == 'Prediction accuracy':
        plt.legend(loc='lower right', bbox_to_anchor=(1, 0), frameon=True)
    if measure_label[dim[0]] == 'Cost':
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1), frameon=True)
    ax = plt.gca()
    ratio = golden_ratio(ax)
    ax.set_aspect(ratio)
    if fig_name is not None:
        fig.savefig(fig_name, bbox_inches='tight')
    if show:
        plt.show()
