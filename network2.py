""" This script defines an advanced feed-forward network class.

Like the baseline network class defined in network.py, this network class
also uses mini-batch stochastic gradient descent (SGD) to learn optimal
weights and biases. However, this network class provides a number of features
that the baseline network class doesn't have, which include:
1. matrix-based approach to back propagation, which can computes the gradients
   for all samples in a mini-batch simultaneously. This approach is faster than
   vector-based approach which can only computes the gradients for one samples
   at the same time.
2. network initialization with different activation+cost functions, including
   sigmoid + quadratic, sigmoid + cross entropy and sigmoid/softmax + log
   likelihood.
3. monitor runtime, prediction accuracy and cost.
4. L2 regularization and dropout (neurons of hidden layers).
5. different weights/biases initialization approaches.
6. save and load trained network model.

Note that some of the above features are not implemented in the codes for the
NNDL book including 2, runtime monitoring in 3 and dropout in 4. A Further note
on dropout: it has been especially useful technique to reduce overfitting in
training large, deep networks, where the problem of overfitting is often acute.
However, for this MNIST recognition model with a typical network architecture
of [784, 30, 10], dropout half hidden neurons seriously deteriorates the
prediction accuracy of the model maybe due to that the network is not deep or
large enough. This might be the reason for not including  dropout feature in
the code for the NNDL book. With that said, implementing dropout is a nice way
to understand it.
"""
import activation_cost as ac
import numpy as np
import random
import time
import json
import copy


class Network:

    def __init__(self, sizes, activation_cost, initialize_wb=True):
        """ A network can be initialized with a size vector that specifies the
        num of neurons in each layer, activation and cost functions used, and
        randomly set weights and biases.

        If loading a network saved in data file, initialize_wb should set to
        False because the data includes trained weights and biases.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights, self.biases = [], []
        if initialize_wb:
            self.weights, self.biases = self.default_weights_initializer()
        self.activation_cost = activation_cost

    def default_weights_initializer(self):
        """Initialize each weight using a normal distribution (0, 1/square(n))
        where n is the number of weights (or connections, neurons) connecting
        to the same neuron, and initialize each bias using standard normal
        distribution (0, 1). The purpose of small initialized weights is to
        avoid neuron saturation which slow down training process.

        Setting the same random seed allows all created networks to get the
        same initial weights and biases, which is necessary when comparing
        the performance of different networks with different architecture or
        hyper parameters.
        """
        np.random.seed(1)
        # the first layer (i.e. input layer) does not have biases
        biases = [np.random.randn(num, 1) for num in self.sizes[1:]]
        # for each matrix in weights: row - layer to, column - layer from
        weights = [np.random.randn(x, y) / np.sqrt(y)
                   for x, y in zip(self.sizes[1:], self.sizes[:-1])]
        return weights, biases

    def large_weights_initializer(self):
        """Initialize each weight/bias using standard normal distribution
        (0, 1).
        """
        np.random.seed(1)
        self.biases = [np.random.randn(num, 1) for num in self.sizes[1:]]
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(self.sizes[1:], self.sizes[:-1])]

    def learning_sgd(self, train_data, mini_batch_size, epochs, eta, l2lbd=0,
                     dropout=0,
                     monitor_runtime=False,
                     monitor_train_accuracy=False,
                     monitor_train_cost=False,
                     evaluation_data=None,
                     monitor_evaluation_accuracy=False,
                     monitor_evaluation_cost=False):
        """Learning using mini-batch stochastic gradient descent:
        1. divide shuffled training data into several mini batches
        2. for each mini batch, update weights/biases using gradient descent
        3. once all mini batches looped, the train of this epoch is finished
        4. evaluation the trained model using test or validation data set and
        monitor training results including accuracy, cost and runtime for each
        epoch.
        """
        n_train = len(train_data)  # number of entries in the training data
        runtime = []  # runtime for training data
        cost_trd = []  # cost for training data
        accuracy_trd = []  # accuracy for training data
        cost_evd = []  # cost for evaluation data
        accuracy_evd = []  # accuracy for evaluation data
        for ep in np.arange(1, epochs+1):
            # divide the training data into a number of mini batches
            mini_batches = [train_data[k:k+mini_batch_size]
                            for k in np.arange(0, n_train, mini_batch_size)]
            start_time = time.time()  # used to monitor runtime for each epoch
            for mini_batch in mini_batches:
                self.update_weight_bias(mini_batch, eta, n_train, l2lbd,
                                        dropout)
            if monitor_runtime:
                runtime.append(time.time() - start_time)
            if monitor_train_accuracy:
                accuracy_trd.append(
                    self.test(train_data, convert=True) / n_train)
            if monitor_train_cost:
                cost_trd.append(self.total_cost(train_data, l2lbd))
            if monitor_evaluation_accuracy:
                n_test = len(evaluation_data)
                n_correct = self.test(evaluation_data)
                accuracy = n_correct / n_test
                accuracy_evd.append(accuracy)
                print('Epoch {0}: {1}/{2}, {3}'
                      .format(ep, n_correct, n_test, accuracy))
            else:
                print('Epoch {} is complete'.format(ep))
            if monitor_evaluation_cost:
                cost_evd.append(
                    self.total_cost(evaluation_data, l2lbd, convert=True))
        results = {'runtime': runtime,
                   'accuracy_trd': accuracy_trd, 'cost_trd': cost_trd,
                   'accuracy_evd': accuracy_evd, 'cost_evd': cost_evd}
        return results

    def update_weight_bias(self, mini_batch, eta, n_train, l2lbd, dropout):
        """Compute average gradients of weights and biases, and update weights
        and biases for a mini batch using gradient descent.
        """
        # dropout neurons before training, the weights and biases of the
        # dropout neurons don't change after training with this min batch
        hidden_sizes, drop_lists = [], []
        w_dropped = [np.zeros((x, y))
                     for x, y in zip(self.sizes[1:], self.sizes[:-1])]
        if dropout > 0 and self.num_layers > 2:  # only drop hidden neurons
            hidden_sizes = self.sizes[1:-1]  # sizes of hidden layers
            drop_lists = [random.sample(range(0, s), int(s * dropout))
                          for s in hidden_sizes]
            w_dropped = self.dropout_neurons(copy.copy(self.weights), None,
                                             hidden_sizes, drop_lists)
        sum_nabla_w, sum_nabla_b = self.backprop(mini_batch, dropout,
                                                 hidden_sizes, drop_lists)
        mb_size = len(mini_batch)  # number of entries in the mini-batch
        nabla_w = [(eta/mb_size)*s for s in sum_nabla_w]
        nabla_b = [(eta/mb_size)*s for s in sum_nabla_b]
        # update weights and biases by original values minus gradient and l2
        # regularization, and plus back the original values of the weights of
        # the dropout neurons which were deducted in previous operation
        self.weights = [(1-l2lbd*eta/n_train)*w - n_w + w_d for w, n_w, w_d
                        in zip(self.weights, nabla_w, w_dropped)]
        self.biases = [b-n_b for b, n_b in zip(self.biases, nabla_b)]

    def backprop(self, mini_batch, dropout, hidden_sizes, drop_lists):
        """Given a mini-batch, compute gradients of all weighs and biases for
        all samples in the mini-batch at the same time, and return the sum of
        gradients of the same weights and biases.
        """
        sum_nabla_w = [np.zeros(w.shape) for w in self.weights]
        sum_nabla_b = [np.zeros(b.shape) for b in self.biases]
        # put all inputs in the mini-batch into a (s1, m)-array where s1 is
        # the number of elements in each input and m is the number of
        # samples in the mini-batch
        inputs = np.concatenate([x for x, y in mini_batch], axis=1)
        # put all results in the mini-batch into a (s2, m)-array where s2 is
        # the number of elements in each result
        results = np.concatenate([y for x, y in mini_batch], axis=1)
        # compute activations of all layers for all inputs and put in a list
        # [a1, a2, ...] where ai is a (ni, m)-array of the activations for the
        # ith layer where ni is the number of neurons in the ith layer
        activation = inputs
        zs = []
        activations = [inputs]
        if dropout > 0 and self.num_layers > 2:  # only drop hidden neurons
            w_temp = copy.copy(self.weights)
            b_temp = copy.copy(self.biases)
            w_temp, b_temp = self.dropout_neurons(
                w_temp, b_temp, hidden_sizes, drop_lists)
        else:
            w_temp = self.weights
            b_temp = self.biases
        for i, (w, b) in enumerate(zip(w_temp, b_temp)):
            z = np.dot(w, activation) + b  # b is broadcasted
            zs.append(z)
            if i == len(self.weights) - 1:
                activation = self.activation_cost.activation_output(z)
            else:
                activation = self.activation_cost.activation_hidden(z)
            activations.append(activation)
        # compute errors for the output layer for samples in the min batch
        errors = self.activation_cost.delta(activation, results, zs[-1])
        sum_errors = np.sum(errors, axis=1)  # array of shape (n,)
        sum_errors = sum_errors[:, np.newaxis]  # reshape to (n, 1)
        sum_nabla_b[-1] = sum_errors
        sum_nabla_w[-1] = np.dot(errors, activations[-2].transpose())
        for l in np.arange(2, self.num_layers):
            # propagate errors back to previous layers except the input layer
            errors = np.dot(w_temp[-l+1].transpose(), errors) * \
                     self.activation_cost.prime(zs[-l])
            sum_errors = np.sum(errors, axis=1)
            sum_errors = sum_errors[:, np.newaxis]
            sum_nabla_b[-l] = sum_errors
            sum_nabla_w[-l] = np.dot(errors, activations[-l-1].transpose())
        if dropout > 0 and self.num_layers > 2:
            # set the nabla for the dropped neurons to zero so that their
            # weights and biases keep unchanged in the training with this
            # mini batch
            sum_nabla_w, sum_nabla_b = self.dropout_neurons(
                sum_nabla_w, sum_nabla_b, hidden_sizes, drop_lists)
        return sum_nabla_w, sum_nabla_b

    def dropout_neurons(self, w, b, hidden_sizes, drop_lists):
        """If both weights and biases are passed, dropout selected neurons by
        setting the weights connected to/from a neuron to be dropped and
        biases of this neuron to zero.

        If nabla of weighs and biases are passed, this method will set the
        nabla of weighs and biases of the dropped neurons to zero.

        If only w is passed and b is None, this method set the weights related
        to un-dropout neurons to zero and only keeps the weights for dropout
        neurons.
        """
        if b is not None:
            for i in range(0, len(hidden_sizes)):
                for j in drop_lists[i]:
                    w[i][j:] = 0 * w[i][j:]
                    b[i][j] = 0 * b[i][j]
                    w[i+1][:j] = 0 * w[i+1][:j]
            return w, b
        else:
            for i in range(0, len(hidden_sizes)):
                for j in drop_lists[i]:
                    w[i][j:] = 0 * w[i][j:]
                    w[i+1][:j] = 0 * w[i+1][:j]
            w = [w1 - w2 for w1, w2 in zip(self.weights, w)]
            return w

    def test(self, data, convert=False):
        """Evaluate the prediction accuracy of the trained network model.
        'convert' should be False for test and validation data, and True for
        training data.

        Here each y in the test/validation data is only a digit, different
        from y in the training data, which is a multidimensional array. If
        cross validation is needed, we may have to keep the format consistent.
        """
        if convert:
            test_results = [(np.argmax(self.feed_forward(x)), np.argmax(y))
                            for (x, y) in data]
        else:
            test_results = [(np.argmax(self.feed_forward(x)), y)
                            for (x, y) in data]
        n_correct = sum(int(x == y) for (x, y) in test_results)
        return n_correct

    def feed_forward(self, input_):
        """Given an input, compute network output (or activation of the output
        layer).

        Note that output is not necessarily the final prediction. For example,
        if we use softmax for digits classification, we need to transform the
        softmax output which is a probability distribution to a single value
        which is the final prediction.
        """
        activation = input_
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, activation) + b
            if i == len(self.weights) - 1:
                activation = self.activation_cost.activation_output(z)
            else:
                activation = self.activation_cost.activation_hidden(z)
            # activation changes iteratively from input layer to hidden layers
            # and finally to output layer
        return activation

    def total_cost(self, data, l2lbd, convert=False):
        """Compute total cost for the input data. 'convert' flag should be
        False for training data and True for test and validation data where
        y have not been vectorized.
        """
        cost = 0.0
        for x, y in data:
            a = self.feed_forward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.activation_cost.cost(a, y)
        cost = cost / len(data)
        # plus the regularization term
        cost += 0.5 * (l2lbd/len(data)) * sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save_net(self, filename):
        """Save trained neuron network to file. Json cannot serialize ndarray,
        so all ndarrays have to be converted to lists.
        """
        weights = [w.tolist() for w in self.weights]
        biases = [b.tolist() for b in self.biases]
        net_data = {'sizes': self.sizes, 'weights': weights, 'biases': biases,
                    'activation_cost': str(self.activation_cost.__name__)}
        with open(filename, 'w') as write_out:
            json.dump(net_data, write_out)


def load_net(filename):
    """Load saved neuron network from file."""
    with open(filename, 'r') as read_in:
        net_data = json.load(read_in)
    cost_activation = getattr(ac, net_data['activation_cost'])
    net = Network(net_data['sizes'], cost_activation, initialize_wb=False)
    net.weights = [np.array(w) for w in net_data['weights']]
    net.biases = [np.array(b) for b in net_data['biases']]
    return net


def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function"""
    return sigmoid(z) * (1 - sigmoid(z))


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth position and
    zeroes elsewhere. This is used to convert a digit in (0...9) into a
    corresponding desired output from the neural network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
