""" This script defines a simple neuron network class.

The network is a feed-forward network with sigmoid neurons. It uses mini-batch
stochastic gradient descent (SGD) to learn optimal weights and biases.
"""


import numpy as np


class Network:

    def __init__(self, sizes):
        """sizes is a vector that specifies the num of neurons in each layer"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        # the first layer (i.e. input layer) does not have biases
        self.biases = [np.random.randn(num, 1) for num in sizes[1:]]
        # for each matrix in weights: row - layer to, column - layer from
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(sizes[1:], sizes[:-1])]

    def sgd(self, train_data, mini_batch_size, epochs, eta, test_data=None):
        """Mini-batch stochastic gradient descent:
        1. shuffle training data set and divide into several mini batches
        2. for each mini batch, update weights and biases once
        3. once all mini batches looped, the train of this epoch is finished
        4. test the trained model using test data set
        """
        n = len(train_data)  # number of entries in the training data
        for ep in np.arange(1, epochs+1):
            np.random.shuffle(train_data)
            mini_batches = [train_data[k:k+mini_batch_size]
                            for k in np.arange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_weight_bias(mini_batch, eta)
            if test_data:
                n_test = len(test_data)
                n_correct = self.test(test_data)
                print('Epoch {0}: {1}/{2}, {3}'
                      .format(ep, n_correct, n_test, n_correct/n_test))
            else:
                print('Epoch {} is complete'.format(ep))

    def update_weight_bias(self, mini_batch, eta):
        """Compute average gradients of weights and biases, and update weights
        and biases for a mini batch
        """
        sum_nabla_w = [np.zeros(w.shape) for w in self.weights]
        sum_nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            nabla_w, nabla_b = self.back_propagation(x, y)
            sum_nabla_w = [s+n_w for s, n_w in zip(sum_nabla_w, nabla_w)]
            sum_nabla_b = [s+n_b for s, n_b in zip(sum_nabla_b, nabla_b)]
        mini_batch_size = len(mini_batch)
        nabla_w = [(eta/mini_batch_size)*s for s in sum_nabla_w]
        nabla_b = [(eta/mini_batch_size)*s for s in sum_nabla_b]
        self.weights = [w-n_w for w, n_w in zip(self.weights, nabla_w)]
        self.biases = [b-n_b for b, n_b in zip(self.biases, nabla_b)]

    def back_propagation(self, input_, output_):
        """Given an example (input and its correct output), compute gradients
        of weights and biases of each layer
        """
        # nabla (or del) denotes ∇ the gradient
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        zs = []
        activation = input_
        activations = [input_]  # a list to store activation of every layer
        # (including the input of the input layer and predicted output of
        # the output layer)
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # compute the gradients of weights and biases from the output layer
        # back to the input layer, using chain rule to compute the derivative
        # of composite functions
        sp = sigmoid_prime(zs[-1])
        prediction = activations[-1]
        nabla_z = (prediction - output_) * sp
        nabla_b[-1] = nabla_z
        nabla_w[-1] = np.dot(nabla_z, activations[-2].transpose())
        for l in np.arange(2, self.num_layers):
            sp = sigmoid_prime(zs[-l])
            nabla_z = np.dot(self.weights[-l+1].transpose(), nabla_z) * sp
            nabla_b[-l] = nabla_z
            nabla_w[-l] = np.dot(nabla_z, activations[-l-1].transpose())
        return nabla_w, nabla_b

    def test(self, test_data):
        """Evaluate the prediction accuracy of the trained NN model. Here each
        y in the test data is only a digit, different from y in the training
        data, which is a multidimensional array. If cross validation is needed,
        we may have to keep the format consistent.
        """
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        n_correct = sum(int(x == y) for (x, y) in test_results)
        return n_correct

    def feed_forward(self, input_):
        """Given an input, compute network output (or activation of the output
        layer). Note that output is not necessarily the final prediction.
        """
        activation = input_
        for w, b in zip(self.weights, self.biases):
            activation = sigmoid(np.dot(w, activation) + b)
            # activation changes iteratively from input layer to hidden layers
            # and finally to output layer
        return activation


def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function"""
    return sigmoid(z) * (1 - sigmoid(z))
