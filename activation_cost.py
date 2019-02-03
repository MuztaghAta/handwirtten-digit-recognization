"""This script defines classes of several combinations of activation+cost
functions for neuron networks. The combinations of activation+cost functions
include:
1. sigmoid + quadratic (or mean square error)
2. sigmoid + cross entropy
3. sigmoid+softmax + log likelihood

When initializing a network, the activation+cost combination will be one of
the properties that needs to be determined.
"""
import numpy as np


def sigmoid(z):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    """Prime derivative of the sigmoid function"""
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):
    """Softmax function"""
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0)


class SigmoidQuadratic:

    @staticmethod
    def cost(a, y):
        """Quadratic cost_trd function
        """
        return 0.5 * np.linalg.norm(a-y)**2

    @staticmethod
    def activation_hidden(z):
        """Activation function for hidden layers"""
        return sigmoid(z)

    @staticmethod
    def activation_output(z):
        """Activation function for the output layer"""
        return sigmoid(z)

    @staticmethod
    def delta(a, y, z):
        """Error for the output layer where error is the derivative of the cost
        with respect to z = w*x+b. Using chain rule, it's a multiplication of
        the derivative of c, the cost, with respect to a, the activation, and
        that of a with respect to z.
        """
        return (a - y) * sigmoid_prime(z)

    @staticmethod
    def prime(z):
        """Prime derivatives with respect to z for hidden layers"""
        return sigmoid_prime(z)


class SigmoidCrossEntropy:

    @staticmethod
    def cost(a, y):
        """Given sigmoid activation function, a will never be 0 or 1."""
        return np.sum(np.nan_to_num(- y*np.log(a) - (1-y)*np.log(1-a)))

    @staticmethod
    def activation_hidden(z):
        """Activation function for hidden layers"""
        return sigmoid(z)

    @staticmethod
    def activation_output(z):
        """Activation function for the output layer"""
        return sigmoid(z)

    @staticmethod
    def delta(a, y, z):
        """Error for the output layer where error is the derivative of the cost
        with respect to z = w*x+b. Using chain rule, it's a multiplication of
        the derivative of c, the cost, with respect to a, the activation, and
        that of a with respect to z.

        Here the term containing z is cancelled for sigmoid activation function
        and cross entropy cost.
        """
        return a - y

    @staticmethod
    def prime(z):
        """Prime derivatives with respect to z for hidden layers"""
        return sigmoid_prime(z)


class SoftmaxLogLikelihood:

    @staticmethod
    def cost(a, y):
        """C = -ln(a_y), where a_y is the element in a with the same index of
        the element in y with value 1.
        """
        a_y = a[np.argmax(y)]
        return np.asscalar(-np.log(a_y))  # covert a size-1 array to a scalar
        # otherwise array cannot be serialized by Json

    @staticmethod
    def activation_hidden(z):
        """Activation function for hidden layers"""
        return sigmoid(z)

    @staticmethod
    def activation_output(z):
        """Activation function for the output layer"""
        return softmax(z)

    @staticmethod
    def delta(a, y, z):
        """Error for the output layer where error is the derivative of the cost
        with respect to z = w*x+b. Using chain rule, it's a multiplication of
        the derivative of c, the cost, with respect to a, the activation, and
        that of a with respect to z.

        Here the term containing z is cancelled for softmax activation function
        and log likelihood cost.
        """
        return a - y

    @staticmethod
    def prime(z):
        """Prime derivatives with respect to z for hidden layers"""
        return sigmoid_prime(z)
