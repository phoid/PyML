import numpy as np


def relu(x):
    """
    Applies ReLU to a matrix:

    """
    return np.maximum(x, 0)


def relu_deriv(x):
    """derivitive of ReLU"""
    return (x > 0).astype(float)


def leaky_relu(x, negative_slope=0.01):
    """
    Applies leaky ReLU to a matrix:

    """
    return np.maximum(x, x * negative_slope)


def leaky_relu_deriv(x, negative_slope=0.01):
    """derivative of leaky ReLU"""
    dx = np.ones_like(x)
    dx[x < 0] = negative_slope
    return dx


def sigmoid(x):
    """
    Applies Sigmoid to a matrix:

    """
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    """derivative of Sigmoid"""
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x):
    """
    Applies softmax to a matrix:

    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
