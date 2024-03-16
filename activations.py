import numpy as np


def relu(x):
    """
    Applies ReLU to a matrix:

    """
    return np.maximum(x, 0)


def leaky_relu(x, negative_slope=0.01):
    """
    Applies leaky ReLU to a matrix:

    """
    return np.maximum(x, x * negative_slope)


def sigmoid():
    """
    Applies Sigmoid to a matrix:

    """
    pass


def softmax():
    """
    Applies softmax to a matrix:


    """
    pass
