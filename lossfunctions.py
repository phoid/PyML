import numpy as np


def mse(Y, yHat):
    """
    Mean-squared error
    """
    return ((Y - yHat) ** 2) / len(yHat)


def mae():
    """
    Mean-absolute error
    """
    pass


def bce():
    """
    Binary cross-entropy
    """
    pass


def backprop(
    error,
):
    pass
