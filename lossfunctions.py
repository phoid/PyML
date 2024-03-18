import numpy as np


def mse(Y, yHat):
    """
    Mean-squared error
    """
    return np.mean(((Y - yHat) ** 2))


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
