import numpy as np


def mse(Y, yHat):
    """
    Mean-squared error
    """
    return np.mean(((Y - yHat) ** 2))


def mse_prime(Y, yHat):
    """
    Derivative of Mean-squared error
    """
    return 2 * (yHat - Y) / Y.size



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
