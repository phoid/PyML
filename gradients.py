import numpy as np


def stochastic(input, cost, weights=None, lr=0.001):
    """
    Calculates a Stocastic approximation of the Gradient
    """
    if weights is not None:
        return lr * input.dot(weights.T) * cost
    else:
        return lr * input * cost
