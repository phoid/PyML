import numpy as np


class Linear:
    """
    defines the weights matrix for an individual linear layer
    """

    def __init__(self, nIn, nOut):
        self.weights = np.random.rand(nOut, nIn) - 0.5

    def backprop(self):
        pass

    def forward(self, X):
        """Apply the linear Combination"""
        return X.T * self.weights


class Conv2d:
    """
    define an m * n weights matrix (Kernel)
    """

    def __init__(self):
        pass

    def backprop():
        pass

    def forward():
        pass
