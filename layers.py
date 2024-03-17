import numpy as np


class Linear:
    """
    defines the weights matrix for an individual linear layer
    """

    def __init__(self, nIn, nOut, bias=True):
        self.weights = np.random.rand(nOut, nIn) - 0.5
        if bias:
            self.bias = np.random.rand(nOut) - 0.5
        else:
            bias = 0

    def forward(self, X):
        """Apply the linear Combination"""
        return (X).dot(self.weights.T)


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
