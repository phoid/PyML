import numpy as np
import weights


class Linear:
    """
    Define the weights matrix for an individual linear layer

    Able to perform a linear operation on input data
    """

    def __init__(self, nIn, nOut, bias=True):
        self.weights = weights.kaiming(nIn, nOut)
        if bias:
            self.bias = np.random.rand(nOut) - 0.5
        else:
            bias = 0

    def forward(self, X):
        """Apply the linear Combination"""
        if hasattr(self, 'bias'):
            return (X).dot(self.weights.T) + self.bias
        return (X).dot(self.weights.T)


class Conv2d:
    """
    define an m * n weights matrix (Kernel)
    """

    def __init__(self):
        self.weights = weights.kaiming(nIn, nOut)

    def forward():
        pass
