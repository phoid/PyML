import numpy as np
import weights


class Linear:
    """
    defines the weights matrix for an individual linear layer
    """

    def __init__(self, nIn, nOut, bias=True):
        self.weights = weights.kaiming(nIn, nOut)
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

    def forward():
        pass
