import numpy as np


def dropout():
    pass


class BatchNorm1D:
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Initialize parameters (gamma and beta)
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Moving averages used during inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
