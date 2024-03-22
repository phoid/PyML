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

    def forward(self, X, mode="train"):
        if mode == "train":
            # Step 1: Calculate mean and variance across the batch dimension
            batch_mean = np.mean(X, axis=0)
            batch_var = np.var(X, axis=0)

            # Step 2: Normalize
            X_normalized = (X - batch_mean) / np.sqrt(batch_var + self.eps)

            # Step 3: Scale and shift (affine transformation)
            out = self.gamma * X_normalized + self.beta

            # Step 4: Update running mean and variance for inference
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * batch_var
            )

            return out

        elif mode == "inference":
            # Use running mean and variance
            X_normalized = (X - self.running_mean) / np.sqrt(
                self.running_var + self.eps
            )
            out = self.gamma * X_normalized + self.beta
            return out

        else:
            raise ValueError("Invalid mode. Choose 'train' or 'inference'")
