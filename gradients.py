import numpy as np


def stochastic(input, cost, weights=None, lr=0.001, clip=True, norm=1.0):
    """
    Calculates a Stocastic approximation of the Gradient
    """
    if weights is not None:
        grad = lr * input.dot(np.array([weights]).T) * cost
    else:
        grad = lr * input * cost
    if clip:
        return clip_by_norm(grad, max_norm=norm)
    return grad


def clip_by_norm(gradients, max_norm=1.0):
    """Clips gradiants based on the norm of the gradients."""
    norm = np.linalg.norm(gradients)
    print(f"Norm: {norm}")
    if norm > max_norm:
        clipping_factor = max_norm / norm
        print(f"Norm: {norm}, clip: {clipping_factor}")
        gradients = gradients * clipping_factor
    return gradients
