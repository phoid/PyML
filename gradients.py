import numpy as np


def calc_gradient(activations, delta, lr=0.001, clip=True, norm=1.0):
    """
    Calculates the gradient update for weights given input activations and layer error (delta).
    Returns: lr * gradient
    """
    # Ensure inputs are numpy arrays
    activations = np.array(activations)
    delta = np.array(delta)
    
    # Calculate gradient: outer product of delta and activations
    # resulting shape should match weights: (n_out, n_in)
    grad = np.outer(delta, activations)
    
    if clip:
        grad = clip_by_norm(grad, max_norm=norm)
    
    return grad * lr


def clip_by_norm(gradients, max_norm=1.0):
    """Clips gradiants based on the norm of the gradients."""
    norm = np.linalg.norm(gradients)
    # print(f"Norm: {norm}")
    if norm > max_norm:
        clipping_factor = max_norm / norm
        # print(f"Norm: {norm}, clip: {clipping_factor}")
        gradients = gradients * clipping_factor
    return gradients
