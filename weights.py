import numpy as np


def kaiming(fan_in, fan_out):
    """Initializes weights with Kaiming He (uniform) distribution."""
    std = np.sqrt(2.0 / fan_in)  # He uniform variance scaling factor
    return np.random.uniform(-std, std, size=(fan_out, fan_in))


def xavier(fan_in, fan_out):
    """Initializes weights with Xavier (Glorot) uniform distribution."""
    std = np.sqrt(6.0 / (fan_in + fan_out))  # Xavier uniform variance scaling factor
    return np.random.uniform(-std, std, size=(fan_out, fan_in))


def uniform(nOut, nIn):
    """Initialized weights with a standard uniform distribution"""
    return np.random.rand(nOut, nIn) - 0.5
