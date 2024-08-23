# Utilities to pollute synthetic data with noise
from numpy.random import default_rng
import numpy as np

try:
    from pygbox import ops
except ImportError as e:
    import ops

rn_gen = default_rng()

def add_poisson(data, rate):
    """Adds Poission distributed values to data"""
    data = ops.normalize(data, 0, 1)
    p_noise = rn_gen.poisson(rate*data)
    data = data + p_noise
    data = ops.normalize(data, 0, 1)
    return data

def add_gauss(data, sigma):
    """Add mormally distributed values to data"""
    data = ops.normalize(data, 0, 1)
    g_noise = rn_gen.standard_normal(data.shape) * sigma
    data = data + g_noise
    data[data<0] = 0
    data = ops.normalize(data, 0, 1)
    return data
