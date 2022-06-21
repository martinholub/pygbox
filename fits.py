import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm

class Model(object):
    def __init__(self, fun):
        self.fun = fun
    def fit(self, x):
        params = self.fun.fit(x)
        print("Params of fit: {}".format(params))
        return params
    def plot(self, ax, params):
        xlow, xhigh = ax.get_xlim()
        x = np.linspace(xlow, xhigh, 100)
        y = self.fun.pdf(x, *params)
        ax.plot(x, y, alpha = .5)

class NormDist(Model):
    def __init__(self):
        super(NormDist, self).__init__(fun = norm)

class LogNormDist(Model):
    def __init__(self):
        super(LogNormDist, self).__init__(fun = lognorm)
