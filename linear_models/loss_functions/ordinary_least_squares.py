import numpy as np


class OLSLoss(object):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __call__(self, beta):
        residuals = self.y - self.x @ beta
        rss = np.dot(residuals, residuals)
        n = self.y.shape[0]
        return rss / n
