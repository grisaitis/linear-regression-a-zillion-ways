import numpy as np


class OLSLoss:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x, self.y = x, y

    def __call__(self, beta: float):
        residuals = self.y - self.x @ beta
        rss = np.dot(residuals, residuals)
        n = self.y.shape[0]
        return rss / n
