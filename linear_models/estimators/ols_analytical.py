import numpy as np


class OLSAnalyticalEstimator:
    def fit(self, x, y):
        return np.linalg.inv(x.T @ x) @ x.T @ y
