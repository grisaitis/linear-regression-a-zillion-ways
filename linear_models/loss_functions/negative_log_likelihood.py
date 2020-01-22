import numpy as np
import scipy.stats.norm as norm


class NegativeLogLikelihoodLoss:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x, self.y = x, y

    def __call__(self, beta: float, sigma: float):
        log_probs = norm.logpdf(x=self.y, loc=self.x @ beta, scale=sigma)
        return -np.sum(log_probs)
