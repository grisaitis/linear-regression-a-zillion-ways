from linear_models.loss_functions import NegativeLogLikelihoodLoss


class NegativeLogLikelihoodOptimizedEstimator:
    def fit(self, x, y, optimizer):
        loss = NegativeLogLikelihoodLoss(x, y)
        beta_hat, sigma_hat = optimizer.minimize(loss)
        return beta_hat, sigma_hat
