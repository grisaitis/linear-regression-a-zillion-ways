from linear_models.loss_functions import OLSLoss


class OLSOptimizedEstimator:
    def fit(self, x, y, optimizer):
        loss = OLSLoss(x, y)
        beta_hat = optimizer.minimize(loss)
        return beta_hat
