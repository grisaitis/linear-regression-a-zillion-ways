import numpy as np


class FakeData(object):
    def __init__(self, n: int, p: int, σ: float, rs: np.random.RandomState):
        self.n = n
        self.p = p
        self.σ = σ
        self.x = rs.uniform(-10, 10, n * p).reshape((n, p))
        self.β = rs.uniform(-10, 10, p)
        ϵ = rs.normal(0, σ, n)
        self.y = self.x @ self.β + ϵ
