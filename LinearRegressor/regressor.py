import numpy as np
from utils import *


class LinearRegressor:
    def __init__(self, n, cost=SquareError):
        self.n = n
        self.w = np.zeros(n +1)
        self.cost = cost

    def predict(self, X):
        return np.dot(X, self.w)

    def fit(self, X, y, optimizer, lr, delta_c):
        ones = np.ones(shape=len(X))
        X = np.c_[ones, X]

        optimizer.optimize()

