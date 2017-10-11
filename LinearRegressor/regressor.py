import numpy as np
import matplotlib.pyplot as plt
from utils import *


class LinearRegressor:
    def __init__(self, n, cost=SquareError):
        self.n = n
        # self.w = np.random.randn(n + 1)
        self.w = np.zeros(n + 1)
        self.cost = cost

    def preprocess(self, X):
        m = np.mean(X, axis=0)
        r = np.max(X, axis=0) - np.min(X, axis=0)
        X = (X - m) / r
        ones = np.ones(shape=len(X))
        return np.c_[ones, X]

    def predict_preproccessed(self, X):
        return np.dot(X, self.w)

    def predict(self, X):
        X = self.preprocess(X)
        return self.predict_preproccessed(X)

    def ccost(self, X, y):
        X = self.preprocess(X)
        return self.cost.compute(self.predict_preproccessed(X), y)

    def fit(self, X, y, optimizer, lr, delta_c):
        X = self.preprocess(X)

        cs = [self.cost.compute(self.predict_preproccessed(X), y)]
        i = [1]
        while len(cs) is 1 or cs[-2] - cs[-1] > delta_c:
            self.w = optimizer.optimize(X, y, lr, self.w, self.cost)
            cs.append(self.cost.compute(self.predict_preproccessed(X), y))
            i.append(i[-1] + 1)
            # if len(i) > 400:
            #     break

        plt.xlabel('iterations')
        plt.ylabel('Emperical risk')
        plt.title('Emperical rist optimization')
        plt.plot(i, cs)
        plt.show()

