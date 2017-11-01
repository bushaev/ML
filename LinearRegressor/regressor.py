import numpy as np
import matplotlib.pyplot as plt
from utils import *


class LinearRegressor:
    def __init__(self, n, cost=SquareError):
        self.n = n
        self.w = np.random.randn(n + 1) * 10000
        self.normalize = True
        self.cost = cost
        self.scaler = Scaler()

    def preprocess(self, X):
        if self.normalize:
            X = self.scaler.transform(X)
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

    def fit(self, X, y, optimizer, lr, delta_c, max_iter=None):
        if str(optimizer) == 'equation' or str(optimizer) == 'genetic':
            self.normalize = False

        self.scaler.fit(X)
        X = self.preprocess(X)

        cs = [self.cost.compute(self.predict_preproccessed(X), y)]
        i = [1]
        count = 0
        while len(cs) is 1 or cs[-2] - cs[-1] > delta_c:
            count += 1
            self.w = optimizer.optimize(X, y, lr, self.w, self.cost)
            cs.append(self.cost.compute(self.predict_preproccessed(X), y))
            i.append(i[-1] + 1)

            if max_iter and len(i) > max_iter:
                break

        print("count, ", count)

        if str(optimizer) == 'SGD':
            plt.xlabel('iterations')
            plt.ylabel('Emperical risk')
            plt.title('Emperical rist optimization')
            plt.plot(i, cs)
            plt.show()

