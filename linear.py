from loss import RMSE
from utils import Scaler
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressor:
    def __init__(self, n, cost=RMSE()):
        self.n = n
        self.w = np.random.randn(n + 1)
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
        if np.sum(X[:, 0] == 1) != X.shape[0]:
            X = self.preprocess(X)
        return self.predict_preproccessed(X)

    def ccost(self, X, y):
        X = self.preprocess(X)
        return self.cost.compute(self.predict_preproccessed(X), y)

    def fit(self, X, y, optimizer, epoch, plot):
        if str(optimizer) == 'equation' or str(optimizer) == 'genetic':
            self.normalize = False

        if str(optimizer) == 'genetic':
            optimizer.pr = lambda X, w : np.dot(X, w)

        self.scaler.fit(X)
        X = self.preprocess(X)

        loss = [self.cost.compute(self.predict_preproccessed(X), y)]
        for _ in range(epoch):
            self.w = optimizer.optimize(X, y, self.w, self.cost)
            loss.append(self.cost.compute(self.predict_preproccessed(X), y))

        if plot:
            plt.xlabel('iterations')
            plt.ylabel('Emperical risk')
            plt.title('Emperical rist optimization')
            plt.plot(np.arange(epoch + 1), loss)
            plt.show()

