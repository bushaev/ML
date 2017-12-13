import numpy as np

class SVMLoss:
    def __init__(self, C=1):
        self.C = C

    def compute(self, y_true, y_pred, w):
        c = 0
        for y, pred in zip(y_true, y_pred):
            if y == 0:
                c += max(0, pred + 1)
            elif y == 1:
                c += max(0, 1 - pred)
            else:
                raise ValueError(y)
        c = c * self.C
        c += np.linalg.norm(w[1:]) ** 2
        return c

    def diff(self, X, y, w):
        pred = np.dot(X, w)
        dW = np.zeros(w.shape)

        for ind, x in enumerate(X):
            if y[ind] == 0 and pred[ind] > -1:
                dW = dW + self.C * x
            elif y[ind] == 1 and pred[ind] < 1:
                dW = dW - self.C * x

        dW = dW + w
        dW[0] = dW[0] - w[0]
        return dW

class RMSE:
    class SolveOptimizer:
        def optimize(self, X, y, w, cost):
            return np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), y)

        def __str__(self):
            return 'equation'

    def compute(self, pred, y, w=None):
        return np.sqrt((1 / (len(y))) * sum((y - pred) ** 2))

    @classmethod
    def solve_optimizer(cls):
        return cls.SolveOptimizer()

    def diff(self, X, y, w):
        pred = np.dot(X, w)
        a = pred - y
        r = np.dot(a, X)
        return r / len(X)
