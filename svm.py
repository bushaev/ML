from utils import GaussianKernel
from loss import SVMLoss
from metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np

class SVM:
    def __init__(self, C, kernel=GaussianKernel()):
        self.loss = SVMLoss(C)
        self.kernel = kernel
        self.landmarks = None
        self.w = None

    def transform(self, X):
        new_x = np.zeros((len(X), len(self.landmarks) + 1))
        for ind, x in enumerate(X):
            x_transformed = np.zeros(len(self.landmarks) + 1)
            x_transformed[0] = 1
            for i, l in enumerate(self.landmarks):
                x_transformed[i + 1] = self.kernel(x, l)
            new_x[ind] = x_transformed
        return new_x

    def predict(self, X):
        X = self.transform(X)
        return np.dot(X, self.w) > 0

    def _predict(self, X):
        return np.dot(X, self.w) > 0

    def fit(self, X, y, optim, n_epoch, plot=True):
        self.landmarks = X
        X = self.transform(X)
        self.w = np.random.rand(len(X[0]))

        losses = []
        for _ in range(n_epoch):
            self.w = optim.optimize(X, y, self.w, self.loss)
            losses.append(self.loss.compute(y, self._predict(X), self.w))

        if plot:
            ind = np.arange(0, n_epoch)
            plt.plot(ind, losses)
            plt.show()

    def confusion_matrix(self, X, y):
        preds = self.predict(X)

        TP = np.sum((y == 1) & (preds == 1))
        FP = np.sum((y == 0) & (preds == 1))
        TN = np.sum((y == 0) & (preds == 0))
        FN = np.sum((y == 1) & (preds == 0))

        return [TP, FN, FP, TN]

    def f1(self, X, y, cm=None):
        cm = cm or self.confusion_matrix(X, y)

        return f1_score(cm)
