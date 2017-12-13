from datasets import read_prices, SpamDataset, read_chips
from svm import SVM
from linear import LinearRegressor
from loss import RMSE
from naive import SpamClissifier
from optim import GD, GeneticOptimizer
import numpy as np

from utils import plot_confusion_matrix, GaussianKernel
from metrics import f1_score

data = read_chips()


f1s = []
for train ,test in data.CV(10):
    X_trn, y_trn = train[:, :2], train[:, 2]
    X_valid, y_valid = test[:, :2], test[:, 2]

    model = SVM(1, kernel=GaussianKernel(gamma=2))
    model.fit(X_trn, y_trn, GD(0.001), n_epoch=600, plot=False)
    f1s.append(model.f1(X_valid, y_valid))

print ("F1 score =", np.mean(f1s))


