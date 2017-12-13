from neighbors import (brute_force, KDTree)
from dist import euclid_dist
from utils import most_common
from metrics import f1_score, accuracy

import numpy as np
import matplotlib.pyplot as plt

class KNearestNeighborClassifier:
    def __init__(self, k, distance=euclid_dist, method='brute', classes=2, kernel=None):
        self.k = k
        self.points = None
        self.distance = distance
        self.d = None
        self.classes = range(classes)
        self.kernel = kernel
        if method == 'brute':
            self.method = brute_force
        elif method is 'kd_tree':
            self.tree = None
            self.method = 'kd_tree'

    def fit(self, p):
        if len(p) > 0:
            self.points = p
            self.d = len(p[0]) - 1

            if self.method is 'kd_tree':
                self.tree = KDTree(p, self.d, self.distance, self.k + 5)
                self.method = self.tree.query

    def predict(self, x, plot=False):
        neighbors = self.method(self.points, x, self.distance, self.k)

        if plot:
            plt.plot(x[0], x[1], 'go')
            x_data = self.points[:, 0]
            y_data = self.points[:, 1]
            plt.plot(x_data, y_data, 'ko')
            xx = neighbors[:, 0]
            yy = neighbors[:, 1]
            cc = neighbors[:, 2]
            plt.plot(xx[cc == 0], yy[cc == 0], 'ro', xx[cc == 1], yy[cc == 1], 'bo')
            plt.show()

        if not self.kernel:
            return most_common(neighbors[:, self.d])
        else:
            scores = []
            for y in self.classes:
                s = [int(n[-1] == y) * self.kernel(self.distance(x[:self.d], n[:self.d]))
                     for n in neighbors]
                scores.append(sum(s))
            return float(np.argmax(scores))

    def test(self, data):
        return np.asarray([self.predict(x) for x in data], dtype=float)

    def contingency_table(self, data):
        pred = self.test(data)
        labels = data[:, self.d]
        TP = sum((pred == 1) & (labels == 1))
        TN = sum((pred == 0) & (labels == 0))
        FP = sum((pred == 1) & (labels == 0))
        FN = sum((pred == 0) & (labels == 1))

        return [TP, FN, FP, TN]

    def accuracy(self, test_data):
        pred = self.test(test_data)
        labels = test_data[:, self.d]
        return accuracy(labels, pred)

    def F1(self, test_data):
        return f1_score(self.contingency_table(test_data))
