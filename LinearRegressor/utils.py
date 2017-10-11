import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_data():
    with open('prices.txt', 'r') as f:
        X = []
        Y = []
        f.readline()
        for line in f:
            area, room, price = line.strip().split(',')
            X.append([area, room])
            Y.append(price)

    return np.asarray(X, dtype=float), np.asarray(Y, dtype=float)


def plot_data():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y = read_data()
    x1, x2 = x[:, 0], x[:, 1]

    ax.scatter(x1, x2, y)
    ax.set_xlabel('area')
    ax.set_ylabel('room')
    ax.set_zlabel('price')
    plt.show()


def plot_final(xx, yy, zz):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y = read_data()
    x1, x2 = x[:, 0], x[:, 1]

    ax.scatter(x1, x2, y)
    ax.plot_trisurf(xx, yy, zz, color='r', alpha=0.2)

    ax.set_xlabel('area')
    ax.set_ylabel('room')
    ax.set_zlabel('price')
    plt.show()


class SquareError:
    class SolveOptimizer:
        def optimize(self, X, y, lr, w, cost):
            return np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), y)

        def __str__(self):
            return 'equation'

    @staticmethod
    def compute(pred, y):
        return (1 / (2 * len(y))) * sum((y - pred) ** 2)

    @staticmethod
    def solve_optimizer():
        return SquareError.SolveOptimizer()

    @staticmethod
    def diff(w, X, y):
        pred = np.dot(X, w)
        a = pred - y
        r = np.dot(a, X)
        return r / len(X)


class SGD:
    def optimize(self, X, y, lr, w, cost):
        return w - cost.diff(w, X, y) * lr

    def __str__(self):
        return 'SGD'
