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

    return np.asarray(X, dtype=int), np.asarray(Y, dtype=int)


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


class SquareError:
    class SolveOptimizer:
        def optimize(self, X, y, lr, cost):
            return np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), y)

    @staticmethod
    def compute(pred, y):
        return np.linalg.norm(pred - y) ** 2

    @staticmethod
    def solve_optimizer():
        return SquareError.SolveOptimizer()


class SGD:
    def __init__(self, lr, cost):
        self.lr = lr
        self.cost = cost

    def optimize(self, X, y, w, lr, cost):
        pass

    def plot_cost(self):
        pass
