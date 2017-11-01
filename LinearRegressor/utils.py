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


def plot_final(xx, yy, zz, norm=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y = read_data()
    x1, x2 = x[:, 0], x[:, 1]
    if norm:
        x1 = (x1 - np.mean(x1)) / np.std(x1)
        x2 = (x2 - np.mean(x2)) / np.std(x2)

    ax.scatter(x1, x2, y)

    ax.plot_surface(xx, yy, zz, color='r', alpha=0.05)

    ax.set_xlabel('area')
    ax.set_ylabel('room')
    ax.set_zlabel('price')
    plt.show()


class SquareError:
    class SolveOptimizer:
        def optimize(self, X, y, lr, w, cost):
            return np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), y)

        def __str__(self):
            return 'equation'

    @staticmethod
    def compute(pred, y):
        return (1 / (len(y))) * sum((y - pred) ** 2)

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


class GeneticOptimizer:
    def __init__(self, mutant_rate, population_size, generations):
        self.mutant_rate = mutant_rate
        self.population_size = population_size
        self.generations = generations

    def create_individual(self, parents):
        np.random.shuffle(parents)
        ind = np.random.randint(0, len(parents[0]))
        return np.concatenate((parents[0][:ind], parents[1][ind:]))

    def mutate_individual(self, individual):
        coefs = np.random.uniform(0.0, 2.0, size=len(individual))
        return individual * coefs

    def mutate_population(self, population):
        length = len(population)
        np.random.shuffle(population)

        for i in range(int(length * self.mutant_rate)):
            population[i] = self.mutate_individual(population[i])

        return population

    def create_population(self, population):
        inds = np.arange(0, len(population))

        new_population = []
        for i in range(self.population_size):
            np.random.shuffle(inds)
            new_population.append(self.create_individual([population[inds[0]], population[inds[1]]]))

        return new_population

    def compute_cost(self, cost, X, y, w):
        pred = np.dot(X, w)
        return cost.compute(pred, y)

    def key_function(self, X, y, cost):
        def f(w):
            return self.compute_cost(cost, X, y, w)

        return f

    def optimize(self, X, y, lr, w, cost):

        m = X.min(axis=0)
        max_y = y.max()

        population = [[np.random.uniform(-max_y / m[j], max_y / m[j]) for j in range(len(w))] for _ in range(self.population_size)]
        costs = []
        for i in range(self.generations):
            kids = self.create_population(population)
            self.mutate_population(kids)

            new_population = np.concatenate((population, kids))
            population = sorted(new_population, key=self.key_function(X, y, cost))[:self.population_size]
            costs.append(self.compute_cost(cost, X, y, population[0]))

        gens = np.arange(1, len(costs) + 1)
        plt.plot(gens, costs)
        plt.show()
        return population[0]

    def __str__(self):
        return 'genetic'


class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std
