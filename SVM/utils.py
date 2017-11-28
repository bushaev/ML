import numpy as np
import matplotlib.pyplot as plt


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
        m = len(X)
        dW = np.zeros(w.shape)

        for ind, x in enumerate(X):
            if y[ind] == 0 and pred[ind] > -1:
                dW = dW + self.C * x
            elif y[ind] == 1 and pred[ind] < 1:
                dW = dW - self.C * x

        dW = dW + w
        dW[0] = dW[0] - w[0]
        return dW


class GD:
    def __init__(self, lr):
        self.lr = lr

    def optimize(self, X, y, w, loss):
        return w - self.lr * loss.diff(X, y, w)


class GaussianKernel:
    def __init__(self, gamma=0.8):
        self.gamma = 0.8
    def __call__(self, *args):
        x, l = args
        return np.exp(-np.linalg.norm(x - l) ** 2 / (self.gamma ** 2))

class Dataset:
    def __init__(self, data, shuffle=True, dtype=float):
        self.data = np.asarray(data, dtype=dtype)
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.data)

    def CV(self, k = 5):
        if self.shuffle:
            np.random.shuffle(self.data)

        n = round(len(self.data) / k)
        for i in range(k - 1):
            yield np.concatenate((self.data[:i * n], self.data[(i + 1) * n:])), self.data[i * n: (i + 1) * n]
        yield self.data[:(i + 1) * n], self.data[(i + 1) * n:]

class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def trainsform(self, X):
        return (X - self.mean) / self.std

def read_chips():
    with open('chips.txt', 'r') as f:
        sl = [line.split(',') for line in f]
        p = np.array([[x, y, int(c[0])] for x,y, c in sl])
    return p

def plot_chips():
    p = read_chips()
    x, y, c = p[:, 0], p[:, 1], p[:, 2]
    plt.plot(x[c == '0'], y[c == '0'], 'ro', x[c == '1'], y[c == '1'], 'bo')
    plt.show()

class GeneticOptimizer:
    def __init__(self, population_size, generations_n, mutant_rate):
        self.population_size = population_size
        self.generations_n = generations_n
        self.mutant_rate = mutant_rate

    def create_individ(self, parents):
        np.random.shuffle(parents)
        ind = np.random.randint(0, len(parents[0]))
        return np.concatenate([parents[0][:ind], parents[1][ind:]])

    def mutate_individ(self, individ):
        coef = np.random.uniform(-2, 2, size=len(individ))
        return  individ * coef

    def mutate_population(self, population):
        length = len(population)
        np.random.shuffle(population)


        for i in range(int(length * self.mutant_rate)):
            population[i] = self.mutate_individ(population[i])

    def create_population(self, population):
        inds = np.arange(0, len(population))

        new_population = []
        for i in range(self.population_size):
            np.random.shuffle(inds)
            parents = [population[inds[0]], population[inds[1]]]
            new_population.append(self.create_individ(parents))

        return new_population

    def compute_loss(self, loss, X, y, w):
        pred = np.dot(X, w) > 0
        return loss.compute(y, pred, w)

    def key_fn(self, X, y, loss):
        def f(w):
            return self.compute_loss(loss, X, y, w)

        return f

    def optimize(self, X, y, w, loss):

        m = X.min(axis=0)
        max_y = y.max()

        population = [[np.random.uniform(-max_y / m[j], max_y / m[j]) for j in range(len(w))] for _ in
                      range(self.population_size)]
        costs = []
        for i in range(self.generations_n):
            kids = self.create_population(population)
            self.mutate_population(kids)

            new_population = np.concatenate((population, kids))
            population = sorted(new_population, key=self.key_fn(X, y, loss))[:self.population_size]
            costs.append(self.compute_loss(loss, X, y, population[0]))

        gens = np.arange(1, len(costs) + 1)
        plt.plot(gens, costs)
        plt.show()
        return population[0]