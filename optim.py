import numpy as np
import matplotlib.pyplot as plt

class GeneticOptimizer:
    def __init__(self, population_size, mutant_rate):
        self.population_size = population_size
        self.mutant_rate = mutant_rate
        self.population = None
        self.pr = None

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

    def key_fn(self, X, y, loss):
        def f(w):
            return loss.compute(self.pr(X, w), y, w)

        return f

    def optimize(self, X, y, w, loss):
        if self.population is None:
            m = X.min(axis=0)
            max_y = y.max()

            self.population = [[np.random.uniform(-max_y / m[j], max_y / m[j])
                                for j in range(len(w))] for _ in range(self.population_size)]

        kids = self.create_population(self.population)
        self.mutate_population(kids)
        new_population = np.concatenate((self.population, kids))
        self.population = sorted(new_population, key=self.key_fn(X, y, loss))[:self.population_size]

        return self.population[0]

    def __str__(self):
        return 'genetic'

class GD:
    def __init__(self, lr):
        self.lr = lr

    def optimize(self, X, y, w, loss):
        return w - self.lr * loss.diff(X, y, w)

    def __str__(self):
        return 'gd'
