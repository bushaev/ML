from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt


def minkowski_distance(p):
    def distance(x1, x2):
        return (sum([abs((x1[i] - x2[i])) ** p for i in range(len(x1))])) ** (1 / p)

    return distance


manhattan_distance = minkowski_distance(p=1)
euclid_distance = minkowski_distance(p=2)


def read_chips():
    with open("chips.txt", 'r') as f:
        sl = [line.split(',') for line in f]
        p = np.array([[x, y, int(c[0])] for x, y, c in sl])
    return p


def plot_chips():
    p = read_chips()
    x, y, c = p[:, 0], p[:, 1], p[:, 2]
    plt.plot(x[c == '0'], y[c == '0'], 'ro', x[c == '1'], y[c == '1'], 'bo')
    plt.show()


def voronoi_chips():
    from scipy.spatial import voronoi_plot_2d, Voronoi
    points = read_chips()[:, [0, 1]]
    vor = Voronoi(points)
    voronoi_plot_2d(vor)
    plt.show()


def most_common(lst):
    from collections import Counter
    data = Counter(lst)
    return data.most_common(1)[0][0]


class Dataset:
    def __init__(self, data, shuffle=True, dtype=float):
        self.data = np.asarray(data, dtype=dtype)
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.data)

    def cross_validate(self, k):
        if self.shuffle:
            np.random.shuffle(self.data)
        n = round(len(self.data) / k)
        for i in range(k - 1):
            yield np.concatenate((self.data[:i * n], self.data[(i + 1) * n:])), self.data[i * n:(i + 1) * n]
        yield self.data[:(i + 1) * n], self.data[(i + 1) * n:]

    def LOO(self):
        for i in range(len(self.data)):
            yield np.concatenate((self.data[:i], self.data[i + 1:])), self.data[i]


def key_function(point, dist, d=2):
    def f(p):
        return dist(point[:d], p[:d])  # p is (object, class)

    return f


def brute_force(data, point, dist, k):
    sd = np.asarray(sorted(data, key=key_function(point=point, dist=dist, d=2)), dtype=float)
    return sd[:k]


class KDTreeNode:
    def __init__(self, nb_elements, left, right, element):
        self.nb_elements = nb_elements
        self.element = element
        self.left = left
        self.right = right

    def leaf(self):
        return self.left is None and self.right is None


class KDTree:
    def __init__(self, data, dim, dist, leaf_size):
        self.data = data
        self.dim = dim
        self.dist = dist
        self.leaf_size = leaf_size
        self.root = self.setup_tree(data, 0)

    def display(self, node, depth):
        print ("  " * depth, node.element)

        if node.leaf():
            return

        self.display(node.left, depth + 1)
        self.display(node.right, depth + 1)

    def setup_tree(self, data, depth):
        n = len(data)
        if n <= self.leaf_size:
            return KDTreeNode(n, None, None, data)

        axis = depth % self.dim
        sd = sorted(data, key=itemgetter(axis))
        median = n // 2

        return KDTreeNode(
            nb_elements=n,
            left=self.setup_tree(sd[:median], depth + 1),
            right=self.setup_tree(sd[median:], depth + 1),
            element=sd[median]
        )

    def query(self, data, point, dist, k):
        return self.execute_query(self.root, point, 0, k)

    def execute_query(self, node, point, depth, k):
        if node.leaf():
            if k >= node.nb_elements:
                return np.asarray(node.element)
            else:
                return brute_force(node.element, point, self.dist, k)

        axis = depth % self.dim

        if point[axis] >= node.element[axis]:
            next_node = node.right
            opposite_node = node.left
        else:
            next_node = node.left
            opposite_node = node.right

        result_r = self.execute_query(next_node, point, depth + 1, k)

        if next_node.nb_elements < k:
            result_l = self.execute_query(opposite_node, point, depth + 1, k - next_node.nb_elements)
            return np.concatenate((result_r, result_l))
        return result_r


def gaussian_kernel(t):
    return np.exp(-0.5 * t) / np.sqrt(2 * np.pi)


# Epanechnikov kernel
def some_other_kernel(t):
    if t < 1:
        return 0.75 * (1 - t ** 2)
    else:
        return 0

