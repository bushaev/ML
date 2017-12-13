import numpy as np
from operator import itemgetter

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
        print("  " * depth, node.element)

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