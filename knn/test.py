import unittest
from utils import *


class DistanceTest(unittest.TestCase):
    def test_euclid(self):
        x = [(0, 0), (1, 1)]
        dist = euclid_distance(x[0], x[1])
        self.assertAlmostEqual(dist, np.sqrt(2))

    def test_manhattan(self):
        x = [(0, 0), (1, 1)]
        dist = manhattan_distance(x[0], x[1])
        self.assertAlmostEqual(dist, 2)


class MethodsTest(unittest.TestCase):
    def test_brute_force(self):
        data = np.asarray(read_chips(), dtype=float)
        new_x = np.random.random(size=2)
        neighbors = brute_force(data, new_x, euclid_distance, 50)

        for i in range(len(neighbors) - 1):
            self.assertLess(euclid_distance(new_x, neighbors[i]), euclid_distance(new_x, neighbors[i + 1]),
                            "brute force doesnt return neighbors in order")

    def test_kd_tree(self):
        data = np.asarray(read_chips(), dtype=float)
        kd = KDTree(data, 2, euclid_distance, 15)
        query_point = np.random.random(size=2)

        neighbors = kd.query(data, query_point, euclid_distance, 5)
        for i in range(len(neighbors) - 1):
            self.assertLess(euclid_distance(query_point, neighbors[i]), euclid_distance(query_point, neighbors[i + 1]),
                            "kd_tree doesnt return neighbors in order")


if __name__ == '__main__':
    unittest.main()
