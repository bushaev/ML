def minkowski_distance(p, x1, x2):
    return (sum([abs((x1[i] - x2[i])) ** p for i in range(len(x1))])) ** (1 / p)

def manhattan_dist(x, y):
    return minkowski_distance(1, x, y)

def euclid_dist(x, y):
    return minkowski_distance(2, x, y)
