from knn import *

data = Dataset(read_chips(), shuffle=True)
plot_chips()
voronoi_chips()


def test(k, dist, method, kernel):
    F1 = 0
    n = 0
    for train, valid in data.LOO():
        model = KNearestNeighborClassifier(k, distance=dist, method=method, classes=2, kernel=kernel)
        model.train(train)
        valid = np.asarray([valid], dtype=float)
        F1 += model.F1(valid)
        n += 1
    return F1 / n


##############
## Find best k
##############
F1s = []
for k in range(1, 20):
    F1s.append(test(k, dist=euclid_distance, method='brute', kernel=None))

k = np.argmax(F1s) + 1
print("Best k is ", k, " with f1 measure ", F1s[k - 1])

# Try KDTree
print("F1 Measure with kd_tree is ", test(k, dist=euclid_distance, method='kd_tree', kernel=None))

# Try different metrics
print("Euclid distance F1 ", test(k, dist=euclid_distance, method='brute', kernel=None))
print("Manhattan distance F1 ", test(k, dist=manhattan_distance, method='brute', kernel=None))

# Try kernels
print("Gaussian kernel F1 ", test(k, dist=euclid_distance, method='brute', kernel=gaussian_kernel))
print("Some other hard-to-spell kernel F1 ", test(k, dist=euclid_distance, method='brute', kernel=some_other_kernel))
