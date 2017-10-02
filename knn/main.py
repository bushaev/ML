from knn import *

data = Dataset(read_chips(), shuffle=True)


def plot():
    plot_chips()
    voronoi_chips()
    for train, valid in data.LOO():
        model = KNearestNeighborClassifier(k)
        model.train(train)
        model.predict(valid, True)


def TK_CV(t, k, dist, method, kernel, accuracy=False):
    total_f1 = 0.0
    for i in range(t):
        total_f1 += CV(k, dist, method, kernel, accuracy)

    return total_f1 / t


def CV(k, dist, method, kernel, accuracy=False):
    F1 = 0
    n = 0
    for train, valid in data.cross_validate(5):
        model = KNearestNeighborClassifier(k, distance=dist, method=method, classes=2, kernel=kernel)
        model.train(train)
        # valid = np.asarray([valid], dtype=float)
        if accuracy:
            F1 += model.accuracy(valid)
        else:
            F1 += model.F1(valid)
        n += 1
    return F1 / n


##############
## Find best k
##############
F1s = []
for k in range(1, 20):
    F1s.append(TK_CV(t=10, k=k, dist=euclid_distance, method='brute', kernel=None))

k = np.argmax(F1s) + 1
print("Best k is ", k, " with f1 measure ", F1s[k - 1])

# Try KDTree
print("F1 Measure with kd_tree is ", TK_CV(t=10, k=k, dist=euclid_distance, method='kd_tree', kernel=None))

# Try different metrics
print("Euclid distance F1 ", TK_CV(t=10, k=k, dist=euclid_distance, method='brute', kernel=None))
print("Manhattan distance F1 ", TK_CV(t=10, k=k, dist=manhattan_distance, method='brute', kernel=None))

# Try kernels
print("Gaussian kernel F1 ", TK_CV(t=10, k=k, dist=euclid_distance, method='brute', kernel=gaussian_kernel))
print("Some other hard-to-spell kernel F1 ",
      TK_CV(t=10, k=k, dist=euclid_distance, method='brute', kernel=some_other_kernel))
print("Best accuracy - ", TK_CV(t=10, k=k, dist=euclid_distance, method='brute', kernel=gaussian_kernel, accuracy=True))

# Try data transform with kernels (multiply)
data = np.asarray(read_chips(), dtype=float)
ndata = transform_data(data, lambda p: (p[0] * p[1]))
data = Dataset(ndata)

print("Data transform(multiply) with kernels ", TK_CV(t=10, k=k, dist=euclid_distance, method='brute', kernel=None))
plot_transform(ndata)

# Data transform (addition)
data = np.asarray(read_chips(), dtype=float)
ndata = transform_data(data, lambda p: (p[0]) ** 2 + (p[1]) ** 2)
data = Dataset(ndata)
print("Data transform(cone) with kernels ", TK_CV(t=10, k=k, dist=euclid_distance, method='brute', kernel=gaussian_kernel))
plot_transform(ndata)
