from knn import *

data = Dataset(read_chips(), shuffle=True)

def plot():
    plot_chips()
    voronoi_chips()
    for train, valid in data.LOO():
        model = KNearestNeighborClassifier(k)
        model.train(train)
        model.predict(valid, True)


def CV(k, dist, method, kernel, accuracy=False):
    F1 = 0
    n = 0
    for train, valid in data.LOO():
        model = KNearestNeighborClassifier(k, distance=dist, method=method, classes=2, kernel=kernel)
        model.train(train)
        valid = np.asarray([valid], dtype=float)
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
    F1s.append(CV(k, dist=euclid_distance, method='brute', kernel=None))

k = np.argmax(F1s) + 1
print("Best k is ", k, " with f1 measure ", F1s[k - 1])

# Try KDTree
print("F1 Measure with kd_tree is ", CV(k, dist=euclid_distance, method='kd_tree', kernel=None))

# Try different metrics
print("Euclid distance F1 ", CV(k, dist=euclid_distance, method='brute', kernel=None))
print("Manhattan distance F1 ", CV(k, dist=manhattan_distance, method='brute', kernel=None))

# Try kernels
print("Gaussian kernel F1 ", CV(k, dist=euclid_distance, method='brute', kernel=gaussian_kernel))
print("Some other hard-to-spell kernel F1 ", CV(k, dist=euclid_distance, method='brute', kernel=some_other_kernel))
print("Best accuracy - ", CV(k, dist=euclid_distance, method='brute', kernel=gaussian_kernel, accuracy=True))

# Try data transform with kernels
data = np.asarray(read_chips(), dtype=float)
ndata = transform_data(data, lambda p: p[0] * (1 / 3) + p[1] * (1 / 3))
data = Dataset(ndata)

print ("Data transform with kernels ", CV(k=k, dist=euclid_distance, method='brute', kernel=exp_kernel))
