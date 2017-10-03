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

metrics = {
    'euclid' : euclid_distance,
    'manhattan' : manhattan_distance
}

best_k = 1
best_kernel = gaussian_kernel
best_metric = euclid_distance
best_f = 0
best_t = None
for t in [None, 1, 2]:
    if t == 1:
        data = np.asarray(read_chips(), dtype=float)
        ndata = transform_data(data, lambda p: (p[0] * p[1]))
        data = Dataset(ndata)
    elif t == 2:
        data = np.asarray(read_chips(), dtype=float)
        ndata = transform_data(data, lambda p: (p[0]) ** 2 + (p[1]) ** 2)
        data = Dataset(ndata)

    for k in range(1, 10):
        for kernel in [gaussian_kernel, some_other_kernel]:
            for metric in ['euclid', 'manhattan']:
                f = TK_CV(10, k=k, dist=metrics[metric], method='brute', kernel=kernel)
                # print ("K:", k, " kernel:", kernel, " metric:", metric, " F1:", f)

                if f > best_f:
                    best_k = k
                    best_f = f
                    best_kernel = kernel
                    best_metric = metric
                    best_t = t


print("K:", best_k, " kernel:", best_kernel.__name__, " metric:", best_metric, " T:", best_t, " F1:", best_f)

Ks = []
F1s = []
for k in range(1, 40):
    F1 = TK_CV(10, k=k, dist=metrics[best_metric], method='brute', kernel=best_kernel)
    F1s.append(F1)
    Ks.append(k)

plt.plot(Ks, F1s)
plt.show()