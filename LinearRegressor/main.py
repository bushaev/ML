from regressor import *
from utils import *


def r(x1, x2, w):
    return w[0] + w[1] * x1 + w[2] * x2


model = LinearRegressor(2)
X, y = read_data()
# model.fit(X, y, SGD(), 1, 0.1)
# model.fit(X, y, SGD(), 0.01, 0.001)
# model.fit(X, y, SGD(), 0.01, 0.0001)
# model.fit(X, y, SquareError.solve_optimizer(), 0.1, 0.1)

model.fit(X, y, GeneticOptimizer(0.2, 100, 5000), 0.1, 0.1, max_iter=1)
print(model.w)
print(model.ccost(X, y))
rooms = np.random.randint(1, 5, size=15)
area = np.random.randint(1000, 4500, size=15)
features = np.asarray(list(zip(area, rooms)), dtype=float)
# area = (area - np.mean(area)) / np.std(area)
# rooms = (rooms - np.mean(rooms)) / np.std(rooms)
area, rooms = np.meshgrid(area, rooms)
z = r(area, rooms, model.w)

prices = model.predict(features)
plot_final(area, rooms, z, False)
