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
model.fit(X, y, GeneticOptimizer(0.2, 100, 2000), 0.1, 0.1, max_iter=1)

rooms = np.random.randint(1, 5, size=15)
area = np.random.randint(1000, 4500, size=15)
area, rooms = np.meshgrid(area, rooms)

features = np.stack((area.reshape(1, -1).T, rooms.reshape(1, -1).T), axis=1).reshape(225, 2)
predictions = model.predict(features).reshape(15, 15)

print (model.ccost(X, y))
prices = model.predict(features)
plot_final(area, rooms, predictions, False)
