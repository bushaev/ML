from regressor import *
from utils import *

model = LinearRegressor(2)
X, y = read_data()
model.fit(X, y, SGD(), 1, 50)
model.fit(X, y, SGD(), 0.3, 0.001)
model.fit(X, y, SGD(), 0.3, 0.0001)
# model.fit(X, y, SquareError.solve_optimizer(), 0.1, 0.1)


# Plot plane
rooms = np.random.randint(1, 5, size=15)
area = np.random.randint(1000, 4500, size=15)
features = np.asarray(list(zip(area, rooms)), dtype=float)
prices = model.predict(features)
plot_final(area, rooms, prices)
