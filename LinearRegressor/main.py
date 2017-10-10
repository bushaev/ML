from regressor import *
from utils import *
model = LinearRegressor(1)
op = SquareError.solve_optimizer()
x = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
y = np.array([1, 2, 3, 4])

# print(op.optimize(x, y, None, None))
plot_data()