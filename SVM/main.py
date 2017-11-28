from utils import *
from svm import SVM

data = Dataset(read_chips())

f1s = []
for train ,test in data.CV(10):
    X_trn, y_trn = train[:, :2], train[:, 2]
    X_valid, y_valid = test[:, :2], test[:, 2]

    model = SVM(1, kernel=GaussianKernel(gamma=2))
    model.fit(X_trn, y_trn, GD(0.001), n_epoch=600, plot=True)
    f1s.append(model.f1(X_valid, y_valid))

print ("F1 score =,", np.mean(f1s))