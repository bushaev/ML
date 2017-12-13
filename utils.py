import numpy as np
import matplotlib.pyplot as plt
import itertools


class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std

class GaussianKernel:
    def __init__(self, gamma=0.8):
        self.gamma = 0.8
    def __call__(self, *args):
        x, l = args
        return np.exp(-np.linalg.norm(x - l) ** 2 / (self.gamma ** 2))

class PolynomialKernel:
    def __init__(self, power=2):
        self.power = power

    def __call__(self, *args, **kwargs):
        x, l = args
        return np.dot(x.T, l) ** self.power

def most_common(lst):
    from collections import Counter
    data = Counter(lst)
    return data.most_common(1)[0][0]

def transform_data(data, f):
    ndata = []
    for x in data:
        ndata.append(np.concatenate(([f(x)], x)))
    return np.asarray(ndata)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',
        cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def pearson(x, y):
    x_m = np.mean(x)
    y_m = np.mean(y)
    l = len(x)

    denom = np.sum([(x[i] - x_m) * (y[i] - y_m) for i in range(l)])
    other_one  = np.sqrt(np.sum([(x[i] - x_m) ** 2 for i in range(l)])
                         * np.sum([(y[i] - y_m) ** 2 for i in range(l)]))


    r =  np.abs(denom / other_one)
    return r if not np.isnan(r) else 0

def spearman(x, y):
    return pearson(
        np.argsort(x),
        np.argsort(y),
    )

def chi2(x, y):
    # x = np.round((x - np.mean(x)) / np.std(x))
    x_classes = np.unique(x)
    y_classes = np.unique(y)
    cov = np.zeros(shape=(len(x_classes), len(y_classes)))
    for i, x_v in enumerate(x_classes):
        for j, y_v in enumerate(y_classes):
            cov[i, j] = np.sum((x == x_v) & (y == y_v))

    n = np.sum(cov)
    cov = cov / n
    col_p = np.sum(cov, axis=0)
    row_p = np.sum(cov, axis=1)

    f_exp = np.zeros(shape=(len(x_classes), len(y_classes)))
    for i in range(len(x_classes)):
        for j in range(len(y_classes)):
            f_exp = row_p[i] * col_p[j]

    f_obs = np.ravel(cov)
    f_exp = np.ravel(f_exp)

    chisq = f_obs
    chisq -= n * f_exp
    chisq **= 2
    with np.errstate(invalid="ignore"):
        chisq /= n * f_exp


    return chisq.sum() / 140
