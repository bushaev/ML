import numpy as np
from datasets import read_arcene
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from svm import SVM
from optim import GD
from utils import (pearson, spearman, chi2)

metrics = {
    'pearson' : pearson,
    'spearman' : spearman,
    'chi2' : chi2,
    'together' : lambda x, y : pearson(x, y) * spearman(x, y) * chi2(x, y)
}

def sort_features(met, n):
    features = []
    inds = np.arange(n)
    for m in met:
        crit = metrics[m]
        features.append(
            np.array(sorted(inds, key=lambda i : crit(X[:, i], y), reverse=True))
        )

    return features

def plot_imoprtance(ind, y, met):
    n_feat = len(ind)
    for m in met:
        ctit = metrics[m]
        plt.bar(np.arange(n_feat),
                height=[ctit(X[:, i], y)
                           for i in ind], label=m)
    plt.legend()
    plt.xticks([])
    plt.show()

def plot_train_valid(ind, met):
    n_feat = len(ind)
    for m in met:
        ctit = metrics[m]
        plt.bar(np.arange(n_feat),
                height=[ctit(X[:, i], y)
                        for i in ind], label=m + '_train')
        plt.bar(np.arange(n_feat),
                height=[ctit(X_valid[:, i], y_valid)
                        for i in ind], label=m + '_valid')
    plt.legend()
    plt.xticks([])
    plt.show()

met = list(metrics.keys())

X, y = read_arcene('train')
X_valid, y_valid = read_arcene('valid')
features = sort_features(['pearson', 'spearman', 'chi2'], 5000)
plot_train_valid(features[0], ['pearson'])
plot_train_valid(features[1], ['spearman'])
plot_train_valid(features[2], ['chi2'])
plot_imoprtance(features[0], y, ['chi2', 'pearson', 'spearman'])
plot_imoprtance(features[1], y, ['spearman', 'chi2', 'pearson'])

it = 500

a = np.arange(it)
f_pear = sorted(a, key= lambda i : pearson(X[:, i], y), reverse=True)
f_spear = sorted(a, key= lambda i : spearman(X[:, i], np.ravel(y)), reverse=True)
f_chi = sorted(a, key= lambda i : chi2(X[:, i], y), reverse=True)
f_tog = sorted(a, key= lambda i : pearson(X[:, i], y) * chi2(X[:, i], y) * spearman(X[:, i], np.ravel(y)), reverse=True)
f1s_pearson = []
f1s_spearman = []
f1s_chi = []
f1s_together = []



for i in range(1, it):
    params = { }
    svm_pearson = KNeighborsClassifier(**params)
    svm_spearman = KNeighborsClassifier(**params)
    svm_chi = KNeighborsClassifier(**params)
    svm_together = KNeighborsClassifier(**params)

    svm_pearson.fit(X[:, f_pear[:i]], np.ravel(y),)
    svm_spearman.fit(X[:, f_spear[:i]], np.ravel(y))
    svm_chi.fit(X[:, f_chi[:i]], np.ravel(y))
    svm_together.fit(X[:, f_tog[:i]], np.ravel(y))

    # f1s_my.append(svm_my.f1(X[:, f_obs[:i]], y))
    f1s_pearson.append( f1_score(y_valid, svm_pearson.predict(X_valid[:, f_pear[:i]])) )
    f1s_spearman.append(f1_score(y_valid, svm_spearman.predict(X_valid[:, f_spear[:i]])))
    f1s_chi.append(f1_score(y_valid, svm_chi.predict(X_valid[:, f_chi[:i]])))
    f1s_together.append(f1_score(y_valid, svm_together.predict(X_valid[:, f_tog[:i]])))

plt.plot(np.arange(1, it), f1s_pearson, label='pearson')
plt.plot(np.arange(1, it), f1s_spearman, label='spearman')
plt.plot(np.arange(1, it), f1s_chi, label='chi test')
plt.plot(np.arange(1, it), f1s_together, label='all together')
plt.legend()
plt.show()

print (max(f1s_pearson))
print (max(f1s_together))
print (max(f1s_chi))
print (max(f1s_spearman))
