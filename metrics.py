import numpy as np

def accuracy(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    return np.mean(y_pred == y_true)

def f1_score(cm):
    TP, FN, FP, TN = cm

    recall = TP / (TP + FN + 1e-15)
    precision = TP / (TP + FP + 1e-15)

    return 2 * precision * recall / (precision + recall + 1e-15)