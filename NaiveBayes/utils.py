from os import listdir
import numpy as np
import itertools
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self):
        parts = listdir('Bayes/pu1')
        self.parts = []

        for part in parts:
            files = listdir('Bayes/pu1/' + part)
            spam = np.array([part + '/' + file for file in files if file.find('spmsg') > 0])
            legit = np.array([part + '/' + file for file in files if file.find('legit') > 0])

            self.parts.append(np.array([legit, spam]))

        self.parts = np.array(self.parts)

    def CV(self):
        for test_ind in range(len(self.parts)):
            test = self.parts[test_ind]
            spam_train = []
            legit_train = []

            for train_ind in range(test_ind):
                legit_train = np.concatenate([legit_train, self.parts[train_ind][0]])
                spam_train = np.concatenate([spam_train, self.parts[train_ind][1]])

            for train_ind in range(test_ind + 1, len(self.parts)):
                legit_train = np.concatenate([legit_train, self.parts[train_ind][0]])
                spam_train = np.concatenate([spam_train, self.parts[train_ind][1]])

            yield (legit_train, spam_train), test

    def data(self):
        data_spam = []
        data_legit = []

        for part in self.parts:
            data_spam = np.concatenate([data_spam, part[1]])
            data_legit = np.concatenate([data_legit, part[0]])

        return data_legit, data_spam

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

def F1(cm):
    TN, FP, FN, TP = cm

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)

    return 2 * precision * recall / (precision + recall)
