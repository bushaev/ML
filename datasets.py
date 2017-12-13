import numpy as np
from os import listdir
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

class Dataset:
    def __init__(self, data, shuffle=True, dtype=float):
        self.data = np.asarray(data, dtype=dtype)
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.data)

    def CV(self, k = 5):
        if self.shuffle:
            np.random.shuffle(self.data)

        n = round(len(self.data) / k)
        for i in range(k - 1):
            yield np.concatenate((self.data[:i * n], self.data[(i + 1) * n:])), self.data[i * n: (i + 1) * n]
        yield self.data[:(i + 1) * n], self.data[(i + 1) * n:]

    def plot(self):
        x, y, c = self.data[:, 0], self.data[:, 1], self.data[:, 2]
        plt.plot(x[c == 0], y[c == 0], 'ro', x[c == 1], y[c == 1], 'bo')
        plt.show()

class SpamDataset:
    def __init__(self):
        parts = listdir('data/Bayes/pu1')
        self.parts = []

        for part in parts:
            files = listdir('data/Bayes/pu1/' + part)
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

def read_chips():
    with open('data/chips.txt', 'r') as f:
        sl = [line.split(',') for line in f]
        p = np.array([[x, y, int(c[0])] for x,y, c in sl])
    return Dataset(p)

def read_spam():
    return SpamDataset()

def read_prices():
    with open('data/prices.txt', 'r') as f:
        X = []
        Y = []
        f.readline()
        for line in f:
            area, room, price = line.strip().split(',')
            X.append([area, room])
            Y.append(price)

    return np.asarray(X, dtype=float), np.asarray(Y, dtype=float)

def read_arcene(type='train'):
    X = []
    y = []
    with open(f'data/arcene/arcene_{type}.data') as f:
        for line in f:
            X.append(line.split())

    with open(f'data/arcene/arcene_{type}.labels') as f:
        for line in f:
            y.append(int(line))

    y = LabelBinarizer().fit_transform(y)
    return np.array(X, dtype=np.float32), np.ravel(y)

