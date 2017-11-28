import numpy as np

class NaiveBayes:
    def __init__(self, n_cat, getfeatures):
        self.n_cat = n_cat
        self.freq = np.array([{} for _ in range(n_cat)])
        self.n = np.zeros(n_cat)
        self.get_features = getfeatures

    def update_freq(self, feature, category):
        assert(category < self.n_cat)
        self.freq[category].setdefault(feature, 0)
        self.freq[category][feature] += 1

    def get_freq(self, feature, category):
        self.freq[category].setdefault(feature, 0)
        return self.freq[category][feature]

    def train_(self, X, y):
        for ind, x in enumerate(X):
            features = self.get_features(x)

            for f in features:
                self.update_freq(f, y[ind])

            self.n[y[ind]] += 1

    def Pr(self, y):
        return self.n[y] / sum(self.n)

    def scores(self, x):
        features = self.get_features(x)
        Prs = [self.Pr(y) for y in range(self.n_cat)]

        for y in range(self.n_cat):
            for f in features:
                Prs[y] += np.log(self.get_freq(f, y) / self.n[y] + 1e-15)

        return Prs

    def test(self, X):
        return np.array([self.predict(x) for x in X])

class SpamClissifier(NaiveBayes):
    def __init__(self):
        super(SpamClissifier, self).__init__(2, self.get_features)
        self.threshold = 0

    def get_features(self, filename):
        with open('Bayes/pu1/' + filename) as f:
            words = f.read().replace('\n', ' ').split()[1:]
            numbers = [int(w) for w in words]

        return np.array(numbers)

    def train(self, data):
        self.train_(data[0], np.zeros(len(data[0]), dtype=int))
        self.train_(data[1], np.ones(len(data[1]), dtype=int))

    def predict(self, x):
        p_leg, p_spam = self.scores(x)
        return p_spam - p_leg > self.threshold

    def score(self, x):
        p_leg, p_spam = self.scores(x)
        return p_spam - p_leg

    def contingency_table(self, data):
        legit_p = self.test(data[0])
        spam_p = self.test(data[1])

        TL = sum(legit_p == 0)
        FL = sum(legit_p == 1)
        TS = sum(spam_p == 1)
        FS = sum(spam_p == 0)

        return np.array([[TL, FL], [FS, TS]])


