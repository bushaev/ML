from math import inf

from utils import *
from naive import *

data = Dataset()
ct = np.zeros(shape=(2, 2), dtype=int)

for train, test in data.CV():
    model = SpamClissifier()
    model.train(train)
    ct += model.contingency_table(test)

plot_confusion_matrix(ct, ['legit', 'spam'])
print ("F score with threshold = 0", F1(np.ravel(ct)))