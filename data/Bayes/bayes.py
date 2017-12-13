from os import listdir, walk
from os.path import isfile, join
import math
import numpy as np

def getSubjectWords(fileName):
    f = open(fileName)
    subject = f.readline().replace('\n', '')
    words = subject.split(' ')
    return words[1:]

def getWordsWithoutSubject(fileName):
    f = open(fileName)
    subject = f.readline() # skip subject
    words = f.read().replace('\n', '').split(' ')
    return words

def getWords(fileName):
    return getSubjectWords(fileName) + getWordsWithoutSubject(fileName) 

def sign(x):
    return 0 if x == 0 else 1 if x > 0 else -1

def checkIsSpam(fileName, pSpam, pLeg, spamDict, legDict, uniqueWordsCount, laplasFactor):
    words = getWords(fileName)
    pSpam = math.log(pSpam)
    pLeg  = math.log(pLeg)
    totalSpamFreq = sum(i[1] for i in spamDict.items())
    totalLegFreq  = sum(i[1] for i in legDict.items())

    for word in words:
        pSpam += math.log(1. * (spamDict.get(word, 0) + laplasFactor) / 
            (uniqueWordsCount * laplasFactor + totalSpamFreq))
        pLeg  += math.log(1. * (legDict.get(word, 0) + laplasFactor) /
            (uniqueWordsCount * laplasFactor + totalLegFreq))
    if (sign(pSpam) != sign(pLeg) or sign(pSpam) == 0 or sign(pLeg) == 0):
        return pSpam > pLeg
    if (sign(pSpam) == 1):
        return pSpam / pLeg > 2
    if (sign(pSpam) == -1):
        return pSpam / pLeg < 0.95

def crossValidation(files, testFiles, laplasFactor):
    spamCount = sum(len(file[0]) for file in files)
    legCount  = sum(len(file[1]) for file in files)
    pSpam = 1.0 * spamCount / (spamCount + legCount)
    pLeg  = 1.0 * legCount  / (spamCount + legCount)
    spamLists  = [x[0] for x in files]
    legLists   = [x[1] for x in files]
    spamFiles  = [inner for outer in spamLists for inner in outer]
    legFiles   = [inner for outer in legLists  for inner in outer]

    spamDict = {}
    legDict = {}
    for spamFile in spamFiles:
        calcFreq(spamFile, spamDict, 1)
    for legFile in legFiles:
        calcFreq(legFile, legDict, 1)
    dict = spamDict.copy()
    dict.update(legDict)
    uniqueWordsCount = len(dict)
    testSpamFiles = testFiles[0]
    testLegFiles  = testFiles[1]
    missCount = 0
    for testSpamFile in testSpamFiles:
        missCount += 0 if checkIsSpam(testSpamFile, pSpam, pLeg, spamDict, legDict, uniqueWordsCount, laplasFactor) else 1
    for testLegFile in testLegFiles:
        missCount += 1 if checkIsSpam(testLegFile, pSpam, pLeg, spamDict, legDict, uniqueWordsCount, laplasFactor) else 0
    total = len(testLegFiles) + len(testSpamFiles)

    def checker(fileName):
        return checkIsSpam(fileName, pSpam, pLeg, spamDict, legDict, uniqueWordsCount, laplasFactor)

    return (checker, 1. * missCount / total)

def calcFreq(fileName, dict, mul):
    words = getWordsWithoutSubject(fileName)
    for w in words:
        dict.setdefault(w, 0)
        dict[w] += 1 * mul
    subjectWords = getSubjectWords(fileName)
    for w in subjectWords:
        dict.setdefault(w, 0)
        dict[w] += 1 * mul
    return words

if __name__ == '__main__':
    files = []
    for part in listdir("pu1"):
        curSpams = []
        curLegs  = []
        for file in listdir("pu1/" + part):
            if "spmsg" in file:
                curSpams.append("pu1/" + part + "/" + file)
            else:
                curLegs.append("pu1/" + part + "/" + file)
        files.append((curSpams, curLegs))
    
    #ks = [1, 2, 5]
    ks = [1]
    t = 1
    for k in ks:
        np.random.shuffle(files)
        result = (None, 1)
        for fr in range(0, len(files), k):
            trainFiles = files[0:fr] + files[fr + k:]
            testPairs  = files[fr:fr+k]
            spamTestFiles = [inner for testPair in testPairs for inner in testPair[0]]
            legTestFiles  = [inner for testPair in testPairs for inner in testPair[1]]
            testFiles = (spamTestFiles, legTestFiles)
            #for laplasFactor in range(1, 10):
            cur = crossValidation(trainFiles, testFiles, 1)
            result = result if result[1] < cur[1] else cur
        print(result[1])


    ll = 0
    ls = 0
    sl = 0
    ss = 0
#28
#
    for testPair in files:
        for spamFile in testPair[0]:
            isSpam = result[0](spamFile)
            if isSpam:
                ss += 1
            else:
                sl += 1
        for legFile in testPair[1]:
            isSpam = result[0](legFile)
            if isSpam:
                ls += 1
            else:
                ll += 1
    print('LL={} LS={}\nSL={} SS={}'.format(ll, ls, sl, ss))
