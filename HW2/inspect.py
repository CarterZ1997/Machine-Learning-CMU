import sys
import csv
import numpy as np
import math
import code

def readfromcsv():
    input = sys.argv[1]
    output = sys.argv[2]
    with open(input) as csvfile:
        filereader = csv.reader(csvfile)
        filelist = list(filereader)
    #print(filelist)
    eAnde = entropyAndError(filelist)
    result = "entropy: %f \nerror: %f" % (eAnde[0], eAnde[1])
    writefile(output, result)

def read_from_csv(path):
    with open(path) as csvfile:
        filereader = csv.reader(csvfile)
        data = list(filereader)
    return data

    # eAnde = entropyAndError(data)
    # result = "entropy: %f \nerror: %f" % (eAnde[0], eAnde[1])

def writefile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)

def entropyAndError(ls):
    itemlen = len(ls[0])
    class1 = ls[1][itemlen-1]
    count = 1
    for i in range(2, len(ls)-1):
        if ls[i][itemlen-1] == class1:
            count += 1
    count2 = len(ls) - 1 - count
    entropy1 = count/float(len(ls)-1) * np.log2(count/float(len(ls)-1))
    #print(entropy1)
    # code.interact(local=locals())
    entropy2 = count2/float(len(ls)-1) * np.log2(count2/float(len(ls)-1))
    entropy = -(entropy1+entropy2)
    if (count >= count2):
        error = count2 / float(len(ls)-1)
    else:
        error = count / float(len(ls)-1)
    return (entropy, error)

readfromcsv()