import argparse
import sys
import csv
import numpy as np


# Tree
class Node:
    def __init__(self, attr, depth):
        self.left = None
        self.right = None
        self.attr = attr
        self.depth = depth
        self.leaf = None


def getMap(data):
    # map the attribute name with the first value, say this is zero
    attrMap = {}
    numAttr = len(data[0]) - 1
    for i in range(0, numAttr):
        attrMap[data[0][i]] = data[1][i]
    return attrMap


def getTrainData(path):
    # with open(path) as csvfile:
    #     data = np.loadtxt(csvfile, dtype=str, delimiter=',')
    #     # data = list(filereader)
    # return data
    with open(path) as csvfile:
        filereader = csv.reader(csvfile)
        data = list(filereader)
    return data


# modify the data to newdata of 0 and 1, assume left is 0 and right is 1
# def newData(data):
#     data = np.array(data)
#     numAttr = len(data[0]) - 1
#     for i in range(0, numAttr):
#         attrcol = getdata(data, i)
#         firstelem = attrcol[1]
#         for i in range(1, len(attrcol)):
#             if attrcol[i] == firstelem:
#                 attrcol[i] = 0
#             else:
#                 attrcol[i] = 1
#     data = np.invert(data)
#     return data


def newData(data, attrMap):
    # change it to cater with the map
    # firstElems = data[1]
    # for i in range(2, len(data)):
    #     for j in range(0, len(firstElems)-1):
    #         if data[i][j] == firstElems[j]:
    #             data[i][j] = 0
    #         else:
    #             data[i][j] = 1
    # for i in range(0, len(firstElems)-1):
    #     data[1][i] = 0
    # return data
    keyList = []
    for i in range(0, len(data[0])-1):
        keyList += [attrMap[data[0][i]]]
    for i in range(1, len(data)):
        for j in range(0, len(data[0])-1):
            if data[i][j] == keyList[j]:
                data[i][j] = 0
            else:
                data[i][j] = 1
    return data


def train_tree(path, max_depth):
    data = getTrainData(path)
    # old_data = data
    attrMap = getMap(data)
    # modify the data to newdata of 0 and 1, assume left is 0 and right is 1
    new_Data = newData(data, attrMap)
    # print(new_Data)
    # build the tree
    depth = 0
    tree = build(new_Data, depth, max_depth)
    return tree, attrMap


# get data of column i
def getdata(data, i):
    result = []
    for elem in data:
        result += [elem[i]]
    return result


# recursively build the tree, print while build
def build(data, depth, max_depth):
    numAttr = len(data[0]) - 1
    classData = getdata(data, len(data[0])-1)
    [A, B, C, D] = majority_vote(data)
    print("[" + str(A) + " " + str(B) + " / " + str(C) + " " + str(D) + "]")
    # get the best mutual info
    mutualList = []
    for i in range(0, numAttr):
        # curmutual is a tuple of (name, mutualinfo)
        curMutual = mutual_info(getdata(data, i), classData)
        mutualList += [curMutual]
    maxMutual = find_max(mutualList)  # an index
    rootAttr = mutualList[maxMutual][0]
    root = Node(rootAttr, depth)

    # if data is perfectly classified/depth=0/no more attribute/mutualinfo 0:
    if no_split(data, maxMutual, mutualList, root, max_depth):
        root.leaf = majority_vote(data)[-1]
        # print the leaf
        return root
    # else keep recursing
    else:
        dataLeft, dataRight = getLRdata(data, maxMutual)
        # print("DATALEFT", dataLeft)
        # dataLeft = modify(dataLeft, maxMutual)
        # dataRight = modify(dataRight, maxMutual)
        # dataRight = getLRdata(data, maxMutual)[1]
        print("| " * (depth+1) + root.attr + " = " + "0: ", end='')
        left_node = build(dataLeft, depth+1, max_depth)
        root.left = left_node
        print("| " * (depth+1) + root.attr + " = " + "1: ", end='')
        right_node = build(dataRight, depth+1, max_depth)
        root.right = right_node

    return root


def modify(data, maxMutual):
    for elem in data:
        elem.remove(elem[maxMutual])
    return data


def majority_vote2(data):  # return list [len(small),smallname,len(big),label]
    column = getdata(data, len(data[0])-1)
    leftList = []
    rightList = []
    if len(data) == 0:
        print("Got here")
        return [0, "Weird", 0, "Weird"]
    for i in range(1, len(column)):
        if column[i] == column[1]:
            leftList += [column[i]]
        else:
            rightList += [column[i]]
    if len(leftList) == 0 or len(rightList) == 0:
        return [0, "Other", len(column)-1, column[1]]
    if len(leftList) >= len(rightList):
        label = leftList[0]
        smallName = rightList[0]
        smallLen = len(rightList)
        bigLen = len(leftList)
        result = [smallLen, smallName, bigLen, label]
    else:
        label = rightList[0]
        smallName = leftList[0]
        smallLen = len(leftList)
        bigLen = len(rightList)
        result = [smallLen, smallName, bigLen, label]
    return result


def majority_vote(data):  # return list [len(small),smallname,len(big),label]
    column = getdata(data, len(data[0])-1)[1:]
    leftValue = column[0]
    rightValue = "Other"
    for value in column:
        if value != leftValue:
            rightValue = value
            break
    leftCount = len([v for v in column if v == leftValue])
    rightCount = len([v for v in column if v == rightValue])

    bigValue = max(leftCount, rightCount)
    if bigValue == leftCount:
        return (rightCount, rightValue, leftCount, leftValue)
    # Left is smaller
    return (leftCount, leftValue, rightCount, rightValue)


def no_split(data, maxMutual, mutualList, root, max_depth):
    if root.depth == max_depth:
        return True
    elif mutualList[maxMutual][1] <= 0:
        return True
    elif len(mutualList) == 1:
        return True
    elif perfect_classified(data, maxMutual):
        return True
    else:
        return False


def perfect_classified(data, maxMutual):
    classData = getdata(data, len(data[0])-1)
    column = getdata(data, maxMutual)
    leftList = []
    rightList = []
    for i in range(1, len(column)):
        if column[i] == column[1]:
            leftList += [classData[i]]
        else:
            rightList += [classData[i]]
    bool1 = (leftList == leftList[0] * len(leftList))
    bool2 = (rightList == rightList[0] * len(rightList))
    if bool1 and bool2:
        return True
    else:
        return False


def getLRdata(data, maxMutual):
    column = getdata(data, maxMutual)
    leftList = [data[0]]
    rightList = [data[0]]
    for i in range(1, len(column)):
        if column[i] == 0:
            leftList += [data[i]]
        else:
            rightList += [data[i]]
    return (leftList, rightList)


def find_max(ls):   # return an index
    numlist = []
    for elem in ls:
        numlist += [elem[1]]
    maxIndex = numlist.index(max(numlist))
    return maxIndex


# helper to calculate entropy and mutual information
def mutual_info(attrdata, classData):  # return a tuple(name, mutualinfo)
    attrname = attrdata[0]
    attrdata = attrdata[1:]
    classData = classData[1:]
    # print("classData1", classData)
    result = singleEntro(classData) - condEntro(classData, attrdata)
    return (attrname, result)


def singleEntro(data):
    if len(data) == 0:
        return 0
    # print("classData2", data)
    # print("single", data)
    length = len(data)
    class1 = data[0]
    count = 1
    for i in range(1, length):
        if data[i] == class1:
            count += 1
    count2 = length - count
    entropy1 = entropycalc(count, length)
    entropy2 = entropycalc(count2, length)
    entropy = -(entropy1+entropy2)
    return (entropy)


def entropycalc(count, length):
    if count == 0:
        return 0
    else:
        return count/float(length) * np.log2(count/float(length))


def condEntro(classData, attrData):
    length = len(attrData)
    attr1 = attrData[0]
    attr1List = []
    attr2List = []
    # print("COND", attrData)
    for i in range(0, length):
        if attrData[i] == attr1:
            attr1List += [classData[i]]
        else:
            attr2List += [classData[i]]
    condEntro1 = len(attr1List)/length * singleEntro(attr1List)
    condEntro2 = len(attr2List)/length * singleEntro(attr2List)
    return(condEntro1+condEntro2)


# Prediction
def predict_labels(tree, inputpath, attrMap):
    inputData = getTrainData(inputpath)
    new_Data = newData(inputData, attrMap)
    # print(new_Data)
    names = new_Data[0]
    new_Data = new_Data[1:]
    labels = []  # list of strings
    for i in range(len(new_Data)):
        labels += [predict(tree, names, new_Data[i])]
    # switch labels to strings
    # print(labels)
    return labels


def writeTolabels(tree, inputpath, outputpath, attrMap):
    labels = predict_labels(tree, inputpath, attrMap)
    result = ""
    for elem in labels:
        result = result + elem + "\n"
    writefile(outputpath, result)


def predict(tree, names, dataRow):
    if tree.leaf is not None:
        return tree.leaf
    else:
        for i in range(0, len(names) - 1):
            if names[i] == tree.attr:
                index = i
        leftorright = dataRow[index]
        # names.remove(names[index])
        # dataRow.remove((dataRow[index]))
        # print(dataRow)
        if leftorright == 0:
            return predict(tree.left, names, dataRow)
        else:
            return predict(tree.right, names, dataRow)


def writefile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)


def printNode(node, depth=0):
    print("  " * depth, "Node: left {}, right {}, arrt {}, depth {}, leaf {}".format(
        node.left != None, node.right != None, node.attr, node.depth, node.leaf))


def printTree(node, depth=0):
    if node is None:
        return
    printNode(node, depth)
    printTree(node.left, depth+1)
    printTree(node.right, depth+1)


def getclasslabel(path):
    data = getTrainData(path)
    result = []
    for elem in data:
        result += [elem[-1]]
    result = result[1:]
    return result


def writeerrors(train, test, train1, test1, path):
    # train
    count1 = 0
    for i in range(0, len(train1)):
        if train1[i] != train[i]:
            count1 += 1
    error1 = count1 / len(train1)
    # test
    count2 = 0
    for i in range(0, len(test1)):
        if test1[i] != test[i]:
            count2 += 1
    error2 = count2 / len(test1)
    result = "error(train): %f \nerror(test): %f" % (error1, error2)
    writefile(path, result)


if __name__ == "__main__":
    # Arguments Parsing
    # reference: https://docs.python.org/2/library/argparse.html
    parser = argparse.ArgumentParser(description="Decision Tree.")
    parser.add_argument("train_input", type=str, help="train input")
    parser.add_argument("test_input", type=str, help="test input")
    parser.add_argument("max_depth", type=int, help="max depth")
    parser.add_argument("train_out", type=str, help="train out")
    parser.add_argument("test_out", type=str, help="test out")
    parser.add_argument("metrics_out", type=str, help="tmetrics out")
    args = parser.parse_args()
    # print(args.train_input)
    # Invoke actual function
    # data = getTrainData(args.train_input)
    # print(data)
    # attrMap = getMap(data)
    tree, attrMap = train_tree(args.train_input, args.max_depth)
    # printTree(tree)
    trainlabels = getclasslabel(args.train_input)
    testlabels = getclasslabel(args.test_input)
    trainPlabels = predict_labels(tree, args.train_input, attrMap)
    testPlabels = predict_labels(tree, args.test_input, attrMap)
    writeTolabels(tree, args.train_input, args.train_out, attrMap)
    writeTolabels(tree, args.test_input, args.test_out, attrMap)
    writeerrors(trainlabels, testlabels, trainPlabels, testPlabels, args.metrics_out)
