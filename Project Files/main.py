import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def inRange(number: float, minimum: float, maximum: float):
    if (minimum <= number <= maximum):
        return True
    return False


def processInput(filename):
    tempList = []
    fin = open(filename, mode='r', encoding='utf-8')
    testLine = fin.readline()
    while (testLine != ''):
        tempList.append(re.split(',', testLine))
        testLine = fin.readline()
    fin.close()
    return tempList


def calculateEntropy(listName: list, uniqueClassLabels: set):
    result = 0
    if len(listName) == 0:
        return 0
    for j in uniqueClassLabels:
        numClassLabel = 0
        for i in listName:
            if (i[-1] == j):
                numClassLabel += 1
        if (numClassLabel == 0):
            return 0
        PsubI = numClassLabel / len(listName)
        result += -1 * PsubI * np.log2(PsubI)
    return result


def calculateGain(indexOfFeature: int, binArray: list, entropyOfSet: float, dataList: list, uniqueClassLabels):
    # get a list of bins for a certain feature
    localBins = binArray[indexOfFeature]
    indicesInBins = []
    summation = 0
    for bins in localBins:
        indicesInThisBin = []
        index = 0
        for features in dataList:
            if inRange(float(features[indexOfFeature]), float(bins[0]), float(bins[1])):
                indicesInThisBin.append(index)
            index += 1
        indicesInBins.append(indicesInThisBin)
    # data is now sorted into bins. indices of respective elements are stored in a list for that specific bin
    tempList = []
    for binIndices in indicesInBins:
        for index in binIndices:
            tempList.append(dataList[index])  # building the sublist to calculate Entropy(S_v)
        summation += (len(binIndices) / len(dataList)) * calculateEntropy(tempList, uniqueClassLabels)

    return (entropyOfSet - summation), indicesInBins


def calculateBins(dataSubSet): # returns [[min,max],[min2,max2]...[min,max]]
    # calculate the min and max values along each axis, separate into bins
    bins = []
    for i in range(0, len(dataSubSet[0]) - 1):  # number of features including class label in a sub list
        tempList = []
        binsForFeature = []
        for j in dataSubSet:
            tempList.append(float(j[i]))
        absMinimum = min(tempList)
        absMaximum = max(tempList)

        binRange = absMaximum - absMinimum
        binDistance = binRange / numberOfBins

        for k in range(0, numberOfBins):
            binsForFeature.append([absMinimum, absMinimum + binDistance])
            absMinimum += binDistance
        bins.append(binsForFeature)
    return bins
    # equidistant bins have been calculated according to a total number of bins


class treeNode:
    def __init__(self):
        self.myClassLabel = None
        self.children = []
        self.splitIndex = None
        self.bin = []

    def addChild(self, childNode, myBin):
        childNode.bin = myBin
        self.children.append(childNode)

def IC3(dataSubSet):
    if (len(dataSubSet) == 0):
        return treeNode()
    # make a list of all of the class labels
    classLabelList = []
    for featureList in dataSubSet:
        if len(featureList) == 0:
            return treeNode()
        classLabelList.append(featureList[-1])
    uniqueClassLabels = set(classLabelList)

    #base case -- if the class label is all the same
    if len(uniqueClassLabels) == 1:
        leafNode = treeNode()
        leafNode.myClassLabel = uniqueClassLabels.pop()
        return leafNode
    #base case -- if there are no remaining attributes to split on
    if len(dataSubSet[0]) == 1:  # just a class label left over
        remainingLabelCount = []
        for label in uniqueClassLabels:
            remainingLabelCount.append(classLabelList.count(label))
        dominatingLabel = remainingLabelCount.index(max(remainingLabelCount))
        leafNode = treeNode()
        leafNode.myClassLabel = dominatingLabel
        return leafNode
    # return the class label with higher occurance

    entropyOfSet = calculateEntropy(dataSubSet, uniqueClassLabels)
    featureBins = calculateBins(dataSubSet)

    gainMap = dict()  # used to store the gain for each split
    for i in range(0, len(dataSubSet[0]) - 1):
        gainMap[i] = calculateGain(i, featureBins, entropyOfSet, dataSubSet, uniqueClassLabels)
    gainMap = sorted(gainMap.items(), key=lambda x: x[1], reverse=True)
    # now features are sorted in order of information gain [(featureIndex, (gain, indicesInBins)), (feature2index,
    # (gain, indicesInBins))...]

    rootNode = treeNode()
    featureInfo = gainMap[0]
    rootNode.splitIndex = featureInfo[0]
    for binIndices in ((featureInfo[1])[1]):
        subDataSet = []
        for index in binIndices:
            splitList = dataSubSet[index]
            del splitList[featureInfo[0]]
            subDataSet.append(splitList) # create the subset of data!
        indexOfBin = ((featureInfo[1])[1]).index(binIndices)
        childBin = (featureBins[featureInfo[0]])[indexOfBin]
        rootNode.addChild(IC3(subDataSet), childBin)    #RECURSIVE CALL
    return rootNode


def printDecisionBoundaries(node, parentBin, patchList, thisSplit):
    if (len(node.children) == 0) and node.myClassLabel is not None and parentBin == []:  # split along one dimension
            if thisSplit == 0:
                bottom = node.bin[0]
                height = node.bin[1] - node.bin[0]
                left, right = plt.xlim()
                width = right - left
                if node.myClassLabel == '0\n':
                    patchList.append(Rectangle((bottom, left), width, height, color='#0000ff', angle=90))
                else:
                    patchList.append(Rectangle((bottom, left), width, height, color='#ff0000', angle=90))
            else:
                left = node.bin[1]
                width = node.bin[1] - node.bin[0]
                bottom, top = plt.ylim()
                height = top - bottom
                if node.myClassLabel == '0\n':
                    patchList.append(Rectangle((bottom, left), width, height, color='#0000ff', angle=270))
                else:
                    patchList.append(Rectangle((bottom, left), width, height, color='#ff0000', angle=270))
    elif len(node.children) == 0 and node.myClassLabel is not None and parentBin != []:  # leaf node, split along two dimensions
        if thisSplit == 0:
            leftCoord = parentBin[1]
            width = parentBin[1] - parentBin[0]
            bottomCoord = node.bin[0]
            height = node.bin[1] - node.bin[0]
            if node.myClassLabel == '0\n':
                patchList.append(Rectangle((bottomCoord, leftCoord), width, height, color='#0000ff', angle=270))
            else:
                patchList.append(Rectangle((bottomCoord, leftCoord), width, height, color='#ff0000', angle=270))
        else:
            leftCoord = node.bin[0]
            width = node.bin[1] - node.bin[0]
            bottomCoord = parentBin[0]
            height = parentBin[1] - parentBin[0]
            if node.myClassLabel == '0\n':
                patchList.append(Rectangle((bottomCoord, leftCoord), width, height, color='#0000ff', angle=0.0))
            else:
                patchList.append(Rectangle((bottomCoord, leftCoord), width, height, color='#ff0000', angle=0.0))
    elif len(node.children) == 0 and node.myClassLabel is None and parentBin == []:
        return
    else: # intermediate node, pass down the parent bin and let child nodes print a rectangle
        for childNode in node.children:
            printDecisionBoundaries(childNode, node.bin, patchList, node.splitIndex)

def printDecisionTree(decisionTree, dataToPlot):
    xList = []
    yList = []
    colors = []
    for point in dataToPlot:
        xList.append(float(point[0]))
        yList.append(float(point[1]))
        if(point[2] == '1\n'):
            colors.append('#ff0000')
        if(point[2] == '0\n'):
            colors.append('#0000ff')
    ax = plt.gca()
    plt.scatter(xList,yList, c = colors)
    plt.grid(False)
    plt.title('Surface Visualization 20 Bins -- synthetic-4.csv')
    plt.xlabel('X Value')
    plt.ylabel('Y Value')
    patchList = []
    printDecisionBoundaries(decisionTree,[],patchList,None)
    pc = PatchCollection(patchList,match_original=True,alpha=0.5)
    ax.add_collection(pc)
    ax.autoscale_view()
    plt.show()

def calculateAccuracy(decisionTree, points):
    numberCorrectlyClassified = 0
    for point in points: # for each feature set
        classLabel = point[-1]
        x = float(point[0])
        y = float(point[1])
        for node in decisionTree.children: # traverse the tree, check if there is a further split
            if len(node.bin) != 0 and len(node.children) == 0:
                if(inRange(x,node.bin[0],node.bin[1])): # no further split, look for class label
                    classLabelGuess = node.myClassLabel
                    if(classLabelGuess == classLabel):
                        numberCorrectlyClassified += 1
                        break
            else:                                       # further split, check next feature value
                for childNode in node.children:
                    if(inRange(y,childNode.bin[0],childNode.bin[1])):
                        classLabelGuess = childNode.myClassLabel
                        if(classLabelGuess == classLabel):
                            numberCorrectlyClassified += 1
                            break
    return numberCorrectlyClassified




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    global numberOfBins
    numberOfBins = 16

    pokemonFeatures = processInput('pokemonStats.csv')
    pokemonLabels = processInput('pokemonLegendary.csv')

    for i in range(len(pokemonFeatures)):
        pokemonFeatures[i] = (pokemonFeatures[i])[0:2]
        if (pokemonLabels[i])[0] == "False\n":
            pokemonFeatures[i].append('0\n')
        else:
            pokemonFeatures[i].append('1\n')

    pokemonFeaturesCopy = processInput('pokemonStats.csv') # To save on stack memory, IC3 is pass by reference. Use separate copy of data for Accuracy metrics and visualization.
    for i in range(len(pokemonFeaturesCopy)):
        pokemonFeaturesCopy[i] = (pokemonFeaturesCopy[i])[0:2]
        if (pokemonLabels[i])[0] == "False\n":
            pokemonFeaturesCopy[i].append('0\n')
        else:
            pokemonFeaturesCopy[i].append('1\n')
    tree = IC3(pokemonFeatures)

    numberCorrect = calculateAccuracy(tree, pokemonFeaturesCopy)

    print("Number Correct: " + str(numberCorrect))
    print("Total Guesses: " + str(len(pokemonLabels)))
    print("Accuracy: " + str(numberCorrect / len(pokemonLabels)))

    # For a demo of visualization uncomment this code
    # dataList = processInput('synthetic-4.csv')
    # dataListCopy = processInput('synthetic-4.csv')
    # tree = IC3(dataList)
    # printDecisionTree(tree, dataListCopy)