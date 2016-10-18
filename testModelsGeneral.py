import copy
import numpy as np
import time
import sys
from perceptron import Perceptron

print("MNIST " + str(time.ctime(time.time())))

perceptrons = []

try:
    #verify what to train
    trainFile = "mnist/mnist_train.csv"
    testFile = "mnist/mnist_test.csv"

    if(len(sys.argv) == 2):
        print("Loading Data " + sys.argv[1])
        trainFile = "mnist/mnist_train" + sys.argv[1] + ".csv"
        testFile = "mnist/mnist_test" + sys.argv[1] + ".csv"
        data = np.genfromtxt(trainFile, delimiter=',')
    else:
        print("Loading Data Default")
        data = np.genfromtxt(trainFile, delimiter=',') #load the default file


    #load models
    print("Loading Models")

    #create and train all the perceptrons
    nPersceptrons = 10
    for x in range(0, nPersceptrons):
        perceptrons.append(Perceptron(filename = trainFile, target = x, data = data))
        perceptrons[x].loadModel("models/")


    #fitting
    print("Loading data to fit")
    dataToFit = np.genfromtxt(testFile, delimiter=',')

    confusionMatrix = [0, 0]

    print("Fitting")
    for sample in dataToFit:
        outputs = []
        for x in range(0, nPersceptrons):
            outputs.append(perceptrons[x].fit(sample[1:]))

        outputTarget = outputs.index(max(outputs))
        print("Target: " + str(sample[0]))
        print(outputs)
        print("Best: " + str(outputTarget))

        if(outputTarget == int(sample[0])):
            confusionMatrix[0] += 1 #hit
        else:
            confusionMatrix[1] += 1 #error

    print(str(confusionMatrix) + " " + str(time.ctime(time.time())))

except IOError:
    print("Train file not found")
