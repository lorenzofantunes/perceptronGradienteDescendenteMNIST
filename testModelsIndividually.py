import copy
import numpy as np
import time
import sys
from perceptron import Perceptron

print("MNIST " + str(time.ctime(time.time())))

perceptrons = []

#load data
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

        print("Normalizing")
        #perceptrons[x].normalizeData()
        perceptrons[x].loadModel("models/")

        print("Digito: " + str(x))
        confusion = perceptrons[x].fitAll(testFile)
        print(confusion)
        print("True Positive: " + str(confusion[0][0]))
        print("True Negative: " + str(confusion[1][1]))
        print("False Positive: " + str(confusion[1][0]))
        print("False Negative: " + str(confusion[0][1]))
        print("Taxa de Acerto: " + str((confusion[0][0] + confusion[1][1])/100))

except IOError:
    print("Train file not found")
