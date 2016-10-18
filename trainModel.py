import copy
import numpy as np
import time
import sys
from perceptron import Perceptron

print("MNIST " + str(time.ctime(time.time())))

#get the target
toTrain = int(sys.argv[1])

#load data
try:
    #verify what to train
    trainFile = "mnist/mnist_train.csv"
    testFile = "mnist/mnist_test.csv"
    if(len(sys.argv) == 3):
        print("Loading Data " + sys.argv[2])
        trainFile = "mnist/mnist_train" + sys.argv[2] + ".csv"
        testFile = "mnist/mnist_test" + sys.argv[2] + ".csv"
        data = np.genfromtxt(trainFile, delimiter=',')
    else:
        print("Loading Data Default")
        data = np.genfromtxt(trainFile, delimiter=',') #load the default file

    print(trainFile)
    print(testFile)
    #create the perceptron
    perceptron = Perceptron(filename = trainFile, learningRate = 0.001, target = toTrain, nInputs = 28*28, negative = 100, maxEpochs = 50, data = data)

    #train the perceptron
    perceptron.train()

    #save the model into a file
    perceptron.saveModel()

    #fit all test samples
    print(perceptron.fitAll(testFile))
    print(perceptron.weights)
except IOError:
    print("Train file not found")
