import copy
import numpy as np
import time
import sys
from perceptron import Perceptron

print("MNIST " + str(time.ctime(time.time())))

#get the target
toTrain = int(sys.argv[1])

#load data
print("Loading Data\n")
data = np.genfromtxt("mnist/mnist_train.csv", delimiter=',')

#create the perceptron
perceptron = Perceptron(filename = "mnist/mnist_train.csv", learningRate = 0.001, target = toTrain, nInputs = 28*28, negative = 100, maxEpochs = 50, data = data)

#train the perceptron
perceptron.train()

#save the model into a file
perceptron.saveModel()

#fit all test samples
print(perceptron.fitAll("mnist/mnist_test.csv"))
