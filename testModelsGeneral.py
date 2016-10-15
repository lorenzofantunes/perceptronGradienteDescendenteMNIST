import copy
import numpy as np
import time
from perceptron import Perceptron

print("MNIST " + str(time.ctime(time.time())))

perceptrons = []

#loading data
print("Loading Data")
data = np.genfromtxt("mnist/mnist_train.csv", delimiter=',')

#load models
print("Loading Models")

#create and train all the perceptrons
nPersceptrons = 10
for x in range(0, nPersceptrons):
    perceptrons.append(Perceptron(filename = "mnist/mnist_train.csv", target = x, data = data))
    perceptrons[x].loadModel("models/")


#fitting
print("Loading data to fit")
dataToFit = np.genfromtxt("mnist/mnist_test.csv", delimiter=',')

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
