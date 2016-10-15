import copy
import numpy as np
import time
from perceptron import Perceptron

print("MNIST " + str(time.ctime(time.time())))

perceptrons = []

#load data
print("Loading Data\n")
data = np.genfromtxt("mnist/mnist_train.csv", delimiter=',')

#create and train all the perceptrons
#for x in range(0, 10):

nPersceptrons = 10
for x in range(0, nPersceptrons):
    perceptrons.append(Perceptron(filename = "mnist/mnist_train.csv", learningRate = 0.001, target = x, nInputs = 28*28, negative = 100, maxEpochs = 50, data = data))
    perceptrons[x].train()

#print(perceptron.fitAll("mnist/mnist_test.csv"))

dataToFit = np.genfromtxt("mnist/mnist_test.csv", delimiter=',')

confusionMatrix = [0, 0]

for sample in dataToFit:
    outputs = []
    for x in range(0, nPersceptrons):
        outputs.append(perceptrons[x].fit(sample[1:]))

    outputTarget = outputs.index(max(outputs))

    if(outputTarget == int(sample[0])):
        confusionMatrix[0] += 1 #hit
    else:
        confusionMatrix[1] += 1 #error

print(str(confusionMatrix) + " " + str(time.ctime(time.time())))
