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

    print("Digito: " + str(x))
    confusion = perceptrons[x].fitAll("mnist/mnist_test.csv")
    print("True Positive: " + str(confusion[0][0]))
    print("True Negative: " + str(confusion[1][1]))
    print("False Positive: " + str(confusion[1][0]))
    print("False Negative: " + str(confusion[0][1]))
    print("Taxa de Acerto: " + str((confusion[0][0] + confusion[1][1])/100))
