import random
import math
import numpy as np
#from scipy.special import expit as sigmoid

from readData import readData

class Perceptron():
    def __init__ (self, maxEpochs = 200000, learningRate = 0.1, learningRateDecrease = 0.95 , maxError = 1.5, filename = "", target = 0, nInputs = 0):
        self.maxEpochs = maxEpochs
        self.learningRate = learningRate
        self.learningRateDecrease = learningRateDecrease
        self.filename = filename
        self.target = target
        self.nInputs = nInputs
        self.weights = []
        self.maxError = maxError

    def train(self):
        #initiate the weights
        self.weights = self.initWeights(self.nInputs)

        self.data = np.genfromtxt(self.filename, delimiter=',')
        self.dataLen = len(self.data)

        #define the number of examples of each class, in this case 10%
        self.nClasses = self.countClasses()
        for k in range(1, len(self.nClasses)):
            self.nClasses[k] = int(((self.nClasses[k] * 5) / 100))

        print(self.nClasses[0])
        print(sum(nClasses[1:]))

        epochs = 0
        while(epochs < self.maxEpochs):
            epochs += 1

            _classes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #targets = [0, 0]

            nSamples = 0
            for i in range(0, self.dataLen):
                #if(iterator.line_num > 1000):
                #    break

                iteratorTarget = self.data[i][0]

                #count the number of classes
                _classes[iteratorTarget] += 1

                if(_classes[int(iteratorTarget)] > self.nClasses[int(iteratorTarget)]):
                    continue

                #nSamples += 1
                #print(nSamples)

                #define the right target for THIS perceptron
                if(iteratorTarget == self.target):
                    iteratorTarget = 1
                    #targets[0] += 1
                else:
                    iteratorTarget = -1
                    #targets[1] += 1

                #set the initial value of the bias
                self.data[i][0] = 1

                #calculate the output for the linear unit
                output = self.sigmoid(np.dot(self.data[i], self.weights))

                #update the weights
                for j in range(0, len(self.weights)):
                    self.weights[j] += (self.learningRate * (iteratorTarget - output) * self.data[i][j])

            #print(targets)
            #calculate the global error
            error = self.globalError()

            print("Epoch: " + str(epochs) + " Error: " + str(error))

            #self.learningRate *= self.learningRateDecrease

            #if()

    def initWeights(self, nWeights):
        cont = 0
        weights = []
        while (cont <= nWeights):
            weights.append(round(random.random(), 1))
            cont += 1

        #print("weights init: " + str(weights))
        return weights

    def countClasses(self):
        classes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for i in range(0, self.dataLen):
            classes[int(self.data[i][0])] += 1

        return classes

    def sigmoid(self, x):
        sig = 1 / (1 + np.exp(-x))

        if(sig < 0.1):
            return 0
        """if(x > 5000):
            return 1

        if(x < -5000):
            return 0

        if "-" in str(sig):
            return 0

        if "+" in str(sig):
            return 1"""

        return sig

    def globalError(self):
        error = 0.0

        #create the iterator
        iterator = readData(self.filename)
        _classes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for i in range(0, self.dataLen):

            #define the right target for THIS perceptron
            iteratorTarget = self.data[i][0]

            #count the number of classes
            _classes[iteratorTarget] += 1
            if(_classes[int(iteratorTarget)] > self.nClasses[int(iteratorTarget)]):
                continue

            if(iteratorTarget == self.target):
                iteratorTarget = 1
            else:
                iteratorTarget = -1

            #set the bias input
            self.data[i][0] = 1

            output = self.sigmoid(np.dot(self.data[i], self.weights))
            #print(output)
            #output = self.sigmoid(output)
            #print(output)

            error += (iteratorTarget - output)**2

        error *= 0.5

        return error

    """def fit(self, filename):
        if(len(self.weights) == 0):
            return "Train the perceptron please."

        iterator = readData(filename)

        confusionMatrix = [0, 0, 0, 0]
        self.target

        for sample in iterator:
            output = 0.0
            sampleTarget = sample[0]

            sample[0] = 1

            for _input in range(0, len(sample)):
                output += self.weights[_input] * int(sample[_input])

            y = self.thresholdFunc(output)

            if(self.target == "1"):
                if(y == 1):
                    confusionMatrix[0] += 1 #positivo verdadeiro
                else:
                    confusionMatrix[1] += 1 #falso positivo
            else:
                if(y == 1):
                    confusionMatrix[3] += 1 #falso negativo
                else:
                    confusionMatrix[2] += 1 #negativo verdadeiro

        return confusionMatrix
"""
