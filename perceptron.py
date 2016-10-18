import csv
import random
import math
import time
import numpy as np
#from scipy.special import expit as sigmoid

class Perceptron():
    def __init__ (self, maxEpochs = 50, learningRate = 0.001, learningRateDecrease = 0.99 , maxError = 1.5, filename = "", target = 0, nInputs = 28*28, negative = 100, data = []):
        self.maxEpochs = maxEpochs
        self.learningRate = learningRate
        self.learningRateDecrease = learningRateDecrease
        self.filename = filename
        self.target = target
        self.nInputs = nInputs
        self.weights = []
        self.maxError = maxError
        self.negative = negative #negative examples
        self.data = data
        self.dataLen = len(self.data)

    def train(self):
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("\n\nTraining " + str(self.target) + " " + str(time.ctime(time.time())))

        #initiate the weights
        print("Initiate Weights")
        self.weights = self.initWeights(self.nInputs)

        #count classes
        print("Counting Classes")
        self.countClasses()

        print("Normalize Data")
        #self.normalizeData()

        print("Training Model:")

        epochs = 0
        errors = [0, 0]
        while(epochs < self.maxEpochs):
            epochs += 1

            _classes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #targets = [0, 0]

            nSamples = 0
            for i in range(0, self.dataLen):
                #if(iterator.line_num > 1000):
                #    break

                #I need an temporary target variable, because my samples does't have an bias inputs, and we set the target as an input(temporary)
                iteratorTarget = 0
                tempTarget  = int(self.data[i][0])

                #count the number of classes
                _classes[tempTarget] += 1

                if(_classes[tempTarget] > self.nClasses[tempTarget]):
                    continue

                #define the right target for THIS perceptron
                if(tempTarget == self.target):
                    iteratorTarget = 1
                    #targets[0] += 1
                else:
                    iteratorTarget = -1
                    #targets[1] += 1

                #settings bias
                self.data[i][0] = 1

                #linear unit
                output = np.dot(self.data[i], self.weights)

                #update the weights
                for j in range(0, len(self.weights)):
                    self.weights[j] += (self.learningRate * (iteratorTarget - output) * self.data[i][j])

                    if(self.weights[j] > 40000):
                        self.weights[j] = 40000

                    if(self.weights[j] < -40000):
                        self.weights[j] = -40000

                self.data[i][0] = tempTarget

            print("\tEpoch: " + str(epochs) + " " + str(time.ctime(time.time())))

        #calculate the global error
        globalError = self.globalError()
        print("\tError " + str(globalError))

    def initWeights(self, nWeights):
        cont = 0
        weights = []
        while (cont <= nWeights):
            weights.append(round(random.random(), 1) -0.5)
            cont += 1

        #print("weights init: " + str(weights))
        return weights

    def countClasses(self):
        self.nClasses = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        #count classes
        for sample in range(0, self.dataLen):
            self.nClasses[int(self.data[sample][0])] += 1

        #define the number of examples of each class, in this case 10%
        for k in range(0, len(self.nClasses)):
            self.nClasses[k] = int(((self.nClasses[k] * self.negative) / 100))

    def normalizeData(self):
        _classes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #print(self.nClasses)

        for sample in range(0, self.dataLen):
            #get the sample target
            iteratorTarget = int(self.data[sample][0])

            #count the number of classes
            _classes[iteratorTarget] += 1

            if(_classes[iteratorTarget] > self.nClasses[iteratorTarget]):
               continue

            #normalize sample data
            self.data[sample][1:] = np.around(np.divide(self.data[sample][1:],255),4)

    def sigmoid(self, x):
        sig = ((1 / (1 + np.exp(-x))) - 0.5) * 2

        #if(sig < 0.1):
        #    return 0
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

        _classes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for i in range(0, self.dataLen):

            #I need an temporary target variable, because my samples does't have an bias inputs, and we set the target as an input(temporary)
            iteratorTarget = 0
            tempTarget  = int(self.data[i][0])

            #count the number of classes
            _classes[tempTarget] += 1
            if(_classes[tempTarget] > self.nClasses[iteratorTarget]):
                continue

            if(tempTarget == self.target):
                iteratorTarget = 1
            else:
                iteratorTarget = -1

            #set the bias input
            self.data[i][0] = 1

            #linear unit
            output = np.dot(self.data[i], self.weights)

            #set the original target back
            self.data[i][0] = tempTarget

            #increment the error for this sample
            error += (iteratorTarget - output)**2

        error *= 0.5

        return error

    def fitAll(self, filename):
        if(len(self.weights) == 0):
            return "Train the perceptron please."

        print("Fitting Data")

        dataToFit = np.genfromtxt(filename, delimiter=',')

        confusionMatrix = [[0, 0], [0, 0]]

        for i in range(0, len(dataToFit)):

            #get the right target
            iteratorTarget  = int(self.data[i][0])

            if(iteratorTarget == self.target):
                iteratorTarget = 1
            else:
                iteratorTarget = -1

            #set the bias input
            dataToFit[i][0] = 1

            #linear unit
            output = np.dot(self.data[i], self.weights)

            #threshold func
            if(output >= 0):
                output = 1
            else:
                output = -1

            #print(output)
            if(output == iteratorTarget): #positivo ou negativo verdadeiro
                if(output == 1):
                    confusionMatrix[0][0] += 1 #positivo verdadeiro
                else:
                    confusionMatrix[1][1] += 1 #negativo verdadeiro
            else:
                if(output == 1):
                    confusionMatrix[1][0] += 1 #falso positivo
                else:
                    confusionMatrix[0][1] += 1 #falso negativo

            #print(str(output) + " " + str(iteratorTarget))
        return confusionMatrix

    def fit(self, inputs):
        inputs = np.insert(inputs, 0, 1)

        return self.sigmoid(np.dot(inputs, self.weights))

    def saveModel(self):
        _weights = list(map(str, self.weights))
        with open("models/" + str(self.target) + ".csv", 'a') as model:
            writer = csv.writer(model, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(_weights)

    def loadModel(self, path):
        self.weights = np.genfromtxt(path + str(self.target) + ".csv", delimiter=',')

        #print(data)
