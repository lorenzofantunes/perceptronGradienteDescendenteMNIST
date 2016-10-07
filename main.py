import copy
from perceptron import Perceptron

print("Perceptron MNIST")

perceptron = Perceptron(filename = "mnist/mnist_train.csv", target = 0, nInputs = 28*28)
#perceptron = Perceptron(filename = "and.csv", target = 0, nInputs = 2)
perceptron.train()

#print(perceptron.fit("mnist/mnist_test.csv"))

"""iterator = readData("mnist/mnist_train.csv")
iterator2 = iterator.copy()
perceptron = Perceptron()

cont = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for sample in iterator:
    #sample = sample.split(",")
    cont[int(sample[0])] += 1
    #print(sample[0])

somatory = 0
for number in cont:
    somatory += number

print(cont)
print(somatory)"""
#perceptron.train(inputs, targets)
