#normalize the pixels to 0 or 1
import csv
import numpy as np

print("Loading Data")

#data = np.genfromtxt("mnist/mnist_train.csv", delimiter=',')
data = np.genfromtxt("mnist/mnist_train.csv", delimiter=',')

print("Normalizing")
for row in range(0, len(data)):
    for value in range(1, len(data[row])):
        if(data[row][value] < 10):
            data[row][value] = 0
        else:
            data[row][value] = 1

for row in data:
    #print(row)
    with open("mnist/mnist_train0or1.csv", 'a') as mnist:
        writer = csv.writer(mnist, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row)
