import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def dataProcessing(data):
    dataMat = []
    labelMat = []
    for i in range(data.iloc[:, 0].size):
        dataMat.append([1.0, float(data.iloc[i, 0]), float(data.iloc[i, 1]), float(data.iloc[i, 2]), float(data.iloc[i, 3])])
        if data.iloc[i, 4] == 'Iris-setosa':
            labelMat.append(1)
        elif data.iloc[i, 4] == 'Iris-versicolor':
            labelMat.append(2)
        else:
            labelMat.append(3)
    return dataMat, labelMat

class Neural_Network(object):
    def __init__(self):
        self.w1 = np.random.random((5, 10))
        self.w2 = np.random.random((10, 3))
        self.costs = []
        # print(self.w1, self.w2)

    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return np.multiply(z, (1-z))

    def crossEntropy(self, Y, o):
        cost= -(np.multiply(Y, np.log(o)) + np.multiply((1 - Y), np.log(1 - o)))
        # print("cost:",cost)
        return np.sum(cost)

    def forward(self, X):

        self.z1 = np.dot(X, self.w1)
        self.z2 = self.sigmoid(self.z1)
        self.z3 = np.dot(self.z2, self.w2)
        o = self.sigmoid(self.z3)
        # print("----------z1------------:", self.z1)
        # print("----------z2------------:", self.z2)
        # print("----------z3------------:", self.z3)
        # print("---------- o ------------:", o)
        return o

    def backward(self, X, Y, o):
        self.o_error = o - Y
        self.o_delta = np.multiply(self.o_error,self.sigmoid_derivative(o))
        self.z2_error = self.o_delta.dot(self.w2.T)
        self.z2_delta = np.multiply(self.z2_error, self.sigmoid_derivative(self.z2))
        self.w1 -= X.T.dot(self.z2_delta)*0.01
        self.w2 -= self.z2.T.dot(self.o_delta)*0.01
        cost = self.crossEntropy(Y, o)
        self.costs.append(cost)

    def predict(self, X):
        return  self.forward(X)


if __name__ == "__main__":
    iris = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    #print(iris)
    X, Y = dataProcessing(iris)
    X = np.mat(X)
    Y = pd.get_dummies(Y).values
    Y = Y.astype(np.float64)
    # print(X)
    # print(Y)
    X_training, X_test, Y_training, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    # print(X_training)
    # print(Y_training)
    NN = Neural_Network()
    for i in range(3000):
        o = NN.forward(X_training)
        NN.backward(X_training, Y_training, o)
    print(NN.costs)
    plt.plot(NN.costs)
    plt.xlabel('Training')
    plt.ylabel('costs')
    plt.show()

    pre = NN.predict(X_test)
    pre = np.argmax(pre, axis=1)
    error = 0
    for i in range(Y_test[:, 0].size):
        index = pre[i, 0]
        if Y_test[i, index] == 0:
            error += 1
    accuracy =  (Y_test[:, 0].size - error)/Y_test[:, 0].size
    print(pre)
    print(Y_test)
    print("Error: ",error)
    print("Error rate: ",np.round(1-accuracy, 2))


