import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.preprocessing import StandardScaler

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
    mlp = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(10, 8), random_state=1,
                        max_iter=3000, learning_rate_init=.01)
    mlp.fit(X_training, Y_training)
    Y_pred = mlp.predict(X_test)
    # print(Y_pred)
    # print("---------")
    # print(Y_test)

    print("Error rate", 1-mlp.score(X_test, Y_test))
    print(mlp.n_layers_)
    print(mlp.n_iter_)
    print(mlp.loss_curve_)
    plt.plot(mlp.loss_curve_)
    plt.xlabel('Training')
    plt.ylabel('costs')
    plt.show()
