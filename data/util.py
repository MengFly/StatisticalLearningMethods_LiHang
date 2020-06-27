import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


def read_data():
    df = pd.read_csv("../resource/iris.data", header=None)
    y = df.iloc[:100, 4].values
    y = np.where(y == "Iris-setosa", -1, 1)
    X = df.iloc[:100, [0, 2]].values
    return X, y


def split_train(len_data, train_scale=0.7):
    idx = np.arange(len_data)
    np.random.shuffle(idx)
    train_size = int(train_scale * len_data)
    return idx[:train_size], idx[train_size:]


def stadardization(X):
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    return X_std


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02, xlabel=('x', 'y')):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    meshXX = np.array([xx1.ravel(), xx2.ravel()]).T
    Z = classifier.predict(meshXX)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.xlabel(xlabel[0])
    plt.ylabel(xlabel[1])

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    if test_idx is not None:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],color='', marker='o',edgecolors='black', s=100, label='test set')
    plt.legend(loc='upper right')
    plt.show()
