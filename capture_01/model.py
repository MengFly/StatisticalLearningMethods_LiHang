import numpy as np


class Perceptron(object):

    def __init__(self, learning_rate=1e-5):
        self.learning_rate = learning_rate
        self.b = 0
        self.W = np.array([])

    def predict(self, X):
        return np.sign(np.dot(X, self.W.T) + self.b)

    def fit(self, X, y, epoch=100):
        if len(self.W) == 0:
            self.W = np.random.random((1, np.shape(X)[-1]))
        for i in range(epoch):
            for t_x, t_y in zip(X, y):
                # 对于错误的分类进行梯度下降
                if (t_y * self.predict(t_x)) < 0:
                    self.W += self.learning_rate * t_x * t_y
                    self.b += self.learning_rate * t_y

    def fit_dual(self, X, y, epoch=100):
        gram_matrix = np.dot(X, np.transpose(X))
        A = np.random.random((1, len(y)))

        for i in range(epoch):
            for i, t_y in enumerate(y):
                if t_y * np.sign(np.sum(np.dot(A * y, gram_matrix[i])) + self.b) <= 0:
                    A[0][i] += self.learning_rate
                    self.b += self.learning_rate
        A = y * A
        self.W = np.sum(np.dot(A, X), axis=0)

    def score(self, X, y):
        predict_y = np.reshape(self.predict(X), np.shape(y))
        return np.sum(predict_y == y) / len(y) * 100
