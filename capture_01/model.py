import numpy as np


class Perceptron(object):

    def __init__(self, learning_rate=1e-5):
        self.learning_rate = learning_rate
        self.b = 0
        self.W = np.array([])

    def predict(self, x):
        return np.sign(np.dot(x, self.W.T) + self.b)

    def fit(self, x, y, epoch=100):
        if len(self.W) == 0:
            self.W = np.random.random((1, np.shape(x)[-1]))
        for i in range(epoch):
            for t_x, t_y in zip(x, y):
                # 对于错误的分类进行梯度下降
                if (t_y * self.predict(t_x)) < 0:
                    self.W += self.learning_rate * t_x * t_y
                    self.b += self.learning_rate * t_y

    def fit_dual(self, x, y, epoch=100):
        gram_matrix = np.dot(x, np.transpose(x))
        a = np.random.random((1, len(y)))

        for i in range(epoch):
            for i, t_y in enumerate(y):
                if t_y * np.sign(np.sum(np.dot(a * y, gram_matrix[i])) + self.b) <= 0:
                    a[0][i] += self.learning_rate
                    self.b += self.learning_rate
        a = y * a
        self.W = np.sum(np.dot(a, x), axis=0)

    def score(self, x, y):
        predict_y = np.reshape(self.predict(x), np.shape(y))
        return np.sum(predict_y == y) / len(y) * 100
