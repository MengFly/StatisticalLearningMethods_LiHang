from capture_01.model import Perceptron
from util import *

x, y = read_data()
train_idx, test_idx = split_train(len(y))

model = Perceptron(learning_rate=1e-2)

# test
model.fit(x=x[train_idx], y=y[train_idx], epoch=10)
print("model accuracy is %.2f %%" % model.score(x[test_idx], y[test_idx]))
plot_decision_regions(x, y, model, test_idx=test_idx)


model.fit_dual(x=x[train_idx], y=y[train_idx], epoch=100)
print("model accuracy is %.2f %%" % model.score(x[test_idx], y[test_idx]))
plot_decision_regions(x, y, model, test_idx=test_idx)