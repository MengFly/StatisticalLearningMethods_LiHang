import numpy as np
class itemgetter(object):
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, item):
        return item[1][self.axis]


a = list(enumerate([[1, 2], [3, 2], [1, 4]]))

print(sorted(a, key=itemgetter(0)))
print(sorted(a, key=itemgetter(1)))
print(a)
