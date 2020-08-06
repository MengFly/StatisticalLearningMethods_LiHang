from capture_02.tree import KDTree


class KNN(object):
    def __init__(self, n):
        self.kd_tree = None
        self.n = n

    def fit(self, xs):
        self.kd_tree = KDTree(xs)

    def predict(self, x):
        if not self.kd_tree:
            raise RuntimeError("not fit any data")
