from operator import itemgetter

import numpy as np


class KDTree(object):
    def __init__(self, values):
        # flatten elements, for next split data and build kdNode
        self.values = np.array([element.flatten() for element in np.array(values)])
        self.max_split_dim = np.shape(values)[1]
        self.root_node = self.build_node(self.values, None, 0)

    def build_node(self, values, parent_node, depth):
        if len(values) == 0:
            return None
        medium_index = len(values) // 2
        axis = depth % self.max_split_dim
        values_sort = sorted(values, key=itemgetter(axis))
        node_value = values_sort[medium_index]

        node = KDNode(node_value, axis, depth)
        node.parent_node = parent_node
        node.left_node = self.build_node(values_sort[:medium_index], node, depth + 1)
        node.right_node = self.build_node(values_sort[medium_index + 1:], node, depth + 1)
        return node

    def __str__(self):
        return str(self.root_node)


class KDNode(object):

    def __init__(self, value, axis, depth):
        """
        :param value: Node contains value
        :param axis: axis of split dime
        :param depth node depth at kdTree
        """
        self.value = value
        self.axis = axis
        self.left_node = None
        self.right_node = None
        self.parent_node = None
        self.depth = depth

    def __str__(self):
        str_child_padding = " " * self.depth * 2
        self_str = "{}(axis={})".format(self.value, self.axis)
        if self.left_node:
            self_str += "\n  {}|_ Left : {}".format(str_child_padding, self.left_node)
        if self.right_node:
            self_str += "\n  {}|_ Right : {}".format(str_child_padding, self.right_node)
        return self_str

    def is_leaf(self):
        return self.left_node is None and self.right_node is None

    def is_root(self):
        return self.parent_node is not None


if __name__ == '__main__':
    test_arr = ((2, 3, 6), (5, 4, 8), (9, 6, 1), (4, 7, 2), (8, 1, 5), (7, 2, 4))
    kd_tree = KDTree(test_arr)
    print(kd_tree)

    test_arr = ((2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2))
    kd_tree = KDTree(test_arr)
    print(kd_tree)
