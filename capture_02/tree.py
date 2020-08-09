from queue import Queue

import numpy as np


class KDTree(object):

    def __init__(self, values):
        # flatten elements, for next split data and build kdNode
        # 为每一个item添加index信息，方便后面对item进行查找的时候定位到在原始数组中的位置
        flatten_values = np.array([element.flatten() for element in np.array(values)])
        index_values = list(enumerate(flatten_values))

        self.max_split_dim = np.shape(flatten_values)[-1]
        self.root_node = self.__build_node(index_values, None, 0)

    def __build_node(self, flatten_values, parent_node, depth):
        if len(flatten_values) == 0:
            return None
        medium_index = len(flatten_values) // 2
        axis = depth % self.max_split_dim
        values_sort = sorted(flatten_values, key=ContainsIndexItemgetter(axis))
        node_value = values_sort[medium_index]
        node = KDNode(node_value, axis, depth)
        node.parent_node = parent_node
        node.left_node = self.__build_node(values_sort[:medium_index], node, depth + 1)
        node.right_node = self.__build_node(values_sort[medium_index + 1:], node, depth + 1)
        return node

    def find_relation_leaf(self, x):
        """
        查找数据点x所属的叶子节点空间的分割叶子节点
        :param x:数据点x
        :return: 所属的叶子节点
        """
        if not x:
            return None
        if np.ndim(x) == 1:
            flatten_values = np.array([x])
        else:
            flatten_values = np.array([element.flatten() for element in np.array(x)])
        if self.max_split_dim == flatten_values.shape[-1]:
            result = []
            for item in flatten_values:
                cur_node = self.root_node
                while True:
                    if cur_node.is_leaf():
                        result.append(cur_node.value)
                        break
                    if cur_node.value[1][cur_node.axis] > item[cur_node.axis]:
                        if cur_node.left_node is not None:
                            cur_node = cur_node.left_node
                        else:
                            cur_node = cur_node.right_node
                    else:
                        if cur_node.right_node is not None:
                            cur_node = cur_node.right_node
                        else:
                            cur_node = cur_node.left_node
            return np.squeeze(result)
        else:
            raise KeyError(
                "dim {} != {}, input dim must eq kdTree dim，".format(flatten_values.shape[-1], self.max_split_dim))

    @staticmethod
    def find_min_distance_node(node, x, method):
        """
        :param node: KDNode, need query node
        :param x: query data point
        :param method: distance method
        :return: min distance and min distance node
        """
        nodes = Queue(maxsize=0)
        nodes.put(node)
        min_distance = None
        min_distance_node = None
        while not nodes.empty():
            n = nodes.get()
            distance = method(x, n.value)
            if min_distance is None:
                min_distance = distance
                min_distance_node = n
            else:
                if min_distance > distance:
                    min_distance = distance
                    min_distance_node = n
            if n.left_node is not None:
                nodes.put(n.left_node)
            elif n.right_node is not None:
                nodes.put(n.right_node)

        return min_distance, min_distance_node

    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum(np.square(x1, x2)))

    def __str__(self):
        return "KdTree->rootNode=\n" + str(self.root_node)


class ContainsIndexItemgetter(object):
    """
    这个排序的的key对应的类的目的是为了让排序后的item同时包含有在原始的数组中的位置信息
    """

    def __init__(self, axis):
        self.axis = axis

    def __call__(self, item):
        return item[1][self.axis]


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
            self_str += "\n  {}|—— Left : {}".format(str_child_padding, self.left_node)
        if self.right_node:
            self_str += "\n  {}|—— Right : {}".format(str_child_padding, self.right_node)
        return self_str

    def is_leaf(self):
        return self.left_node is None and self.right_node is None

    def is_root(self):
        return self.parent_node is not None


if __name__ == '__main__':
    test_arr = ((2, 3, 6), (5, 4, 8), (9, 6, 1), (4, 7, 2), (8, 1, 5), (7, 2, 4))
    kd_tree = KDTree(test_arr)
    print(kd_tree)
    print(kd_tree.find_relation_leaf(((1, 2, 3), (4, 5, 6), (9, 7, 1))))

    test_arr = ((2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2))
    kd_tree = KDTree(test_arr)
    print(kd_tree)
    print(kd_tree.find_relation_leaf(((1, 2), (3, 4))))

    # test_arr = np.random.randn(20, 5, 1)
    # kd_tree = KDTree(test_arr)
    # print(kd_tree)
