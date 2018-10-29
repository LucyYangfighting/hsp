#!/usr/bin/env python  
# -*- coding:utf-8 -*-
import numpy as np
from sklearn import preprocessing
from collections import Counter
import logging
import networkx as nx
import random
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(filename)s ： %(funcName)s ： %(message)s',
                    level=logging.INFO,
                    filename='SPvec.log',
                    filemode='a+')


class SPVecGraph(object):

    def __init__(self, nx_G, typelist, nodelist, node_type):
        self.nx_G = nx_G
        self.typelist = typelist
        self.nodelist = nodelist
        self.node_type = node_type
        self.input_matrix = self.build_input_matrix()
        self.data_scaled = self.preprocessing_data()

        self.num_elements = len(self.nodelist)
        self.dim = len(self.typelist)

        self.inputs_sorted, self.inputs_sorted_index = self.get_inputs_sorted()

    def preprocessing_data(self):
        min_max_scaler = preprocessing.MinMaxScaler()
        matrix = self.input_matrix[:, 1:]
        data_scaled = min_max_scaler.fit_transform(matrix)
        return data_scaled

    def build_input_matrix(self):
        """
        得到样本矩阵，真正的输入
        :return:
        # TODO: extract from main file and treat as data processing part
        """
        if os.path.exists("data/matrix.out"):
            return np.loadtxt("data/matrix.out")
        else:
            matrix = []
            for node in self.nodelist:
                neighbors = nx.neighbors(self.nx_G, node)
                mapping_type = []
                for neighbor in neighbors:
                    mapping_type.append(self.node_type[str(neighbor)])
                c = Counter(mapping_type)
                feature = [node]
                for t in self.typelist:
                    feature.append(c[t])
                matrix.append(feature)
            input_matrix = np.array(matrix)
            np.savetxt("data/matrix.out", fmt="%d", X=input_matrix, encoding='utf-8')
            return input_matrix

    def get_inputs_sorted(self):
        inputs = self.data_scaled
        inputs_sorted = np.sort(inputs, axis=0)
        inputs_sorted_index = np.argsort(inputs, axis=0)
        return inputs_sorted, inputs_sorted_index

    def get_r(self, n):
        """
        get the r
        :param n: the number of elements
        """
        return 1 / n

    def change_r(self, r):
        """
        :param r: origin r
        :return: scaled r
        """
        d_mean = self.get_average_degree()
        dim = self.inputs_sorted.shape[1]
        r_new = np.power(d_mean, 1 / dim) * r
        return r_new

    def get_average_degree(self):
        """
        get the average degree of the graph
        # TODO:应该是该类型下的分母
        """
        degree_view = nx.degree(self.nx_G)
        degree = 0
        # [(1, 5), (2, 2), (5, 2), (6, 1), (7, 1), (8, 1), (3, 2), (4, 6), (9, 1), (10, 1), (11, 1), (12, 1)]
        for _, de in degree_view:
            degree += de
        n_nodes = self.nx_G.number_of_nodes()
        return degree / n_nodes

    def find_neighbor(self, cr):
        """
        得到hypercube中最近邻节点
        :param data_scaled:
        :param cr:
        :return:
        """
        if os.path.exists("data/output.txt"):
            pass
        else:
            mappingg_index_nodelist = {}
            for index, val in enumerate(self.nodelist):
                mappingg_index_nodelist[index] = val

            # 得到每维特征排序后的data_array，方便后续查找数据，先排序，复杂度是O(dim*n*logn),
            # 排序之后顺序换了样本顺序换了，需要保存下来故有了index_sorted
            result = []
            for i in range(self.num_elements):
                # 求第i个样本点的近邻点
                res_array = []
                dim = self.inputs_sorted.shape[1]
                k = 0
                d = self.get_average_degree()
                l = []
                se = set()
                # 这样会陷入死循
                time = 0
                while k < 2 * d and time < 5:
                    for j in range(dim):
                        # while cr < 1:
                        min = self.data_scaled[i][j] - cr
                        max = self.data_scaled[i][j] + cr
                        inputs_sorted_j = self.inputs_sorted[:, j]
                        inputs_sorted_index_j = self.inputs_sorted_index[:, j]
                        res = self.binary_search_interval(inputs_sorted_j, inputs_sorted_index_j, min, max)
                        # if len(res) == 0:
                        #     cr = 2 * cr
                        # logging.info("结点%s的近邻 ： %s", str(i), res)
                        res_array.append(res)
                    # 求交集intersection
                    l = [mappingg_index_nodelist[i]]
                    se = set(res_array[0])
                    for k in range(1, len(res_array)):
                        se = se.intersection(set(res_array[k]))
                    k = len(se)
                    cr = 2 * cr
                    time = time + 1
                true_node = []
                for s in se:
                    true_node.append(mappingg_index_nodelist[s])

                m = l + list(true_node)
                logging.info(m)
                result.append(m)
            self.write_txt("output", result)

    def binary_search_interval(self, feature_j, feature_j_index, min, max):
        """
        二分查找得到区间(f-r，f+r)的样本,只是要找到区间中的值，二分找而不是从前往后O(n)
        找出该维度特征上对应区间的（min,max）的点，返回这些点（点最原始的索引值）
        :param feature_j:
        :param feature_j_index:
        :param min:
        :param max:
        :return:
        """
        res = []
        start = 0
        end = len(feature_j)
        mid = 0
        while start <= end:
            # mid = int((start + end) / 2)  # 存在越界风险
            mid = int(start + (end - start) / 2)
            if min <= feature_j[mid] <= max:
                break
            if feature_j[mid] < min:
                start = mid + 1
            if feature_j[mid] > max:
                end = mid - 1

        i = j = mid
        while min <= feature_j[i] <= max:
            res.append(feature_j_index[i])
            i = i - 1
            if i < 0:
                break

        while min <= feature_j[j] <= max:
            res.append(feature_j_index[j])
            j = j + 1
            if j >= self.num_elements:
                break
        return res

    def new_graph(self):
        if os.path.exists("data/new_edgelist.txt"):
            # with open("data/new_edgelist.txt", 'r', encoding='utf-8') as f:
            #     return [line for line in f.readlines()]
            pass
        else:
            edgelist = []
            with open('data/output.txt', "r") as f:
                lines = f.readlines()
                for line in lines:
                    m = line.strip().split(' ')
                    node = m[0]
                    for i in m[1:]:
                        if node != i:
                            edgelist.append([node, i])
                np.savetxt("data/new_edgelist.txt", fmt='%s', X=np.array(edgelist), encoding='utf-8')
                # self.write_txt("new_edgelist", edgelist)

    def build_corpus(self, G, num_paths, path_length, alpha=0,
                     rand=random.Random(0)):

        if os.path.exists("data/walks.txt"):
            with open("data/walks.txt", 'r', encoding='utf-8') as f:
                return [line for line in f.readlines()]
        else:
            walks = []
            nodes = list(nx.nodes(G))
            for cnt in range(num_paths):
                rand.shuffle(nodes)
                for node in nodes:
                    '''随机游走参数是游走的长度，随机方式rand，alpha是可能性一个参数，start是开始节点'''
                    walks.append(self.random_walk(path_length, rand=rand, alpha=alpha, start=node))
            walks = np.array(walks)
            # np.savetxt("data/walks.txt", fmt="%s", X=walks, encoding='utf-8')
            self.write_txt("walks", walks)
            return walks

    def write_txt(self, name, list):
        with open("data/" + name + ".txt", "a") as f:
            for li in list:
                s = ''
                for l in li:
                    s = s + str(l) + ' '
                s += '\n'
                f.writelines(s)
        f.close()

    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        """ Returns a truncated random walk.
            path_length: Length of the random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        G = self.nx_G
        if start:
            path = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            path = [rand.choice([node for node in nx.nodes(G)])]

        while len(path) < path_length:
            cur = path[-1]
            neighbors = [n for n in nx.neighbors(G, cur)]
            if len(neighbors) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(neighbors))
                else:
                    path.append(path[0])
            else:
                break
        return [int(node) for node in path]