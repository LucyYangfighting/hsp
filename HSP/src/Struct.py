#!/usr/bin/env python  
# -*- coding: utf-8 -*-
"""
@author: TenYun  
@contact: qq282699766@gmail.com  
@time: 2018/10/1 13:43 
"""
import argparse
import networkx as nx
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    filename='Struct2Vec.log',
                    filemode='a+')

def read_graph(args):
    """
    Reads the input network in networkx.
    """
    logging.info("Start -- Reads the input network in networkx...")
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        logging.info("init all edges weight 1...")
        # G.edges() 形式如[(1,2),(1,3),(4,3)...]
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
        logging.info("init end...")
    if not args.directed:
        G = G.to_undirected()
    logging.info("End -- Reads the input network in networkx...")
    return G

args = {"--input": "G:\groom\groom-deco\hsp-master\HSP\src\graph.txt", "--weighted": "false"}


class Struct2Vec(object):

    def __init__(self):
        pass
        # self.G = read_graph(args)
        self.inputs_sorted, self.inputs_sorted_index, self.nodes,  = self.get_inputs()

    def get_average_degree(self):
        """
        get the average degree of the graph
        :return:
        """
        return nx.degree(self.G) / self.G.number_of_nodes()

    def build__input_matrix(self):
        """
        构造输入矩阵，形如
        [
            [1，1, 2, 0, 0, 3, 4, ... ],
            [2，3, 4, 0, 1, 3, 4, ... ],
            ...
        ]
        每一行的第一个数代表nodeID，后面代表直接邻居的结点类型分布
        :return:
        """
        G = self.G
        matrix = []
        global line
        for node in nx.nodes(G):
            line = [node]
            neighbors = nx.all_neighbors(node)
            line.append(neighbors)
            matrix.append(line)
        return matrix

    def get_inputs(self):
        inputs = np.array(self.build__input_matrix())
        nodes = inputs[:, 0]
        inputs = inputs[:, 1:]
        inputs = self.normalization(inputs)
        inputs_sorted = np.sort(inputs, axis=0)
        inputs_sorted_index = np.argsort(inputs, axis=0)
        return inputs_sorted, inputs_sorted_index, nodes

    def get_R(self):
        """
        get the half of r length
        """
        inputs_sorted = self.inputs_sorted
        dim = inputs_sorted.shape[1]
        average_intervals = self.get_average_intervals(inputs_sorted)
        r = np.power(np.cumproduct(average_intervals)[-1] * self.get_average_degree() / np.power(2, dim), 1 / dim)
        return r

    def get_average_intervals(self, inputs_sorted):
        """
        Calculate the average interval
        """
        inputs_sorted_copy_margin = np.r_[inputs_sorted[1:, :], [
            inputs_sorted[-1, :]]]  # ignore the first line of the matrix and duplicate the last line
        return (inputs_sorted_copy_margin - inputs_sorted).sum(axis=0) / (inputs.shape[0] - 1)

    def normalization(self, arr):
        """
        normalize to eliminate dimensional influence
        """
        return (arr - arr.min()) / (arr.max() - arr.min())

    def get_andidate_nodes(self, node, r):
        """
        get the candidates of a node
        """
        inputs_sorted = self.inputs_sorted
        inputs_sorted_index = self.inputs_sorted_index
        start = inputs_sorted + r
        end = inputs_sorted - r



s2v = Struct2Vec()
inputs = np.array([[3, 1, 2], [-2, -1, 3], [-3, -2, 4], [-1, -1, 5], [2, 1, 6], [3, 2, 7]])
# print(inputs)
# inputs = s2v.normalization(inputs)
# print(np.sort(inputs, axis=0))
# print(np.argsort(inputs, axis=0))
ar = np.array([[0., 0.1, 0.5],
               [0.1, 0.2, 0.6],
               [0.2, 0.2, 0.7],
               [0.5, 0.4, 0.8],
               [0.6, 0.4, 0.9],
               [0.6, 0.5, 1.]])
a = np.array(ar[1:, :])
b = np.array(ar[-1, :])
ab = np.r_[ar[1:, :], [ar[-1, :]]]
# print(ab-ar)
inter = (ab - ar).sum(axis=0) / (ar.shape[0] - 1)
# print(np.cumprod(inter)[-1])
# print(a)
# print(b)
# print(np.r_[a, b])
print(np.power(1, 2))
