#!/usr/bin/env python  
# -*- coding:utf-8 -*-
import argparse
import networkx as nx
from src import SPVec
from gensim.models import Word2Vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(filename)s ： %(funcName)s ： %(message)s',
                    level=logging.INFO,
                    filename='SPvec.log',
                    filemode='a+')


def parse_args():
    """
    解析 SP2vec 参数.
    """
    parser = argparse.ArgumentParser(description="Run node2vec.")
    # nargs:参数的数量
    parser.add_argument('--input', nargs='?', default='', help='Input graph path')
    parser.add_argument('--output', nargs='?', default='', help='Embeddings path')
    # 特征维度：默认为128维
    parser.add_argument('--dimensions', type=int, default=128, help='Number of dimensions. Default is 128.')
    # 每个节点步行长度：默认为80
    parser.add_argument('--walk-length', type=int, default=80, help='Length of walk per source. Default is 80.')
    # 每个节点的步行数量：默认值为10
    parser.add_argument('--num-walks', type=int, default=10, help='Number of walks per source. Default is 10.')
    # 滑动窗口大小：默认值为10
    parser.add_argument('--window-size', type=int, default=10, help='Context size for optimization. Default is 10.')
    # SGD的迭代次数：默认值为1
    parser.add_argument('--iter', default=1, type=int, help='Number of epochs in SGD')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers. Default is 8.')
    # 是否有权值：默认无权值
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)
    # 默认为无向图
    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=True)
    return parser.parse_args()


def read_graph(filepath):
    """ 读取networkx中的输入网络."""
    if args.weighted:
        G = nx.read_edgelist(filepath, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(filepath, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()
    return G


def get_dim_numnodes():
    """
    得到类型种类数，即dim
    :return:
    """
    # f = open(args.input, "r")
    # lines = f.readlines()
    typelist = set()
    dict_tailnode = {}
    con_type_list = [30, 168, 19, 81, 167, 114, 15, 184, 130, 43, 5, 12, 29, 46, 58, 129, 126, 44, 170,
                     59, 13, 191, 60, 123, 74, 28, 23, 201, 47, 37, 4, 116, 121, 2, 200, 109, 33, 61, 7, 204]
    with open(args.type, "r") as f:
        for line in f.readlines():
            m = line.strip().split(' ')
            head = m[0]
            tail_type = int(m[1])
            if tail_type not in con_type_list:
                tail_type = -1
            typelist.add(tail_type)
            dict_tailnode[head] = tail_type

    typeset = sorted(typelist)
    return list(typeset), dict_tailnode


def learn_embeddings(walks):
    """Learn embeddings by optimizing the Skipgram objective using SGD.通过使用SGD优化Skipgram目标来学习嵌入"""
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=args.iter)
    logging.info("Start save the model...")
    model.wv.save_word2vec_format(args.output)


def main(args):
    nx_G = read_graph(args.input)
    type_list, node_type = get_dim_numnodes()
    node_list = nx.nodes(nx_G)
    SP = SPVec.SPVecGraph(nx_G, type_list, node_list, node_type)
    r = SP.get_r(nx.number_of_nodes(nx_G))  # 2.7529984971381205e-07
    change_r = SP.change_r(r)  # 3.6870336653401483e-07
    SP.find_neighbor_redis(change_r)

    # 生成新的图开始随机游走，walks是随机游走生成的多个节点序列
    # SP.new_graph()
    # new_G = read_graph("data/karate.edgelist")
    # walks = SP.build_walks(new_G, num_walks=args.num_walks, walk_length=args.walk_length, alpha=0)
    # learn_embeddings(walks)


if __name__ == '__main__':
    args = parse_args()
    args.input = 'data/umls-subset100w.csv'
    args.type = 'data/umls_CUI_types_washed.csv'
    # args.input = 'graph.txt'
    # args.type = 'type.txt'
    args.output = './result.emb'

    main(args)
