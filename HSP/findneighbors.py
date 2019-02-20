#!/usr/bin/env python  
# -*- coding:utf-8 -*-
__author__ = 'yangqingqing'
__time__ = '2019/1/13 下午9:30'
from sklearn import preprocessing
import logging
import os
from random import choice
import sqlite3
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import re
from collections import defaultdict
from collections import Counter
import pandas as pd
from multiprocessing import Pool

import random
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    filename='use_pandas_find_neighbors.log',
                    filemode='w')
db_path='/Users/yangqingqing/Desktop/dataset-rdf/dbpedia_en.sqlite'
def settypelist():
    global typelist
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("SELECT DISTINCT(entity_type_id) FROM property_triples")
    value = cursor.fetchall()
    alltype_id = [m[0] for m in value]
    typelist = alltype_id
def write_dict(file_path,dict):
    with open(file_path, "w") as f:
        for k in dict.keys():
            s=str(k)
            if dict[k]:
                for i in dict[k]:
                    s=s+' '+str(i)
            s=s+'\n'
            f.writelines(s)
    f.close()
def find_neighbors(hypercube_radius,df,dim,i):
    neighbours=[]
    res_array = []
    for j in range(0,dim):
        #cr = hypercube_radius[j]
        cr=0
        col = "f" + str(j)
        datax = df[["node_id", col]]
        min_value = df.loc[i, col] - cr
        max_value = df.loc[i, col] + cr
        nodes = datax["node_id"][(datax[col] > min_value) & (datax[col] < max_value)].tolist()
        res_array.append(nodes)
        # 求交集intersection，得到索引邻近点序号
    new_res_array=[]

    for r in res_array:
        if len(r)!=0:
          new_res_array.append(r)

    if len(new_res_array)>0:
        neighbours=new_res_array[0]
        for r in new_res_array:
            neighbours = set(neighbours).intersection(r)

    #neighbours = set(res_array[0]).intersection(*res_array[1:])

    if df.loc[i,"node_id"] in neighbours:
           neighbours.remove(df.loc[i,"node_id"])
    #邻居点过多，随机抽取一些
    if len(neighbours)>100:
        random_neighbors=random.sample(neighbours, 20)
        return list(random_neighbors)

    return list(neighbours)

def process_data(t):
    # read file
    logging.info("%s type number", t)
    dict_node_neigh = defaultdict(list)
    datalist = []
    dict_nodes = defaultdict(list)
    f = open('/Users/yangqingqing/PycharmProjects/HAS_master/data/adata_number_csv/' + str(t) + '.txt', "r")
    lines = f.readlines()
    for line in lines:
        m = line.strip().split('\t')
        l = m[1:]
        datalist.append(l)
    data = np.array(datalist)
    if data.size == 0:
        return dict_node_neigh

    dim=len(datalist[0])
    num_elements=data.shape[0]
    type_df = pd.read_csv('/Users/yangqingqing/PycharmProjects/HAS_master/data/adata_number_csv/' + str(t) + '.txt', "\t",
                          names=createHeaders(dim))
    print(num_elements,dim)
    #找每一维的半径cr
    hypercube_radius = []
    for j in range(dim):
        col = "f" + str(j)
        median = type_df[col].median()
        b = 1.4826  # 这个值应该是看需求加的，有点类似加大波动范围之类的
        mad = b * (abs(type_df[col] - median)).median()
        lower_limit = median - (3 * mad)
        upper_limit = median + (3 * mad)
        validate_nodes = type_df[col][(type_df[col]> lower_limit) & (type_df[col] < upper_limit)].tolist()
        validate_nodes_len = len(set(validate_nodes))

        if validate_nodes_len==0:
            temp = type_df[col].tolist()
            cr=1/len(temp)
        else:
            print(max(validate_nodes), min(validate_nodes))
            print(validate_nodes_len)
            cr = (max(validate_nodes) - min(validate_nodes)+1) / validate_nodes_len
        hypercube_radius.append(cr)
    print(len(hypercube_radius))
    print(hypercube_radius)
    #"=================================="
    #多核的使用，使用pandas

    p = Pool()
    for i in range(num_elements):
        dict_node_neigh[type_df.loc[i,"node_id"]]=p.apply_async(find_neighbors,args=(hypercube_radius,type_df,dim,i)).get()

    write_dict('/Users/yangqingqing/PycharmProjects/HAS_master/data/aneigh_number/' + str(t) + '.txt', dict_node_neigh)
    p.close()
    p.join()

def createHeaders(vec_size):
    headers = ['node_id']
    for i in range(0, vec_size):
        headers.append('f'+str(i))
    return headers

if __name__ == '__main__':
    # find_neighbors(8)
    # settypelist()
    # for t in typelist:
    #     find_neighbors(t)
    process_data(352139)