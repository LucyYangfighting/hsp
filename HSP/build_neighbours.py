#!/usr/bin/env python  
# -*- coding: utf-8 -*-
import logging
import random
from multiprocessing.pool import Pool
import argparse
import numpy as np
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(filename)s : %(funcName)s : %(message)s',
                    level=logging.INFO,
                    filename='bn.log',
                    filemode='a+')


def preprocessing_data():
    if os.path.exists("data/data_scaled2.matrix"):
        logging.info("start load data_scaled2.matrix")
        data_scaled = np.loadtxt("data/data_scaled2.matrix")
        return data_scaled[:, 0], data_scaled[:, 1:]


def get_inputs_sorted(data_scaled):
    logging.info("start load inputs_sorted")
    inputs = data_scaled
    inputs_sorted = np.sort(inputs, axis=0)
    inputs_sorted_index = np.argsort(inputs, axis=0)
    return inputs_sorted, inputs_sorted_index


def binary_search_interval(feature_j, feature_j_index, min, max, num_elements):
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
        if j >= num_elements:
            break
    return res


def write_txt(name, list):
    with open("data/" + name + ".txt", "a") as f:
        for li in list:
            s = ''
            for l in li:
                s = s + str(l) + ' '
            s += '\n'
            f.writelines(s)


def find_neighbor_mul(cr, start, end, filenum):
    """ 多进程实现 """
    try:
        if os.path.exists("data/neighbors_" + str(filenum) + ".txt"):
            pass
        else:
            nodelist, data_scaled = preprocessing_data()
            num_elements = len(nodelist)
            inputs_sorted, inputs_sorted_index = get_inputs_sorted(data_scaled)
            result = []
            if end > num_elements:
                end = num_elements
            dim = data_scaled.shape[1]
            d = 18
            for i in range(start, end):
                res_array = []
                k = 0
                time = 0
                se = set()
                while k < 2 * d and time < 2:
                    for j in range(dim):
                        min = data_scaled[i][j] - cr
                        max = data_scaled[i][j] + cr
                        inputs_sorted_j = inputs_sorted[:, j]
                        inputs_sorted_index_j = inputs_sorted_index[:, j]
                        res = binary_search_interval(inputs_sorted_j, inputs_sorted_index_j, min, max, num_elements)
                        res_array.append(res)
                    se = set(res_array[0]).intersection(*res_array[1:])
                    k = len(se)
                    cr = 2 * cr
                    time = time + 1
                true_node = [nodelist[s] for s in se]
                if len(true_node) > 4 * d:
                    true_node = random.sample(true_node, int(4*d))
                l = [int(nodelist[i])]
                m = l + true_node
                if i % 1000 == 0:
                    logging.info("find neighbors for node %s finished", m[0])
                result.append(m)
            write_txt("neighbors_" + str(filenum) + ".txt", str(result))
    except Exception as ex:
        msg = "the error is :%s" % ex
        print(msg)


def parse_args():
    parser = argparse.ArgumentParser(description="Run find_neighbor_mul.")
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers. Default is 8.')
    parser.add_argument('--count', type=int, default=1000, help='the lines of data every process deal with ')
    return parser.parse_args()


def main(args):
    p = Pool(args.workers)
    count = args.count
    cr = 3.6870336653401483e-07 / 2
    for i in range(args.workers):
        p.apply_async(find_neighbor_mul, args=(cr, i * count, (i + 1) * count, i))
    p.close()
    p.join()
    # find_neighbor_mul(cr, 0 * count, count, 0)
    print("all done!!!")


if __name__ == "__main__":
    args = parse_args()
    main(args)