#!/usr/bin/env python  
# -*- coding:utf-8 _*-
import argparse
import logging
import random
from multiprocessing.pool import Pool
import pandas as pd
import time
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(filename)s : %(funcName)s : %(message)s',
                    level=logging.INFO,
                    filename='with_pandas.log',
                    filemode='a+')


def write_txt(name, list):
    with open("data/" + name + ".txt", "a") as f:
        for line in list:
            line = [str(w) for w in line]
            writeline = " ".join(line)
            writeline += "\n"
            f.write(writeline)


def build_neighbours(cr, start, end, filenum):
    data = pd.read_csv("data/data_scaled2.matrix", sep=" ",
                       names=['nodeid', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'])
    result = []
    d = 18
    if end > len(data):
        end = len(data)
    for i in range(start, end):
        res_array = []
        k = 0
        count = 0
        while k < 2 * d and count < 2:
            for j in range(1, 11):
                col = "x" + str(j)
                datax = data[["nodeid", col]]
                min = data.loc[i, col] - cr
                max = data.loc[i, col] + cr
                nodes = datax[(datax[col] > min) & (datax[col] < max)]["nodeid"].astype(int)
                res_array.append(nodes)
            neighbours = set(res_array[0]).intersection(*res_array[1:])
            k = len(neighbours)
            cr = 2 * cr
            count = count + 1
        if len(neighbours) > 4 * d:
            neighbours = random.sample(list(neighbours), int(4 * d))
        startnode = [int(data.loc[i]["nodeid"])]
        result.append(startnode + list(neighbours))
        if i % 1 == 0:
            logging.info("find %s nodes neighbours...", str(startnode[0]))
    write_txt("neighbors_" + str(filenum) + ".txt", result)


def parse_args():
    parser = argparse.ArgumentParser(description="Run find_neighbor_mul.")
    parser.add_argument('--start_workers', type=int, default=0, help='Number of parallel workers. Default is 8.')
    parser.add_argument('--end_workers', type=int, default=2, help='Number of parallel workers. Default is 8.')
    parser.add_argument('--count', type=int, default=100, help='the lines of data every process deal with ')
    return parser.parse_args()


def main(args):
    p = Pool(args.end_workers - args.start_workers)
    count = args.count
    cr = 3.6870336653401483e-07 / 2
    for i in range(args.start_workers, args.end_workers):
        p.apply_async(build_neighbours, args=(cr, i * count, (i + 1) * count, i))
    p.close()
    p.join()
    # find_neighbor_mul(cr, 0 * count, count, 0)
    print("all done!!!")


if __name__ == "__main__":
    args = parse_args()
    main(args)