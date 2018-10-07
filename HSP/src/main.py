import numpy as np
from sklearn import preprocessing
import networkx
'''
以下都是input
'''
# dim = 6
# num_elements = 74902
dim=6
num_elements=1000

# data = np.array([[3, 1], [-2, -1], [-3, -2], [-1, -1], [2, 1], [3, 2]])
degree = []  # 这里要得到知识图谱中每个实体的度（出入都包括,一般是同类型之间的度），作为input，自行计算
file_path = '/Users/yangqingqing/Documents/Entity_Profiling/structural/1858106result.txt'

def preprocessing_data():
    # 归一化处理,使用的是z-scores
    # scaler = preprocessing.StandardScaler()
    # data_scaled = scaler.fit_transform(data)
    # 归一化处理，使用的是min-max
    # datalist = []
    # f = open('/Users/yangqingqing/Documents/Entity_Profiling/structural/1858106.txt', "r")
    # lines = f.readlines()
    # for line in lines:
    #    m = line.strip().split(' ')
    #    l = m[1:]
    #    datalist.append(l)
    # data = np.array(datalist)
    # TODO
    data = np.random.randint(0, 20, size=[1000, 6])

    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data)
    return data_scaled

def get_r(n):
    '''
    求 r
    :return: r
    '''
    # J = []  # 平均距离
    # for i in range(dim):
    #     data_c = data_scaled[:, i]
    #     data_column = sorted(data_c)
    #     sum = 0
    #     for j in range(1, num_elements):
    #         sum += data_column[j] - data_column[j - 1]
    #     print(sum)
    #     J.append(sum / (num_elements - 1))
    # s = 1
    # print(J)
    # for i in J:
    #     s = s * i
    # print(s)
    # r = pow(s, 1 / dim)
    return 1 / n

def change_r(r):
    '''
    传入原始的半径r
    :param r:
    :return:
    '''
    sum_d = 0
    for i in range(num_elements):
        sum_d += degree[i]
    # 计算平均度d_mean和w
    d_mean = sum_d / num_elements
    w = d_mean / pow(2, dim)
    # 缩放后的r_new
    r_new = pow(w, 1 / dim) * r
    return r_new

def find_neighbor(data_scaled, cr):
    '''
    得到hypercube中最近邻节点
    :param data_scaled:
    :param cr:
    :return:
    '''
    # 得到每维特征排序后的data_array，方便后续查找数据，先排序，复杂度是O(dim*n*logn),
    data_sorted = np.sort(data_scaled, axis=0)  # 按列排序
    print("================================")
    index_sorted = np.argsort(data_scaled, axis=0)  # 按列排序的到变化后的索引
    print(index_sorted)
    # 排序之后顺序换了样本顺序换了，需要保存下来故有了index_sorted
    result = []
    for i in range(num_elements):
        # 求第i个样本点的近邻点
        res_array = []
        for j in range(dim):
            min = data_scaled[i][j] - cr
            max = data_scaled[i][j] + cr
            temp_D = data_sorted[:, j]
            temp_I = index_sorted[:, j]
            res = BinarySearch_interval(temp_D, temp_I, min, max)
            print("%s样本: %s近邻",str(i),res)
            res_array.append(res)
        # 求交集intersection
        l = []
        l.append(i)
        s=set(res_array[0])
        for i in range(1,len(res_array)):
            s = s.intersection(set(res_array[i]))
        m = l + list(s)
        print(m)
        result.append(m)

    #write_txt(result)
def BinarySearch_interval(list1, list_Index, min, max):
    '''
    二分查找得到区间(f-r，f+r)的样本,只是要找到区间中的值，二分找而不是从前往后O(n)
    找出该维度特征上对应区间的（min,max）的点，返回这些点（点最原始的索引值）
    :param list:
    :param list_Index:
    :param min:
    :param max:
    :return:
    '''
    res = []
    start = 0
    end = len(list1)
    while start <= end:
        mid = int((start + end) / 2)
        if list1[mid] >= min and list1[mid] <= max:
            break
        if list1[mid] < min:
            start = mid + 1
        if list1[mid] > max:
            end = mid - 1

    i = mid
    j = mid
    while list1[i] >= min and list1[i] <= max:
        res.append(list_Index[i])
        i = i - 1
        if i < 0:
            break

    while list1[j] >= min and list1[j] <= max:
        res.append(list_Index[j])
        j = j + 1
        if j >= num_elements:
            break
    return res

def write_txt(list):
    with open(file_path, "a") as f:
        for li in list:
            s = ''
            for l in li:
                s = s + str(l) + ' '
            s += '\n'
            f.writelines(s)
    f.close()
def main():
    data_scaled = preprocessing_data()
    change=False
    if change:
       tempr = get_r(data_scaled)
       r = change_r(tempr)
    else:
       r = get_r(data_scaled)
    #r = 0.1
    print(r) 
    # 将结果写入文件中
    find_neighbor(data_scaled, r)
if __name__ == '__main__':
    main()


