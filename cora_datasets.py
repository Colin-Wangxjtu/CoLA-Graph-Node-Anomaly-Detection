from ctypes import sizeof
import pandas as pd
import numpy as np

import torch

# hpyer-parameters
Epoch = 10
Batch = 64
Rounds = 256

# part1 数据集导入及处理 该part应该被封装为类or函数，先能把流程走完再优化吧
def cora_import():
    # 数据集导入
    raw_data = pd.read_csv(r'../datasets/cora/cora.content', sep='\t', header=None) # sep为分隔符,header指定表头,返回的数据结构是DataFrame
    num = raw_data.shape[0] # 数据总数

    a = list(raw_data.index) # DataFrame.index返回行的索引,在这里是[0:2708]
    b = list(raw_data[0])
    c = zip(b, a) # 将两个列表打包成zip类
    map = dict(c)
    # 最终结果是一个字典,编号:序号

    attribute = torch.tensor(raw_data.iloc[:, 1:-1].values, dtype=torch.float32) # 提取第二列到倒数第二列的元素,只包含属性
    labels = pd.get_dummies(raw_data[1434]) # 将最后一列的标签独热编码(即用0和1划分是否处于该类)

    raw_data_cites = pd.read_csv(r'../datasets/CORA/cora.cites', sep='\t', header = None)

    # 创建邻接矩阵
    matrix = np.zeros((num, num), dtype='int8')
    for i in range(len(raw_data_cites[0])):
        matrix[map[raw_data_cites.loc[i, 1]]][map[raw_data_cites.loc[i, 0]]] = 1
        matrix[map[raw_data_cites.loc[i, 0]]][map[raw_data_cites.loc[i, 1]]] = 1

    # 至此,数据导入完成 得到的数据有:邻接矩阵matrix, 标签的独热编码labels, 属性矩阵attribute
    # 先生成一个无向图的邻接矩阵，为了方便生成异常结点。后续可以更改

    # 插入异常结点
    anomaly_lables = np.zeros(num) # 创建一个全零向量,用以标注是否为异常结点
    p = 15 # 加入的结构异常结点的每组的结点数
    q = 5 # 加入的结构异常结点的组数
    k = 50 # 加入内容异常结点时的候选结点

    anomaly_nodes = np.random.randint(0,num,size=(2*p*q))
    for i in range(2*p*q):
        anomaly_lables[anomaly_nodes[i]] = 1 # 将随机选中的异常结点的异常标签改为1

    # 插入异常结构结点
    struct_anomaly = anomaly_nodes[:p*q].reshape(5,15)
    for times in range(q):
        for i in range(p-1):
            for j in range(i+1, p): # 将该列表内的结点全连接
                matrix[struct_anomaly[times][i], struct_anomaly[times][j]] = 1
                matrix[struct_anomaly[times][j], struct_anomaly[times][i]] = 1

    # 插入异常内容结点
    context_anomaly = anomaly_nodes[p*q: 2*p*q]
    for i in range(p*q):
        candidate_set = np.random.randint(0,num,size=k)
        e_distance = np.array(np.linalg.norm(attribute[context_anomaly[i]]-attribute[candidate_set[n]]) for n in range(k))
        index = np.argmax(e_distance) # 计算并找出任取的k个结点中与目标节点欧氏距离最远的结点的下标
        attribute[candidate_set[index]] = attribute[context_anomaly[i]] # 将目标结点的值赋给找出的结点

    # 该过程生成了2pq个异常结点，pq个为内容异常，pq个为结构异常
    return matrix, attribute, anomaly_nodes

