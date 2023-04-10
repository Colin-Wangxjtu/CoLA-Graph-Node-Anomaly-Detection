from ctypes import sizeof
import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import Dataset

# hpyer-parameters
Epoch = 10
Batch = 64
Rounds = 256

# part1 数据集导入及处理 该part应该被封装为类or函数，先能把流程走完再优化吧

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
# 且对于每个结点均有异常标签，值为1则表明该节点为异常节点

# part2 实例对采样
random_seq = np.arange(num)
np.random.shuffle(random_seq) # 获得一个乱序数组
s = 4 # 每一个局部子图的大小
try_time = 15 # 避免锁死在孤立点的尝试次数
def get_pair_index(matrix): # 输入邻接矩阵，输出一个元组(正实例对, 负实例对)
    num = len(matrix[0])
    p_pairs = np.zeros((num, s+1), dtype='int16') # 正实例对
    n_pairs = np.zeros((num, s+1), dtype='int16') # 负实例对
    for i in range(num):
        nownode = random_seq[i] # nownode是当前结点的下标

        # 获得当前结点的一个正实例对
        subgraph = [] # subgraph是选择target node附近的结点,可能用了RWR?
        subgraph.append(nownode)
        flag = 0 # 用来标志RWR回到起点的次数，以避免有孤立点卡死
        while len(subgraph) < s:
            adj_list = np.where(matrix[nownode] == 1) # 寻找当前结点的邻接点
            if np.random.choice(np.array([0,1])): # 0.5的概率回到起始结点
                nownode = np.random.choice(adj_list[0])
            else:
                nownode = random_seq[i]
            if nownode == random_seq[i]:
                flag += 1
            if nownode not in subgraph:
                subgraph.append(nownode)
                flag = 0
            if flag > try_time: # 做出此判定时nownode一定是起始结点
                subgraph.append(np.random.choice(adj_list[0]))
                flag = 0
        pairs = np.append(random_seq[i], subgraph)
        p_pairs[i] = pairs

        # 获得当前结点的一个负实例对
        subgraph = []
        target_node = np.random.choice(random_seq) # 随机选取一个非当前结点的目标节点
        while target_node == random_seq[i]:
            target_node = random.choice(random_seq)
        nownode = target_node
        subgraph.append(nownode)
        flag = 0
        while len(subgraph) < s:
            adj_list = np.where(matrix[nownode] == 1) # 寻找当前结点的邻接点
            if np.random.choice(np.array([0,1])): # 0.5的概率回到起始结点
                nownode = np.random.choice(adj_list[0])
            else:
                nownode = target_node
            if nownode == target_node:
                flag += 1
            if nownode not in subgraph:
                subgraph.append(nownode)
                flag = 0
            if flag > try_time:
                subgraph.append(np.random.choice(adj_list[0]))
                flag = 0
        pairs = np.append(random_seq[i], subgraph)
        n_pairs[i] = pairs

    return (p_pairs, n_pairs)
        
def get_sub_adj(matrix): # 创建局部邻接矩阵
    pairs_index = get_pair_index(matrix)
    p_sub_adj = torch.zeros(len(matrix[0]), s, s)
    n_sub_adj = torch.zeros(len(matrix[0]), s, s)
    for i in range(len(matrix[0])):
        for j in range(s):
            for k in range(s):
                if matrix[pairs_index[0][i, j+1], pairs_index[0][i, k+1]] == 1: # 当在矩阵中的两点有边，则局部邻接矩阵也有边
                    p_sub_adj[i][j][k] = 1
                if matrix[pairs_index[1][i, j+1], pairs_index[1][i, k+1]] == 1:
                    n_sub_adj[i][j][k] = 1
    return (p_sub_adj, n_sub_adj)

def get_sub_att(matrix, attribute): # 创建局部子图的属性矩阵
    pairs_index = get_pair_index(matrix)
    p_sub_att = []
    n_sub_att = []
    tnode_att = []
    for i in range(len(matrix[0])): # 属性矩阵中targetnode的属性为0
        att = torch.tensor([[0]*(len(attribute[pairs_index[0][i][1]])), attribute[pairs_index[0][i][2]], attribute[pairs_index[0][i][3]], attribute[pairs_index[0][i][4]]])
        p_sub_att.append(att) # 返回正实例的局部子图的属性矩阵
        att = torch.tensor([[0]*(len(attribute[pairs_index[1][i][1]])), attribute[pairs_index[1][i][2]], attribute[pairs_index[1][i][3]], attribute[pairs_index[1][i][4]]])
        n_sub_att.append(att) # 负的
        att = torch.tensor(attribute[pairs_index[0][i][0]].clone())
        tnode_att.append(att) # 返回目标节点的属性向量
    return tnode_att, (p_sub_att, n_sub_att)

def get_pairs(matrix, att): # 返回一个元组(目标节点属性矩阵，(局部子图邻接矩阵，局部子图属性矩阵)，实例对标签)
    data_pairs = []
    label_pairs = []
    sub_adjs = get_sub_adj(matrix)
    tnode_atts, sub_atts = get_sub_att(matrix, att)
    for i in range(len(matrix[0])):
        tnode_att = tnode_atts[i]
        p_sub_adj = sub_adjs[0][i]
        p_sub_att = sub_atts[0][i]
        p_label = 1
        data_pairs.append((tnode_att, p_sub_adj, p_sub_att))
        label_pairs.append(p_label)
        n_sub_adj = sub_adjs[1][i]
        n_sub_att = sub_atts[1][i]
        n_label = 0
        data_pairs.append((tnode_att, n_sub_adj, n_sub_att))
        label_pairs.append(n_label)
    return data_pairs, label_pairs

class Pair_Dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

def Pair_Dataget():
    data, label = get_pairs(matrix, attribute)
    Pairs_Data = Pair_Dataset(data, label)
    return Pairs_Data, random_seq, anomaly_nodes
