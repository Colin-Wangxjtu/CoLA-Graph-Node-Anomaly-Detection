import numpy as np
import torch
import cora_datasets_copy
def get_pairs(node, att, matrix):
    s = 4 # 每一个局部子图的大小
    try_time = 15 # 避免锁死在孤立点的尝试次数
    
    # 获得当前结点的一个正实例对
    nownode = node
    p_sub_index = [] # 正实例对的下标列表
    p_sub_index.append(nownode)
    flag = 0 # 用来标志RWR回到起点的次数，以避免有孤立点卡死
    while len(p_sub_index) < s:
        adj_list = np.where(matrix[nownode] == 1) # 寻找当前结点的邻接点
        if np.random.choice(np.array([0,1])): # 0.5的概率回到起始结点
            nownode = np.random.choice(adj_list[0])
        else:
            nownode = node
        if nownode == node:
            flag += 1
        if nownode not in p_sub_index:
            p_sub_index.append(nownode)
            flag = 0
        if flag > try_time: # 做出此判定时nownode一定是起始结点
            p_sub_index.append(np.random.choice(adj_list[0]))
            flag = 0

    # 获得当前结点的一个负实例对
    target_node = np.random.choice(np.arange(len(matrix[0]))) # 随机选取一个非当前结点的目标节点
    while target_node == node:
        target_node = np.random.choice(np.arange(len(matrix[0])))
    nownode = target_node
    n_sub_index = [] # 负实例对的下标列表
    n_sub_index.append(nownode)
    flag = 0
    while len(n_sub_index) < s:
        adj_list = np.where(matrix[nownode] == 1) # 寻找当前结点的邻接点
        if np.random.choice(np.array([0,1])): # 0.5的概率回到起始结点
            nownode = np.random.choice(adj_list[0])
        else:
            nownode = target_node
        if nownode == target_node:
            flag += 1
        if nownode not in n_sub_index:
            n_sub_index.append(nownode)
            flag = 0
        if flag > try_time:
            n_sub_index.append(np.random.choice(adj_list[0]))
            flag = 0

    # 创建正负实例对的邻接矩阵
    p_sub_adj = torch.zeros(s, s)
    n_sub_adj = torch.zeros(s, s)
    for i in range(s):
        for j in range(s):
            if matrix[p_sub_index[i], p_sub_index[j]] == 1:
                p_sub_adj[i, j] = 1
            if matrix[n_sub_index[i], n_sub_index[j]] == 1:
                n_sub_adj[i, j] = 1
    
    # 创建正负实例对的属性矩阵
    p_sub_att = torch.zeros((s, att.shape[1]))
    n_sub_att = torch.zeros((s, att.shape[1]))
    node_att = att[node].clone()
    for i in range(1, s): # 属性矩阵中targetnode的属性为0
        p_sub_att[i] = att[p_sub_index[i]].clone()
        n_sub_att[i] = att[n_sub_index[i]].clone()
    return (node_att, p_sub_adj, p_sub_att), (node_att, n_sub_adj, n_sub_att)

matrix, att, _ = cora_datasets_copy.cora_import()
get_pairs(355, att, matrix)