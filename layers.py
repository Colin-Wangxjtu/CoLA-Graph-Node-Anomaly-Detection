import math
import torch

from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):

    def __init__(self, in_dim, out_dim, device): # 一个折叠层的，需要输入维度和输出维度
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.weight = Parameter(torch.FloatTensor(in_dim, out_dim)) # 每层的参数由Parameter自动设置和更新
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, adj_batch, att_batch): # GCN的卷积公式
        output_tensor = torch.zeros((adj_batch.size(0), adj_batch.size(1), self.out_dim)).to(self.device)
        for data_num in range(adj_batch.size(0)): # 先对batch中的每一个数据求output
            adj = adj_batch[data_num]
            att = att_batch[data_num]
            s_degree = torch.zeros((adj.size(0), adj.size(0))).to(self.device) # 度矩阵
            for i in range(adj.size(0)):
                s_degree[i, i] = 1/math.sqrt(adj.sum(dim=1)[i])
            s_adj = adj + torch.eye(adj.size(0)).to(self.device)
            output_tensor[data_num] = torch.mm(s_degree, torch.mm(torch.mm(s_adj, s_degree), torch.mm(att, self.weight)))
            # output_tensor[data_num] = torch.mm(adj, torch.mm(att, self.weight))
        return output_tensor, self.weight
        

def Maplayer(att, weight):
    return torch.mm(att, weight)

def Readout(Emb_Graph): # Readout Module，不知道dim对不对
    return torch.mean(Emb_Graph, dim=1)

class Discriminator(Module): # 鉴别层  
    def __init__(self, dim, device):
        super().__init__()
        self.dim = dim
        self.device = device
        self.weight = Parameter(torch.FloatTensor(dim, dim)) 
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_()

    def forward(self, elg, eth):
        output = torch.zeros(elg.size(0))
        for i in range(elg.size(0)):
            output[i] = torch.dot(torch.matmul(elg[i], self.weight), eth[i])
        return output