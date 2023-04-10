import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, Maplayer, Readout, Discriminator

class CLM_Model(nn.Module):

    def __init__(self, in_dim, out_dim, device):
        super().__init__()
        self.device = device
        self.GCN = GraphConvolution(in_dim, out_dim, self.device)
        self.Dis = Discriminator(out_dim, self.device)

    def forward(self, x):
        self.GCN.to(self.device)
        self.Dis.to(self.device)
        tarnode_att = x[0]  # dim = batch * in_dim
        sub_adj = x[1]      # dim = batch * s * s
        sub_att = x[2]      # dim = batch * s * in_dim
        elg, weight = self.GCN(sub_adj, sub_att)
        # print(weight[0, 0])
        elg = F.relu(elg)
        elg = Readout(elg)
        eth = Maplayer(tarnode_att, weight)
        eth = F.relu(eth)
        s = self.Dis(elg, eth).to(self.device)
        s = torch.sigmoid(s)
        return s
