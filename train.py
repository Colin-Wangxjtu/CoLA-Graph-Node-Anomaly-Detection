from statistics import mode
import torch
from torch.utils.data import DataLoader

import numpy as np

from cora_datasets_copy import cora_import
from CLM import CLM_Model
from utils import get_pairs

num = 2708
batch = 300
Epoch = 30
dim_attribute = 1433 # 由数据集的属性矩阵尺寸决定
dim_embidding = 64
s = 4
anomaly_num = 150
score_times = 25
device = 'cuda'

random_seq = np.arange(num)
np.random.shuffle(random_seq)
dataset = torch.tensor(random_seq)
matrix, att, anomaly_nodes = cora_import()
datas = DataLoader(dataset, batch_size=batch, shuffle=True)

model = CLM_Model(dim_attribute, dim_embidding, device)
model.to(device)

lost = torch.nn.BCELoss()
lost = lost.to(device)

optimizer = torch.optim.Adam(model.parameters()) # 创建优化器

for epoch in range(Epoch):
    for batch in datas:
        data = [torch.zeros((2*len(batch), dim_attribute)), torch.zeros((2*len(batch), s, s)), torch.zeros((2*len(batch), s, dim_attribute))]
        label = torch.zeros(2*len(batch))
        for i, one_data  in enumerate(batch):
            pair = get_pairs(one_data, att, matrix)
            data[0][2*i], data[1][2*i], data[2][2*i] = pair[0][0], pair[0][1], pair[0][2]
            label[2*i] = 1
            data[0][2*i+1], data[1][2*i+1], data[2][2*i+1] = pair[1][0], pair[1][1], pair[1][2]
            label[2*i+1] = 0
        data[0], data[1], data[2], label = data[0].to(device), data[1].to(device), data[2].to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lost(output, label)
        loss.backward()
        optimizer.step()
    # print('loss for epoch {} ='.format(epoch),loss.item())
    print(loss.item())
torch.save(model.state_dict(), "model_parameter.pkl")
# model.load_state_dict(torch.load('model_parameter.pkl'))

pred_anomaly = []
true_anomaly = []
score = []
correct = 0
for i in range(num):
    sc = 0
    for j in range(score_times):
        p_data, n_data = get_pairs(i, att, matrix)
        data = (torch.cat((p_data[0], n_data[0])).reshape(2, 1433).to(device), torch.cat((p_data[1], n_data[1])).reshape(2, 4, 4).to(device), torch.cat((p_data[2], n_data[2])).reshape(2, 4, 1433).to(device))
        sc += model(data)[0] - model(data)[1]
    score.append(sc/score_times)
scord_sorted = sorted(range(len(score)), key=lambda i:score[i], reverse=False)
for i in range(anomaly_num):
    if scord_sorted[i] in anomaly_nodes:
        correct += 1

print('correct num = {}\ntotal num = {}'.format(correct, anomaly_num))
