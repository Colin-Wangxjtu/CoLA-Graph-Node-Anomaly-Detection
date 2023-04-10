from statistics import mode
import torch
from torch.utils.data import DataLoader

from cora_datasets import Pair_Dataget
from CLM import CLM_Model

batch = 300
Epoch = 20
dim_attribute = 1433 # 由数据集的属性矩阵尺寸决定
dim_embidding = 64
device = 'cuda'

dataset, shuffle_list, anomaly_nodes = Pair_Dataget()
datas = DataLoader(dataset, batch_size=batch, shuffle=True)

model = CLM_Model(dim_attribute, dim_embidding, device)
model.to(device)

lost = torch.nn.BCELoss()
lost = lost.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=5e-4) # 创建优化器

i = 0

# for epoch in range(Epoch):
#     for data, label in datas:
#         label = label.float()
#         data[0], data[1], data[2], label = data[0].to(device), data[1].to(device), data[2].to(device), label.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = lost(output, label)
#         loss.backward()
#         optimizer.step()
#     print('loss for epoch {} ='.format(epoch),loss.item())
# torch.save(model.state_dict(), "model_parameter.pkl")
model.load_state_dict(torch.load('model_parameter.pkl'))

pred_anomaly = []
true_anomaly = []
correct = 0
sum = 0
for i in range(int(len(dataset)/2)):
    p_data = (dataset[2*i][0][0], dataset[2*i][0][1], dataset[2*i][0][2])
    n_data = (dataset[2*i+1][0][0], dataset[2*i+1][0][1], dataset[2*i+1][0][2])
    data = (torch.cat((p_data[0], n_data[0])).reshape(2, 1433).to(device), torch.cat((p_data[1], n_data[1])).reshape(2, 4, 4).to(device), torch.cat((p_data[2], n_data[2])).reshape(2, 4, 1433).to(device))
    if model(data)[0] - model(data)[1] < 0.00001 :
        pred_node = shuffle_list[i]
        pred_anomaly.append(pred_node)
        if pred_node in anomaly_nodes:
            correct += 1
        sum += 1

print('correct num = {}\ntotal num = {}'.format(correct, sum))