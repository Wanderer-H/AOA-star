import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
import time
import random
import scipy.io as io
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*72*72, 128)  #16 47 47  for 200  # 16*11*11*4 for 100  #16 72 72 for 300
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, input):
        x1 = self.pool(Func.relu(self.conv1(input)))
        x2 = self.pool(Func.relu(self.conv2(x1)))
        # x2 = Func.relu(self.conv2(x1))
        x3 = x2.view(-1, 16*72*72)
        x4 = Func.relu(self.fc1(x3))
        x5 = Func.relu(self.fc2(x4))
        x6 = self.fc3(x5)
        return x6

# class MyLoss(nn.Module):
#     def __init__(self):
#         super(MyLoss, self).__init__()
#
#     def forward(self, h, g):
#         e_value = math.e
#         # loss = 0.5 * (1 + (100 / (1 + e_value ** (-0.1 * (g - 120))))) * (h - g) ** 2  #(1 + (300 / (1 + e_value ** (-0.04*(g-100)))))
#         loss = (h - g) ** 2
#         return loss.mean()

def sample(batch_size, State, H_value):

    idxes = [random.randint(0, len(State) - 1) for _ in range(batch_size)]
    states, Hs = [], []

    for i in idxes:
        states.append(np.asarray(State[i]))
        Hs.append(np.asarray(H_value[i]))

    return np.array(states, dtype=np.float32), \
           np.array(Hs, dtype=np.float32)

if __name__ == '__main__':
    data_path = './'
    data_label = '10_9_12_15'
    State = np.load(data_path + "Dynamic_State_" + data_label + ".npy")
    H_value = np.load(data_path + "H_value_" + data_label + ".npy")

    # Action = np.load("Action_1000_7_15_9.npy")

    net = Net()
    print(net)

    cost = nn.MSELoss(reduction='mean')  #MyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    start = time.time()
    net.to(device)
    batch_size = 8
    Epochs = 1

    running_loss = 0.0
    l = []
    l_mean = [0]
    # for i, data in enumerate(replay_buffer._storage, 0):
    for i in range(Epochs):
        # 获取输入数据
        experience = sample(batch_size, State, H_value)
        inputs, labels = experience[0], experience[1]    #此处对后者处理可以修改逼近的目标
        # print(np.shape(inputs[0]))

        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        inputs, labels = inputs.to(device), labels.to(device)
        # 清空梯度缓存
        optimizer.zero_grad()

        outputs = net(inputs)
        outputs = torch.reshape(outputs, [batch_size])
        # print("the output is:", outputs)
        # print("the truth is:", labels)

        loss = cost(outputs, labels)
        loss.backward(retain_graph=True)
        optimizer.step()

        # 打印统计信息
        running_loss += loss.item()
        l.append(loss.item())

        # if (i + 1) % 300 == 0:
        # l_mean.append(np.mean(l[-300:]))
        l_mean.append(l_mean[-1]*0.1+loss.item()*0.9)

        if (i + 1) % 1000 == 0:
            # 每 2000 次迭代打印一次信息
            print('[%d] loss: %.3f' % (i + 1, running_loss / 1000))
            running_loss = 0.0



    X = np.linspace(0, 1, len(l))  # X轴坐标数据
    X_ = np.linspace(0, 1, len(l_mean))  # X轴坐标数据

    print(len(l))
    print(len(l_mean))
    print(len(X_))
    # Y = X * X  # Y轴坐标数据
    # plt.plot(X,Y,lable="$sin(X)$",color="red",linewidth=2)

    # np.save('l_mean.npy', l_mean)
    # np.save('l.npy', l)
    # io.savemat('loss.mat', {'l_mean': l_mean, 'l': l})
    torch.save(net.state_dict(), "Dynamic_State_300.pth")  # 保存参数
    print('Finished Training! Net saved. Total cost time: ', time.time() - start)

    plt.figure(figsize=(8, 6))  # 定义图的大小
    plt.xlabel("Epoch(s)")  # X轴标签
    plt.ylabel("loss")  # Y轴坐标标签
    plt.title("Loss")  # 曲线图的标题

    # plt.plot(X, l)  # 绘制曲线图
    plt.plot(X_, l_mean)  # 绘制曲线图
    # 在ipython的交互环境中需要这句话才能显示出来
    plt.show()

