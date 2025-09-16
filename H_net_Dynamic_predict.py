import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
from Astar_25D_CPU import *
import torch.nn as nn
import torch.nn.functional as Func


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*11*11*4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, input):
        x1 = self.pool(Func.relu(self.conv1(input)))
        x2 = self.pool(Func.relu(self.conv2(x1)))
        # x2 = Func.relu(self.conv2(x1))
        x3 = x2.view(-1, 16*11*11*4)
        x4 = Func.relu(self.fc1(x3))
        x5 = Func.relu(self.fc2(x4))
        x6 = self.fc3(x5)
        return x6


def test_net(net_path, model_name, data_path, winsize, data_label, eps):
    Net_Deviation = []
    _2D_Deviation = []
    _3D_Deviation = []
    Integrated = []
    net_estimation = []
    Net_0_20_Deviation = []
    Net_20_40_Deviation = []
    Net_40_60_Deviation = []
    Net_60_80_Deviation = []
    Net_80_100_Deviation = []
    Net_100_120_Deviation = []
    Net_120_140_Deviation = []
    Net_140_160_Deviation = []
    Net_160_180_Deviation = []
    Net_180_200_Deviation = []
    Net_200_220_Deviation = []
    Net_220_240_Deviation = []

    amount = 4000
    net = Net()
    params = torch.load(net_path+model_name)  # 加载参数  Dynamic_State_win0_3
    net.load_state_dict(params)  # 应用到网络结构中
    # net.to(device)
    State = np.load(data_path+"Dynamic_State_"+eps+"_"+winsize+"_"+data_label+".npy")
    # State = np.load(data_path + "Static_State_" + data_label + ".npy")
    H_value = np.load(data_path+"H_value_"+eps+"_"+winsize+"_"+data_label+".npy")
    linear_2D = np.load(data_path+"linear_2D_"+eps+"_"+winsize+"_"+data_label+".npy")
    linear_3D = np.load(data_path+"linear_3D_"+eps+"_"+winsize+"_"+data_label+".npy")

    print(np.shape(State[0]))
    # exit()
    #
    Err = 0

    for i in range(amount):
        print(i)
        # index = random.randint(0, len(H_value) - 1)
        index = i
        # print("Index:", index)
        # State_ = np.transpose(State[index], [2, 0, 1])
        # inputs = np.expand_dims(State_, 0)

        inputs = np.expand_dims(State[index], 0)

        inputs = torch.from_numpy(inputs)

        inputs = inputs.type(torch.FloatTensor)

        H_hat = net(inputs).data.numpy()[0][0]

        abs_Error = np.abs(H_hat - H_value[index])

        # Err += abs_Error

        if 0 <= H_value[index] < 20:
            Net_0_20_Deviation.append(abs_Error)
        elif 20 <= H_value[index] < 40:
            Net_20_40_Deviation.append(abs_Error)
        elif 40 <= H_value[index] < 60:
            Net_40_60_Deviation.append(abs_Error)
        elif 60 <= H_value[index] < 80:
            Net_60_80_Deviation.append(abs_Error)
        elif 80 <= H_value[index] < 100:
            Net_80_100_Deviation.append(abs_Error)
        elif 100 <= H_value[index] < 120:
            Net_100_120_Deviation.append(abs_Error)
        elif 120 <= H_value[index] < 140:
            Net_120_140_Deviation.append(abs_Error)
        elif 140 <= H_value[index] < 160:
            Net_140_160_Deviation.append(abs_Error)
        elif 160 <= H_value[index] < 180:
            Net_160_180_Deviation.append(abs_Error)
        elif 180 <= H_value[index] < 200:
            Net_180_200_Deviation.append(abs_Error)
        elif 200 <= H_value[index] < 220:
            Net_200_220_Deviation.append(abs_Error)
        else:
            Net_220_240_Deviation.append(abs_Error)
        # deviation = (H_hat - H_value[index])
        # print('deviation:', deviation)

        net_estimation.append(H_hat)
        Net_Deviation.append(np.abs(H_hat - H_value[index]))
        _2D_Deviation.append(np.abs(linear_2D[index] - H_value[index]))
        _3D_Deviation.append(np.abs(linear_3D[index] - H_value[index]))
        Integrated.append(-1.*H_hat+1*linear_2D[index]+1*linear_3D[index] - H_value[index])
        # Net_h*0.25 + linear_h_2D + linear_h_3D*1.0

        # if linear_2D[index] - H_value[index] > 0:
        #     print("linear_2D", linear_2D[index], H_value[index])
        # if linear_3D[index] - H_value[index] > 0:
        #     print("\t\t linear_3D", linear_3D[index], H_value[index], H_hat)
        print("\t\t Value", linear_2D[index], linear_3D[index], H_value[index], H_hat)
        # print(Err)
        #
        # print("Estimation of the H is:", H_hat)
        # print("The Ground Truth is:", H_value[index])

    # print("The mean of the predicted error is:", Err / 100)
    print("The total net Deviation is:", np.sum(Net_Deviation))
    # print("The total 2D Deviation is:", np.sum(_2D_Deviation))
    # print("The total 3D Deviation is:", np.sum(_3D_Deviation))
    # print("The total Integrated is:", np.sum(Integrated))
    print("The Net variation is:", np.var(Net_Deviation))
    # print("The 2D variation is:", np.var(_2D_Deviation))
    # print("The 3D variation is:", np.var(_3D_Deviation))
    # print("The Integrated variation is:", np.var(Integrated))
    print("The Net mean is:", np.mean(Net_Deviation))
    # print("The 2D mean is:", np.mean(_2D_Deviation))
    # print("The 3D mean is:", np.mean(_3D_Deviation))
    # print("The Integrated mean is:", np.mean(Integrated))
    # print("2D Maximum:", max(_2D_Deviation))
    # print("3D Maximum:", max(_3D_Deviation))
    # print("Integrated Maximum:", max(Integrated))

    # print("The Net 0-20 mean is:", np.mean(Net_0_20_Deviation))
    # print("The Net 20-40 mean is:", np.mean(Net_20_40_Deviation))
    # print("The Net 40-60 mean is:", np.mean(Net_40_60_Deviation))
    # print("The Net 60-80 mean is:", np.mean(Net_60_80_Deviation))
    # print("The Net over 80 mean is:", np.mean(Net_Over_80_Deviation))
    #
    # print("The Net 0-20 var is:", np.var(Net_0_20_Deviation))
    # print("The Net 20-40 var is:", np.var(Net_20_40_Deviation))
    # print("The Net 40-60 var is:", np.var(Net_40_60_Deviation))
    # print("The Net 60-80 var is:", np.var(Net_60_80_Deviation))
    # print("The Net over 80 var is:", np.var(Net_Over_80_Deviation))


    X_net = np.linspace(0, len(Net_Deviation), len(Net_Deviation))  # X轴坐标数据   0至n，分为n份
    X_2D = np.linspace(0, len(_2D_Deviation), len(_2D_Deviation))
    X_3D = np.linspace(0, len(_3D_Deviation), len(_3D_Deviation))
    X_Integrated = np.linspace(0, len(Integrated), len(Integrated))

    Err_means = [np.mean(Net_0_20_Deviation), np.mean(Net_20_40_Deviation), np.mean(Net_40_60_Deviation),
                 np.mean(Net_60_80_Deviation), np.mean(Net_80_100_Deviation), np.mean(Net_100_120_Deviation),
                 np.mean(Net_120_140_Deviation), np.mean(Net_140_160_Deviation), np.mean(Net_160_180_Deviation),
                 ]

    Err_x = [20, 40, 60, 80, 100, 120, 140, 160, 180]
    # Err_x = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]
    Err_var = [np.var(Net_0_20_Deviation), np.var(Net_20_40_Deviation), np.var(Net_40_60_Deviation),
               np.var(Net_60_80_Deviation), np.var(Net_80_100_Deviation), np.var(Net_100_120_Deviation),
               np.var(Net_120_140_Deviation), np.var(Net_140_160_Deviation), np.var(Net_160_180_Deviation),
               ]

    # print(len(Net_0_20_Deviation), len(Net_20_40_Deviation), len(Net_40_60_Deviation), len(Net_60_80_Deviation),
    #       len(Net_80_100_Deviation), len(Net_100_120_Deviation), len(Net_120_140_Deviation), len(Net_140_160_Deviation),
    #       len(Net_160_180_Deviation)
    #       )

    plt.figure('Error means')
    plt.clf()
    plt.scatter(Err_x, Err_means)
    plt.draw()
    plt.pause(1)

    plt.figure('Error var')
    plt.clf()
    plt.scatter(Err_x, Err_var)
    plt.draw()
    plt.pause(1)

    plt.figure('Error Analysis')
    plt.clf()
    plt.scatter(X_net, Net_Deviation)
    # plt.scatter(X_2D, _2D_Deviation)
    # plt.scatter(X_3D, _3D_Deviation)
    # plt.scatter(X_Integrated, Integrated)
    # plt.title('Net mean: ' + str(np.round(np.mean(Net_Deviation), 2)))
    # plt.xlabel('Net var: ' + str(np.round(np.var(Net_Deviation), 2)))
    # plt.legend(["Net", "2D", "3D", "Integrated"])
    # plt.legend(["Net", "Integrated"])
    plt.draw()
    plt.pause(1)
    np.save('Dynamic_Err_means_'+winsize+'.npy', Err_means)
    np.save('Dynamic_Err_var_'+winsize+'.npy', Err_var)

    X_net_ = np.linspace(0, len(net_estimation), len(net_estimation))  # X轴坐标数据   0至n，分为n份
    # X_2D_ = np.linspace(0, len(linear_2D), len(linear_2D))
    # X_3D_ = np.linspace(0, len(linear_3D), len(linear_3D))
    # X_h = np.linspace(0, len(H_value), len(H_value))
    plt.figure('Value Analysis')
    plt.clf()
    plt.scatter(X_net_, net_estimation)
    plt.scatter(X_net_, linear_2D[0:amount])
    plt.scatter(X_net_, linear_3D[0:amount])
    plt.scatter(X_net_, H_value[0:amount])
    plt.legend(["Net", "2D", "3D", "H"])#恢复这一段即可
    # plt.draw()
    # plt.pause(1)


    # plt.scatter(H_value[0:amount], net_estimation - H_value[0:amount])
    # plt.scatter(X_nete, H_value[0:amount])
    # plt.scatter(X_nete, net_estimation)
    # plt.legend(["Error"]) #查看误差

    # plt.title('Net mean: ' + str(np.round(np.mean(Net_Deviation), 2)))
    # plt.xlabel('Net var: ' + str(np.round(np.var(Net_Deviation), 2)))
    # plt.legend(["Net", "Integrated"])
    plt.show()


if __name__ == '__main__':

    winsize = "win_10"
    net_path = './'
    model_name = 'Dynamic_State_'+winsize+'_2.pth'  #Dynamic_State_win0_3
    data_path = './'
    data_label = '2_23_25'
    eps = "100"

    # net_path = './Dynamic_s_FixedMap/'
    # model_name = 'Model_Dynamic_s_FixedMap3.pth'
    # data_path = './Dynamic_s_RandomMap/'
    # data_label = '100_21_16_20'

    # net_path = './Original_s_RandomMap/'
    # model_name = 'Model_Original_s_RandomMap3.pth'
    # data_path = './Original_s_RandomMap/'
    # data_label = '100_21_16_16'

    # model = base_path + model_name
    test_net(net_path, model_name, data_path, winsize, data_label, eps)

