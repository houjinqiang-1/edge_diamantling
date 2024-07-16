import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch.backends.cudnn as cudnn
import torch
import random
import testReal
import os

node_num = {'Padgett-Florentine-Families_multiplex': 16,
            'AirTrain': 69,  # [(1,2)]
            'Brain': 90,  # [(1,2)]
            # 'fao_trade_multiplex': 214,
            'Phys': 246,  # [(1,2), (1,3), (2,3)]
            'celegans_connectome_multiplex': 279,  # [(1,2), (1,3), (2,3)]
            # 'HumanMicrobiome_multiplex': 305,
            # 'xenopus_genetic_multiplex': 416,
            # 'pierreauger_multiplex': 514,
            'rattus_genetic_multiplex': 2640,  # [(1,2)]
            'sacchpomb_genetic_multiplex': 4092,  # [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4),(3,5)]
            'drosophila_genetic_multiplex': 8215,  # [(1,2)]
            'arxiv_netscience_multiplex': 14489,  # [(1,2),(1,4),(1,5),(2,4),(2,5),(2,6),(2,8),(3,4)]
            'Internet': 4202010733,
            'adj':471
            }


nums_dict = {'AirTrain': [(1,2)],
             'Brain': [(1,2)],
             'Phys': [(1,2), (1,3), (2,3)],  # [(1,2), (1,3), (2,3)],
             'celegans_connectome_multiplex': [(2,3)],
             'rattus_genetic_multiplex': [(1,2)],
             'sacchpomb_genetic_multiplex': [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4),(3,5)],
             'drosophila_genetic_multiplex': [(1,2)],
             'arxiv_netscience_multiplex': [(1,4)], #[(1,2)]#[(1,2),(1,4),(1,5),(2,4),(2,5),(2,6),(2,8),(3,4)]
             'Padgett-Florentine-Families_multiplex':[(1,2)],
             'adj':[(1,2)]
             }

#画出HDA和OUR的ANC图
def judge():
    # data_name = 'celegans_connectome_multiplex'
    for j in range(100):
        data_name_1 = 'adj1'
        data_name_2 = 'adj2'
        # num0 = 0
        # num1 = 1
        # num2 = 2
        path_1 = './data/syn_400-500' + f'/{data_name_1}_{j}'
        path_2 = './data/syn_400-500' + f'/{data_name_2}_{j}'
        if not os.path.exists(f'./ANC/{j}'):
            os.makedirs(f'./ANC/{j}')
        MCC1 = np.loadtxt(f'./synthetic/score_{j}', usecols=(0,))
        # 将MCC1保存到txt文件中
        np.savetxt(f'./ANC/{j}/our_mcc.txt', MCC1, fmt='%f')
        solution = np.loadtxt(f'./synthetic/soulation_{j}',usecols=(0,))
        np.savetxt(f'./ANC/{j}/our_solution.txt', solution, fmt='%d')
        MCC2_1 = np.load(path_1 + '.npy')
        G1 = nx.from_numpy_array(MCC2_1)
        MCC2_2 = np.load(path_2 + '.npy' )
        G2 = nx.from_numpy_array((MCC2_2))

        MCC2 = G1.number_of_edges() + G2.number_of_edges()
        print(MCC2)
        # 计算填充数量
        fill_count = MCC2 - len(MCC1)
        # 扩展MCC1的长度并填充
        if fill_count > 0:
            fill_value = MCC1[-1]  # 使用MCC1的最后一个值填充
            MCC1 = np.append(MCC1, np.full(fill_count, fill_value))
        x1 = np.ones(len(MCC1))
        for i in range(0,len(MCC1)):
            x1[i] = i/len(MCC1)
        x2 = np.ones(MCC2)
        for i in range(0,MCC2):
            x2[i] = i/MCC2

        plt.figure()
        plt.plot(x1,MCC1,label='ANC')
        # plt.plot(x2,MCC2,label='ANC')
        plt.ylabel('Value')
        plt.title(f'data:{data_name_1}-{j},iter:best radio:0.001')
        plt.legend()
        # 设置 x 轴范围为 0 到 0.2
        plt.xlim(0, 1)
        plt.savefig(f'./ANC/{j}/img_best.png')
        plt.show()

# testReal.main()
judge()
