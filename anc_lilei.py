import numpy as np
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch
import random
import os

#datanames = ['Synthetic-50','Synthetic-100','Synthetic-200','Synthetic-500']
#datanames = ['Lazega-Law-Firm_multiplex']
#datanames = ['celegans_connectome_multiplex','CS-Aarhus_multiplex','EUAirTransportation_multiplex','Kapferer-Tailor-Shop_multiplex','Krackhardt-High-Tech_multiplex','Lazega-Law-Firm_multiplex','sacchpomb_genetic_multiplex','fao_trade_multiplex']
#画出HDA和OUR的ANC图
#datanames = ['celegans_connectome_multiplex','CS-Aarhus_multiplex','Kapferer-Tailor-Shop_multiplex','Lazega-Law-Firm_multiplex','Synthetic-50','Synthetic-100','Synthetic-200','Synthetic-500']
x_lims = [0.5,0.3,0.5,1,0.1,0.1,0.1,0.1]
#datanames = ['Lazega-Law-Firm_multiplex']
def judgeCN():
    types = ['OUR','HDA','HCA','CI']
    import dict1
    node_num = dict1.node_num
    nums_dict = dict1.nums_dict
    colors = {'OUR': 'red', 'HDA': 'green', 'HCA': 'orange', 'CI': 'blue'}
    #for dataname in datanames:
    for i in range(0,8):
        dataname =datanames[i]
        my_xlim = x_lims[i]
        for nums in nums_dict[dataname]:
            num1 = nums[0]
            num2 = nums[1]

            plt.figure()
            # 在图上标记两个指定的数
            x_position = 0.05 * plt.xlim()[1]  # 设置 x 位置为图的左侧
            y_position = 0.95 * plt.ylim()[1]  # 设置 y 位置为图的顶部
            for type in types:
                MCCs = np.loadtxt(f'{type}/CN/{dataname}/{num1}-{num2}/{dataname}-{num1}-{num2}-{type}-CN-pairs.txt',
                    usecols=(0,))
                score = np.loadtxt(f'{type}/CN/{dataname}/{num1}-{num2}/{dataname}-{num1}-{num2}-{type}-CN-score.txt',
                    usecols=(0,))
                score = np.round(score, 5)
                x = np.ones(len(MCCs))
                for i in range(0, len(MCCs)):
                    x[i] = i / (len(MCCs) - 1)
                plt.plot(x, MCCs, label=f'{type}',color=colors[type])
                #plt.text(x_position, y_position, f'{type}_ANC: {score}', fontsize=8, color='black')
                y_position -=  0.05 * plt.ylim()[1]
            plt.xlabel(f'Fraction of key players')
            plt.ylabel('residual CN connectivity')
            plt.title(f'data:{dataname}-{num1}-{num2}')
            plt.legend()
            plt.grid(True)  # Add grid
            plt.xlim(0, my_xlim)
            plt.savefig(f'./ANC/报告3/{dataname}-{num1}-{num2}.png')
            #plt.show()


def judgeND():
    # types = ['OUR','HDA','HCA','CI','minSum','FINDER','NIRM','HBA']
    #types = ['OUR', 'HDA', 'HCA', 'CI','HBA']
    types = ['data']
    import dict1
    node_num = dict1.node_num
    nums_dict = dict1.nums_dict
    #colors = {'OUR': 'red', 'HDA': 'green', 'HCA': 'orange', 'CI': 'blue',}
    '''
    datanames = ['AirTrain', 'Brain', 'Phys', 'CS-Aarhus_multiplex', 'Kapferer-Tailor-Shop_multiplex',
                 'Krackhardt-High-Tech_multiplex', 'humanHIV1_genetic_multiplex', 'Synthetic-50', 'Synthetic-100',
                 'Synthetic-200', 'Synthetic-500']
    '''
    '''
    datanames = ['bos_genetic_multiplex','celegans_connectome_multiplex','CKM-Physicians-Innovation_multiplex','CS-Aarhus_multiplex','EUAirTransportation_multiplex','HumanMicrobiome_multiplex','Krackhardt-High-Tech_multiplex'
    ,'london_transport_multiplex','Padgett-Florentine-Families_multiplex','sacchpomb_genetic_multiplex','Vickers-Chan-7thGraders_multiplex','fb-tw','yeast_landscape_multiplex','arxiv_netscience_multiplex']
    '''
    # datanames = ['bos_genetic_multiplex','celegans_connectome_multiplex','CKM-Physicians-Innovation_multiplex','CS-Aarhus_multiplex','EUAirTransportation_multiplex','HumanMicrobiome_multiplex','Krackhardt-High-Tech_multiplex'
    # ,'london_transport_multiplex','Padgett-Florentine-Families_multiplex','sacchpomb_genetic_multiplex','Vickers-Chan-7thGraders_multiplex','fb-tw']
    datanames = ['adj']
    for dataname in datanames:
        for nums in nums_dict[dataname]:
            num1 = nums[0]
            num2 = nums[1]
            plt.figure()
            # 在图上标记两个指定的数
            x_position = 0.05 * plt.xlim()[1]  # 设置 x 位置为图的左侧
            y_position = 0.95 * plt.ylim()[1]  # 设置 y 位置为图的顶部
            for type in types:
                MCCs = np.loadtxt(f'{type}/ND/{dataname}/{num1}-{num2}/{dataname}-{num1}-{num2}-{type}-ND-pairs.txt',
                    usecols=(0,))
                score = np.loadtxt(f'{type}/ND/{dataname}/{num1}-{num2}/{dataname}-{num1}-{num2}-{type}-ND-score.txt',
                    usecols=(0,))
                score = np.round(score, 5)
                x = np.ones(len(MCCs))
                for i in range(0, len(MCCs)):
                    x[i] = i / (len(MCCs) - 1)
                plt.plot(x, MCCs, label=f'{type}')
                plt.text(x_position, y_position, f'{type}_ANC: {score}', fontsize=8, color='black')
                y_position -=  0.05 * plt.ylim()[1]
            plt.xlabel(f'Fraction of key players')
            plt.ylabel('residual ND connectivity')
            plt.title(f'data:{dataname}-{num1}-{num2}')
            plt.legend()
            plt.xlim(0, 1)
            plt.grid(True)  # Add grid
            plt.savefig(f'./directed_ANC/ND/{dataname}-{num1}-{num2}.png')
            #plt.show()
judgeND()
#judgeCN()
