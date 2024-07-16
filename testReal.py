

from FINDER_torch import FINDER
import numpy as np
import time
import os
import pandas as pd
import torch.backends.cudnn as cudnn
import torch
import random
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

g_type = "GMM"


def GetSolution(STEPRATIO, MODEL_FILE):
    ######################################################################################################################
    ##................................................Get Solution (model).....................................................
    dqn = FINDER()
    data_test_path = './data/'
    #data_test_name = ['Digg','HI-II-14']
    data_test_name = ['celegans_connectome_multiplex']
    date_test_n = [279]
    data_test_layer = [(2, 3)]
    model_file = './models_edge/{}'.format(MODEL_FILE)
    ## save_dir
    save_dir = './results/new_edge/real'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    ## begin computing...
    print ('The best model is :%s'%(model_file))
    dqn.LoadModel(model_file)
    df = pd.DataFrame(np.arange(2*len(data_test_name)).reshape((2,len(data_test_name))),index=['time','score'], columns=data_test_name)
    #################################### modify to choose which stepRatio to get the solution
    stepRatio = STEPRATIO
    for j in range(len(data_test_name)):
        print ('\nTesting dataset %s'%data_test_name[j])
        #data_test = data_test_path + data_test_name[j] + '.txt'
        data_test = data_test_path + data_test_name[j] + '.edges'
        #solution, time = dqn.EvaluateRealData(model_file, data_test, save_dir, stepRatio)
        solution, time, score= dqn.EvaluateRealData(model_file, data_test, save_dir, stepRatio, date_test_n[j], data_test_layer[j])
        df.iloc[0,j] = time
        df.iloc[1,j] = score
        print('Data:%s, time:%.2f, score:%.6f'%(data_test_name[j], time, score))
    save_dir_local = save_dir + '/StepRatio_%.4f' % stepRatio
    if not os.path.exists(save_dir_local):
        os.mkdir(save_dir_local)
    df.to_csv(save_dir_local + '/sol_time&score_%s.csv'% data_test_name[j], encoding='utf-8', index=False)

# def EvaluateSolution(STEPRATIO, MODEL_FILE_CKPT, STRTEGYID):
#     #######################################################################################################################
#     ##................................................Evaluate Solution.....................................................
#     dqn = FINDER()
#     data_test_path = './data/real/'
#     #data_test_name = ['Digg', 'HI-II-14']
#     data_test_name = ['ACM3025']
#
#     save_dir = '../results/FINDER_CN/real/StepRatio_%.4f/'%STEPRATIO
#     ## begin computing...
#     df = pd.DataFrame(np.arange(2 * len(data_test_name)).reshape((2, len(data_test_name))), index=['solution', 'time'], columns=data_test_name)
#     for i in range(len(data_test_name)):
#         print('\nEvaluating dataset %s' % data_test_name[i])
#         #data_test = data_test_path + data_test_name[i] + '.txt'
#         data_test = data_test_path + data_test_name[i] + '.mat'
#         solution = save_dir + data_test_name[i] + '.txt'
#         t1 = time.time()
#         # strategyID: 0:no insert; 1:count; 2:rank; 3:multiply
#         ################################## modify to choose which strategy to evaluate
#         strategyID = STRTEGYID
#         #score, MaxCCList = dqn.EvaluateSol(data_test, solution, strategyID, reInsertStep=0.001)
#         score, MaxCCList = dqn.EvaluateSol_ACM3025(adj1, adj2, data_test, solution, strategyID, reInsertStep=0.001)
#         t2 = time.time()
#         df.iloc[0, i] = score
#         df.iloc[1, i] = t2 - t1
#         result_file = save_dir + '/MaxCCList_Strategy_' + data_test_name[i] + '.txt'
#         with open(result_file, 'w') as f_out:
#             for j in range(len(MaxCCList)):
#                 f_out.write('%.8f\n' % MaxCCList[j])
#         print('Data: %s, score:%.6f' % (data_test_name[i], score))
#     df.to_csv(save_dir + '/solution_score.csv', encoding='utf-8', index=False)


def main():
    model_file_ckpt = 'Mcc_truncNormal-TORCH-Model_barabasi_albert_30_50/nrange_30_50_iter_490000.ckpt'
    GetSolution(0, model_file_ckpt)


if __name__=="__main__":
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    main()
