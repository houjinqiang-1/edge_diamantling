#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from FINDER_torch import FINDER
from tqdm import tqdm

def main():
    dqn = FINDER()
    data_test_path = './data/synthetic/uniform_cost/'
#     data_test_name = ['30-50', '50-100', '100-200', '200-300', '300-400', '400-500']
#     data_test_name =['30-50', '50-100']
    data_test_name = ['test_400-500']
    model_file = './models_edge/Mcc_truncNormal-TORCH-Model_barabasi_albert_30_50/nrange_30_50_iter_500000.ckpt'
    # model_file = '../FINDER_ND/models/k=6_TORCH-Model_GMM_30_50/nrange_30_50_iter_10000.ckpt'
    file_path = './results_edge/FINDER_CN/synthetic'
#     if not os.path.exists(file_path):
    if not os.path.exists('./results_edge/FINDER_CN'):
        os.makedirs('./results_edge/FINDER_CN')
    if not os.path.exists('./results_edge/FINDER_CN/synthetic'):
        os.makedirs('./results_edge/FINDER_CN/synthetic')
        
    with open('%s/result.txt'%file_path, 'w') as fout:
        for i in tqdm(range(len(data_test_name))):
            data_test = data_test_path + data_test_name[i]
            score_mean, score_std, time_mean, time_std = dqn.Evaluate(data_test, model_file)
            fout.write('%.2f±%.2f,' % (score_mean , score_std))
            fout.flush()
            print('data_test_%s has been tested!' % data_test_name[i])


if __name__=="__main__":
    main()
