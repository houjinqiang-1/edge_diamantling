# -*- coding: utf-8 -*-

from FINDER_torch import FINDER
import os,sys
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import random
os.chdir(sys.path[0])

def main():
    dqn = FINDER()
    dqn.Train()


if __name__=="__main__":
    cudnn.benchmark = False  #这行代码将 cuDNN 的 benchmark 模式设置为 False。
                             #cuDNN 的 benchmark 模式会根据输入的大小和形状自动选择最优的卷积算法。
                             #禁用 benchmark 模式可以确保每次运行时都使用相同的卷积算法,提高可重复性
    cudnn.deterministic = True
                             #这行代码将 cuDNN 的确定性模式设置为 True。
                             #确定性模式确保每次运行时,cuDNN 的计算结果都是确定的和可重复的。
                             #启用确定性模式可能会影响性能,但对于需要结果可重复的场景非常有用。
    random.seed(0)
                             #这行代码为 Python 的内置 random 模块设置随机数种子为 0。
                             #设置随机数种子可以确保每次运行时,random 模块生成的随机数序列都是相同的。
    np.random.seed(0)
                             #这行代码为 NumPy 的随机数生成器设置随机数种子为 0。
                             #NumPy 是一个常用的科学计算库,设置其随机数种子可以确保每次运行时,NumPy 生成的随机数序列都是相同的。
    torch.manual_seed(0)
                             #这行代码为 PyTorch 的 CPU 随机数生成器设置随机数种子为 0。
                             #设置 PyTorch 的随机数种子可以确保每次运行时,PyTorch 在 CPU 上生成的随机数序列都是相同的。
    torch.cuda.manual_seed(0)
                             #这行代码为 PyTorch 的 CUDA 随机数生成器设置随机数种子为 0。
                             #设置 CUDA 随机数种子可以确保每次运行时,PyTorch 在 GPU 上生成的随机数序列都是相同的。
    main()
