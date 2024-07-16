import numpy as np
import time
import Hyperbolic as hyp


#N 是节点数量
def GMM(N,rand_g):
    ############################# Adjust GMM Parameters ###############################
    #random number generator seed

    L=2
    nu=0.2  # v
    g=rand_g

    gamma1=2.5
    kbar1=6.0
    T1=0.4

    gamma2=2.5
    kbar2=6.0
    T2=0.4
    ####################################################################################


    t1=time.time()


    kmin1=hyp.CalculateKmin(kbar1, gamma1)#计算最小度数
    #print("kmin1: ",kmin1)

    C1=hyp.CalculateC(kbar1, T1, gamma1)
    #print("C1: ",C1)

    R1=hyp.CalculateR(N,C1)
    #print("R1: ",R1)

    kmin2=hyp.CalculateKmin(kbar2, gamma2)
    #print("kmin2: ",kmin2)

    C2=hyp.CalculateC(kbar2, T2, gamma2)
    #print("C2: ",C2)

    R2=hyp.CalculateR(N,C2)
    #print("R2: ",R2)


    kappa1=hyp.SampleKappa(N, kmin1, gamma1)

    kappa2=hyp.SampleConditionalKappa(N, nu, kappa1[:], kmin1, gamma1, kmin2, gamma2)

    theta1=hyp.SampleTheta(N)

    theta2=hyp.SampleConditionalTheta(N, g, theta1[:])


    r1=hyp.ChangeVariablesFromS1ToH2(N, kappa1[:], R1, kmin1)
    r2=hyp.ChangeVariablesFromS1ToH2(N, kappa2[:], R2, kmin2)

    # hyp.PrintCoordinates(r1[:],theta1[:],kappa1[:],"coords1.txt")
    # hyp.PrintCoordinates(r2[:],theta2[:],kappa2[:],"coords2.txt")


    #Do this part Fast in C++

    links1=hyp.CreateNetworks(kappa1,theta1,T1,kbar1)
    links2=hyp.CreateNetworks(kappa2,theta2,T2,kbar2)

    #在这里加上嵌入网络

    return links1,links2



