
'''
This script includes various mean estimation experiments, such as studying the effects of the different hyperparameters.
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import time
import random
from scipy import stats
from utils import fineMeanEst, privateRangeEst, twoShot
from algos import *


d = 50
mean = [0.0]*d
cov = np.eye(d)
c = [0]*d
r = 10*np.sqrt(d)
p = 0.5
eps = np.sqrt(2.0*p)

Ps1 = [p]
Ps2 = [(1.0/4.0)*p, (3.0/4.0)*p]
Ps3 = [(1.0/8.0)*p, (1.0/8.0)*p, (3.0/4.0)*p]
Ps4 = [(1.0/12.0)*p, (1.0/12.0)*p, (1.0/12.0)*p, (3.0/4.0)*p]

err_nonpr = []
err_naive = []
err_kv = []
err_t1 = []
err_t2 = []
err_t3 = []
err_t4 = []
err_ratio = []
counters = []

for n in np.linspace(1000, 10000, num=12):
    print(n)
    non_pr = []
    means_naive = []
    means_kv = []
    means_t1 = []
    means_t2 = []
    means_t3 = []
    means_t4 = []
    for i in range(100):
        X = np.random.multivariate_normal(mean, cov, int(n))
        non_pr.append(L2(np.mean(X, axis=0)-mean))
        means_naive_coord = []
        means_kv_coord = []
        for j in range(d):
            means_naive_coord.append(fineMeanEst(X[:,j].copy(), 1, r, eps/np.sqrt(d)) - mean[j])
            means_kv_coord.append(twoShot(X[:,j].copy(), np.sqrt(1.0/2.0)*eps/np.sqrt(d), np.sqrt(1.0/2.0)*eps/np.sqrt(d), 0, r, 1) - mean[j])
        means_naive.append(L2(np.asarray(means_naive_coord)))
        means_kv.append(L2(np.asarray(means_kv_coord)))
        means_t1.append(L2(multivariate_mean_iterative(X.copy(), c, r, 1, Ps1)-mean))
        means_t2.append(L2(multivariate_mean_iterative(X.copy(), c, r, 2, Ps2)-mean))
        means_t3.append(L2(multivariate_mean_iterative(X.copy(), c, r, 3, Ps3)-mean))
        means_t4.append(L2(multivariate_mean_iterative(X.copy(), c, r, 4, Ps4)-mean))
    
    err_nonpr.append(stats.trim_mean(non_pr,0.1)) 
    err_naive.append(stats.trim_mean(means_naive,0.1))
    err_kv.append(stats.trim_mean(means_kv,0.1))
    err_t1.append(stats.trim_mean(means_t1,0.1))
    err_t2.append(stats.trim_mean(means_t2,0.1))
    err_t3.append(stats.trim_mean(means_t3,0.1))
    err_t4.append(stats.trim_mean(means_t4,0.1))
    counters.append(n)

np.savetxt("./results/synthetic_mean/counters-1.txt", np.array(counters))
np.savetxt("./results/synthetic_mean/nonpr-1.txt", np.array(err_nonpr))
np.savetxt("./results/synthetic_mean/naive-1.txt", np.array(err_naive))
np.savetxt("./results/synthetic_mean/kv-1.txt", np.array(err_kv))
np.savetxt("./results/synthetic_mean/t1-1.txt", np.array(err_t1))
np.savetxt("./results/synthetic_mean/t2-1.txt", np.array(err_t2))
np.savetxt("./results/synthetic_mean/t3-1.txt", np.array(err_t3))
np.savetxt("./results/synthetic_mean/t4-1.txt", np.array(err_t4))
    
non_private = plt.scatter(x=counters,y=err_nonpr,s=2)
naive = plt.scatter(x=counters,y=err_naive,s=2)
kv = plt.scatter(x=counters,y=err_kv,s=2)
t1 = plt.scatter(x=counters,y=err_t1,s=2)
t2 = plt.scatter(x=counters,y=err_t2,s=2)
t3 = plt.scatter(x=counters,y=err_t3,s=2)
t4 = plt.scatter(x=counters,y=err_t4,s=2)
plt.xlabel('n')
plt.ylabel('Mean Est L2 Error')
plt.legend(( non_private, naive, kv, t1, t2, t3, t4,),
           ('Non-private', 'Naive', 'KV', 't = 1', 't = 2', 't = 3', 't = 4'),
           scatterpoints=1,
           loc='upper right',
           ncol=2,
           fontsize=12)
plt.title('Multivariate Mean Estimation')


#Effect of varying r. Fix parameters d = 50, r = 10*sqrt(d), rho = 0.5. t = 1 through 4. 
d = 50
mean = [0.0]*d
cov = np.eye(d)
c = [0]*d
n = 1000
p = 0.5
eps = np.sqrt(2.0*p)

Ps1 = [p]
Ps2 = [(1.0/4.0)*p, (3.0/4.0)*p]
Ps3 = [(1.0/8.0)*p, (1.0/8.0)*p, (3.0/4.0)*p]
Ps4 = [(1.0/12.0)*p, (1.0/12.0)*p, (1.0/12.0)*p, (3.0/4.0)*p]
Ps10 = [(1.0/36.0)*p,(1.0/36.0)*p,(1.0/36.0)*p,(1.0/36.0)*p,(1.0/36.0)*p,(1.0/36.0)*p,(1.0/36.0)*p,(1.0/36.0)*p,(1.0/36.0)*p,(3.0/4.0)*p]

err_nonpr = []
err_naive = []
err_kv = []
err_t1 = []
err_t2 = []
err_t3 = []
err_t4 = []
err_t10 = []
err_ratio = []
counters = []

for r in np.linspace(np.sqrt(d), 10000*np.sqrt(d), num=10):
    print(r)
    non_pr = []
    means_naive = []
    means_kv = []
    means_t1 = []
    means_t2 = []
    means_t3 = []
    means_t4 = []
    means_t10 = []
    for i in range(100):
        X = np.random.multivariate_normal(mean, cov, int(n))
        non_pr.append(L2(np.mean(X, axis=0)-mean))
        means_naive_coord = []
        means_kv_coord = []
        for j in range(d):
            means_naive_coord.append(fineMeanEst(X[:,j].copy(), 1, r, eps/np.sqrt(d)) - mean[j])
            means_kv_coord.append(twoShot(X[:,j].copy(), np.sqrt(1.0/2.0)*eps/np.sqrt(d), np.sqrt(1.0/2.0)*eps/np.sqrt(d), 0, r, 1) - mean[j])
        means_naive.append(L2(np.asarray(means_naive_coord)))
        means_kv.append(L2(np.asarray(means_kv_coord)))
        means_t1.append(L2(multivariate_mean_iterative(X.copy(), c, r, 1, Ps1)-mean))
        means_t2.append(L2(multivariate_mean_iterative(X.copy(), c, r, 2, Ps2)-mean))
        means_t3.append(L2(multivariate_mean_iterative(X.copy(), c, r, 3, Ps3)-mean))
        means_t4.append(L2(multivariate_mean_iterative(X.copy(), c, r, 4, Ps4)-mean))
        means_t10.append(L2(multivariate_mean_iterative(X.copy(), c, r, 10, Ps10)-mean))
    
    err_nonpr.append(stats.trim_mean(non_pr,0.1)) 
    err_naive.append(stats.trim_mean(means_naive,0.1))
    err_kv.append(stats.trim_mean(means_kv,0.1))
    err_t1.append(stats.trim_mean(means_t1,0.1))
    err_t2.append(stats.trim_mean(means_t2,0.1))
    err_t3.append(stats.trim_mean(means_t3,0.1))
    err_t4.append(stats.trim_mean(means_t4,0.1))
    err_t10.append(stats.trim_mean(means_t10,0.1))
    counters.append(r)
    
np.savetxt("./results/synthetic_mean/counters-2.txt", np.array(counters))
np.savetxt("./results/synthetic_mean/nonpr-2.txt", np.array(err_nonpr))
np.savetxt("./results/synthetic_mean/naive-2.txt", np.array(err_naive))
np.savetxt("./results/synthetic_mean/kv-2.txt", np.array(err_kv))
np.savetxt("./results/synthetic_mean/t1-2.txt", np.array(err_t1))
np.savetxt("./results/synthetic_mean/t2-2.txt", np.array(err_t2))
np.savetxt("./results/synthetic_mean/t3-2.txt", np.array(err_t3))
np.savetxt("./results/synthetic_mean/t4-2.txt", np.array(err_t4))
np.savetxt("./results/synthetic_mean/t10-2.txt", np.array(err_t10))
    
non_private = plt.scatter(x=counters,y=err_nonpr,s=2)
naive = plt.scatter(x=counters,y=err_naive,s=2)
kv = plt.scatter(x=counters,y=err_kv,s=2)
t1 = plt.scatter(x=counters,y=err_t1,s=2)
t2 = plt.scatter(x=counters,y=err_t2,s=2)
t3 = plt.scatter(x=counters,y=err_t3,s=2)
t4 = plt.scatter(x=counters,y=err_t4,s=2)
t10 = plt.scatter(x=counters,y=err_t10,s=2)
plt.xlabel('r')
plt.ylabel('Mean Est L2 Error')
plt.legend(( non_private, naive, kv, t1, t2, t3, t4,t10),
           ('Non-private', 'Naive', 'KV', 't = 1', 't = 2', 't = 3', 't = 4', 't = 10'),
           scatterpoints=1,
           loc='upper right',
           ncol=2,
           fontsize=12)
plt.title('Multivariate Mean Estimation')

#Varying privacy. Fix parameters d = 50, r = 10*sqrt(d), n = 2000. t = 1 through 4.
d = 50
mean = [0.0]*d
cov = np.eye(d)
c = [0]*d
r = 10*np.sqrt(d)
n = 2000

err_nonpr = []
err_naive = []
err_kv = []
err_t1 = []
err_t2 = []
err_t3 = []
err_t4 = []
err_ratio = []
counters = []

for p in np.geomspace(0.005, 0.5, num=10):
    eps = np.sqrt(2.0*p)
    print(p)
    Ps1 = [p]
    Ps2 = [(1.0/4.0)*p, (3.0/4.0)*p]
    Ps3 = [(1.0/8.0)*p, (1.0/8.0)*p, (3.0/4.0)*p]
    Ps4 = [(1.0/12.0)*p, (1.0/12.0)*p, (1.0/12.0)*p, (3.0/4.0)*p]
    non_pr = []
    means_naive = []
    means_kv = []
    means_t1 = []
    means_t2 = []
    means_t3 = []
    means_t4 = []
    for i in range(100):
        X = np.random.multivariate_normal(mean, cov, int(n))
        non_pr.append(L2(np.mean(X, axis=0)-mean))
        means_naive_coord = []
        means_kv_coord = []
        for j in range(d):
            means_naive_coord.append(fineMeanEst(X[:,j].copy(), 1, r, eps/np.sqrt(d)) - mean[j])
            means_kv_coord.append(twoShot(X[:,j].copy(), np.sqrt(1.0/2.0)*eps/np.sqrt(d), np.sqrt(1.0/2.0)*eps/np.sqrt(d), 0, r, 1) - mean[j])
        means_naive.append(L2(np.asarray(means_naive_coord)))
        means_kv.append(L2(np.asarray(means_kv_coord)))
        means_t1.append(L2(multivariate_mean_iterative(X.copy(), c, r, 1, Ps1)-mean))
        means_t2.append(L2(multivariate_mean_iterative(X.copy(), c, r, 2, Ps2)-mean))
        means_t3.append(L2(multivariate_mean_iterative(X.copy(), c, r, 3, Ps3)-mean))
        means_t4.append(L2(multivariate_mean_iterative(X.copy(), c, r, 4, Ps4)-mean))
    
    err_nonpr.append(stats.trim_mean(non_pr,0.1)) 
    err_naive.append(stats.trim_mean(means_naive,0.1))
    err_kv.append(stats.trim_mean(means_kv,0.1))
    err_t1.append(stats.trim_mean(means_t1,0.1))
    err_t2.append(stats.trim_mean(means_t2,0.1))
    err_t3.append(stats.trim_mean(means_t3,0.1))
    err_t4.append(stats.trim_mean(means_t4,0.1))
    counters.append(p)
    
np.savetxt("./results/synthetic_mean/counters-6.txt", np.array(counters))
np.savetxt("./results/synthetic_mean/nonpr-6.txt", np.array(err_nonpr))
np.savetxt("./results/synthetic_mean/naive-6.txt", np.array(err_naive))
np.savetxt("./results/synthetic_mean/kv-6.txt", np.array(err_kv))
np.savetxt("./results/synthetic_mean/t1-6.txt", np.array(err_t1))
np.savetxt("./results/synthetic_mean/t2-6.txt", np.array(err_t2))
np.savetxt("./results/synthetic_mean/t3-6.txt", np.array(err_t3))
np.savetxt("./results/synthetic_mean/t4-6.txt", np.array(err_t4)) 
    
non_private = plt.scatter(x=counters,y=err_nonpr,s=2)
naive = plt.scatter(x=counters,y=err_naive,s=2)
kv = plt.scatter(x=counters,y=err_kv,s=2)
t1 = plt.scatter(x=counters,y=err_t1,s=2)
t2 = plt.scatter(x=counters,y=err_t2,s=2)
t3 = plt.scatter(x=counters,y=err_t3,s=2)
t4 = plt.scatter(x=counters,y=err_t4,s=2)
plt.xlabel('n')
plt.ylabel('Mean Est L2 Error')
plt.legend(( non_private,  t2, t3, t4,),
           ('Non-private', 'Naive', 'KV', 't = 1', 't = 2', 't = 3', 't = 4'),
           scatterpoints=1,
           loc='upper right',
           ncol=2,
           fontsize=12)
plt.title('Multivariate Mean Estimation')

# Testing the role of the tails of the distribution.
d = 50
mean = [10.0]*d
cov = np.eye(d)
c = [0.0]*d
r = 10.0*np.sqrt(d)
p = .5
t = 2
Ps = [p/4.0,3.0*p/4.0]

err_gauss_nonpr = []
err_gauss_pr = []
err_lap_pr = []
err_t_pr = []
counters = []

for n in np.linspace(1000, 10000, num=12):
    print(int(n))
    gauss_nonpr = []
    gauss_pr = []
    lap_pr = []
    t_pr = []
    for i in range(100):
        ## Gaussian
        X = np.random.multivariate_normal(mean, cov, int(n))
        gauss_nonpr.append(L2(np.mean(X, axis=0) - mean))
        gauss_pr.append(L2(multivariate_mean_iterative(X.copy(), c, r, 2,Ps) - mean))
        
        ## Laplacian
        X = np.random.laplace(mean[0], cov[0,0]/np.sqrt(2), (int(n),d))
        lap_pr.append(L2(multivariate_mean_iterative(X.copy(), c, r, 2,Ps) - mean))
        
        ## t distribution with 3 df
        X = np.zeros((int(n),d))
        for i in range(int(n)):
            X[i] = np.random.standard_t(df = 3,size=d)/np.sqrt(3.0/1.0) + mean
        t_pr.append(L2(multivariate_mean_iterative(X.copy(), c, r, 2, Ps) - mean))
    
    err_gauss_nonpr.append(stats.trim_mean(gauss_nonpr,0.1)) 
    err_gauss_pr.append(stats.trim_mean(gauss_pr,0.1)) 
    err_lap_pr.append(stats.trim_mean(lap_pr,0.1)) 
    err_t_pr.append(stats.trim_mean(t_pr,0.1))
    counters.append(n)

np.savetxt("./results/synthetic_mean/counters-7.txt", np.array(counters))
np.savetxt("./results/synthetic_mean/gauss_nonpr-7.txt", np.array(err_gauss_nonpr))
np.savetxt("./results/synthetic_mean/gauss_pr-7.txt", np.array(err_gauss_pr))
np.savetxt("./results/synthetic_mean/lap_pr-7.txt", np.array(err_lap_pr))
np.savetxt("./results/synthetic_mean/t_pr-7.txt", np.array(err_t_pr))
    
gauss_non_private = plt.scatter(x=counters,y=err_gauss_nonpr,s=2)
gauss_private = plt.scatter(x=counters,y=err_gauss_pr,s=2)
lap_private = plt.scatter(x=counters,y=err_lap_pr,s=2)
t_private = plt.scatter(x=counters,y=err_t_pr,s=2)

plt.legend(( gauss_non_private,  gauss_private, lap_private, t_private),
           ('Non Private','Gauss','Laplace','Student3'),
           scatterpoints=1,
           loc='upper right',
           ncol=2,
           fontsize=10)

plt.xlabel('n')
plt.ylabel('Mean Est L2 Error')
plt.title('Multivariate Mean Estimation')
