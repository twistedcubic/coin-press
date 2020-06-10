
'''
This script includes various covariance estimation experiments, such as studying the effects of the different hyperparameters.
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import torch
import utils
from scipy import stats
from algos import *

#Headline experiments, spherical. Vary n. Fix parameters d = 10, u = 10*sqrt(d), rho = 0.5. t = 1 through 4.
args = utils.parse_args()
args.d = 10
dist_mean = torch.zeros(args.d)
dist_cov = torch.eye(args.d)
args.u = 10*np.sqrt(args.d)
args.total_budget = .5

Ps1 = [args.total_budget]
Ps2 = [(1.0/4.0)*args.total_budget, (3.0/4.0)*args.total_budget]
Ps3 = [(1.0/8.0)*args.total_budget, (1.0/8.0)*args.total_budget, (3.0/4.0)*args.total_budget]
Ps4 = [(1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (3.0/4.0)*args.total_budget]
Ps5 = [(1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (3.0/4.0)*args.total_budget]

n_l = np.linspace(3000, 8000, num=12)
err_nonpr = []
err_t1 = []
err_t2 = []
err_t3 = []
err_t4 = []
err_t5 = []

for i, n in enumerate(n_l):
    args.n = int(n)
    non_pr = []
    covs_t1 = []
    covs_t2 = []
    covs_t3 = []
    covs_t4 = []
    covs_t5 = []
    print(n)
    for i in range(100):
        if i % 50 == 0: print(i)
        X = torch.distributions.MultivariateNormal(dist_mean, dist_cov).sample((args.n,))
        non_pr.append(mahalanobis_dist(utils.cov(X.clone()), dist_cov))

        args.t = 1
        args.rho = Ps1
        covs_t1.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
        
        args.t = 2
        args.rho = Ps2
        covs_t2.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
        
        args.t = 3
        args.rho = Ps3
        covs_t3.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
          
        args.t = 4
        args.rho = Ps4
        covs_t4.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
                
        args.t = 5
        args.rho = Ps5
        covs_t5.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
        
    err_nonpr.append(stats.trim_mean(non_pr,0.1))
    err_t1.append(stats.trim_mean(covs_t1,0.1))
    err_t2.append(stats.trim_mean(covs_t2,0.1))
    err_t3.append(stats.trim_mean(covs_t3,0.1))
    err_t4.append(stats.trim_mean(covs_t4,0.1))
    err_t5.append(stats.trim_mean(covs_t5,0.1))
    
np.savetxt("./results/synthetic_cov/n-1.txt", np.array(n_l))
np.savetxt("./results/synthetic_cov/nonpr-1.txt", np.array(err_nonpr))
np.savetxt("./results/synthetic_cov/t1-1.txt", np.array(err_t1))
np.savetxt("./results/synthetic_cov/t2-1.txt", np.array(err_t2))
np.savetxt("./results/synthetic_cov/t3-1.txt", np.array(err_t3))
np.savetxt("./results/synthetic_cov/t4-1.txt", np.array(err_t4))    
np.savetxt("./results/synthetic_cov/t5-1.txt", np.array(err_t5))    
    
fig, ax = plt.subplots()
ax.plot(n_l, err_nonpr, marker="x", label='Non-private', color='#1f77b4')
ax.plot(n_l, err_t1, marker="x", label='t = 1', color='#9467bd')
ax.plot(n_l, err_t2, marker="x", label='t = 2', color='#d62728')
ax.plot(n_l, err_t3, marker="x", label='t = 3', color='#e377c2')
ax.plot(n_l, err_t4, marker="x", label='t = 4', color='#7f7f7f')
ax.plot(n_l, err_t5, marker="x", label='t = 5', color='#bcbd22')

ax.set_xlabel('n')
ax.set_ylabel('Frobenius Error')
ax.set_title("Comparison")
ax.legend()


#Effect of varying u, spherical. Fix parameters d = 10, n = 7000, rho = 0.5. t = 1 through 4. 
args = utils.parse_args()
args.d = 10
dist_mean = torch.zeros(args.d)
dist_cov = 1*torch.eye(args.d)
args.n = 7000
args.total_budget = 0.5

Ps1 = [args.total_budget]
Ps2 = [(1.0/4.0)*args.total_budget, (3.0/4.0)*args.total_budget]
Ps3 = [(1.0/8.0)*args.total_budget, (1.0/8.0)*args.total_budget, (3.0/4.0)*args.total_budget]
Ps4 = [(1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (3.0/4.0)*args.total_budget]
Ps5 = [(1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (1.0/16.0)*args.total_budget, (3.0/4.0)*args.total_budget]


u_l = np.geomspace(np.sqrt(args.d), 10000*np.sqrt(args.d), num=10)
err_nonpr = []
err_t1 = []
err_t2 = []
err_t3 = []
err_t4 = []
err_t5 = []

for i, u in enumerate(u_l):
    args.u = u
    non_pr = []
    covs_t1 = []
    covs_t2 = []
    covs_t3 = []
    covs_t4 = []
    covs_t5 = []
    print(u)
    for i in range(100):
        if i % 50 == 0: print(i)
        X = torch.distributions.MultivariateNormal(dist_mean, dist_cov).sample((args.n,))
        non_pr.append(mahalanobis_dist(utils.cov(X.clone()), dist_cov))

        args.t = 1
        args.rho = Ps1
        covs_t1.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
        
        args.t = 2
        args.rho = Ps2
        covs_t2.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
        
        args.t = 3
        args.rho = Ps3
        covs_t3.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
          
        args.t = 4
        args.rho = Ps4
        covs_t4.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
        
        args.t = 5
        args.rho = Ps5
        covs_t5.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
        
    err_nonpr.append(stats.trim_mean(non_pr,0.1))
    err_t1.append(stats.trim_mean(covs_t1,0.1))
    err_t2.append(stats.trim_mean(covs_t2,0.1))
    err_t3.append(stats.trim_mean(covs_t3,0.1))
    err_t4.append(stats.trim_mean(covs_t4,0.1))
    err_t5.append(stats.trim_mean(covs_t5,0.1))
    
np.savetxt("./results/synthetic_cov/u-2.txt", np.array(u_l))
np.savetxt("./results/synthetic_cov/nonpr-2.txt", np.array(err_nonpr))
np.savetxt("./results/synthetic_cov/t1-2.txt", np.array(err_t1))
np.savetxt("./results/synthetic_cov/t2-2.txt", np.array(err_t2))
np.savetxt("./results/synthetic_cov/t3-2.txt", np.array(err_t3))
np.savetxt("./results/synthetic_cov/t4-2.txt", np.array(err_t4))
np.savetxt("./results/synthetic_cov/t5-2.txt", np.array(err_t5))       
    
fig, ax = plt.subplots()
ax.plot(u_l, err_nonpr, marker="x", label='Non-private', color='#1f77b4')
ax.plot(u_l, err_t1, marker="x", label='t = 1', color='#9467bd')
ax.plot(u_l, err_t2, marker="x", label='t = 2', color='#d62728')
ax.plot(u_l, err_t3, marker="x", label='t = 3', color='#e377c2')
ax.plot(u_l, err_t4, marker="x", label='t = 4', color='#7f7f7f')

ax.set_xlabel('K')
ax.set_ylabel('Frobenius Error')
ax.set_title("Comparison")
ax.legend()


#Low dimension, spherical. Vary n. Fix parameters d = 2, u = 10*sqrt(d), rho = 0.5. t = 1 through 4.
args = utils.parse_args()
args.d = 2
dist_mean = torch.zeros(args.d)
dist_cov = torch.eye(args.d)
args.u = 10*np.sqrt(args.d)
args.total_budget = .5

Ps1 = [args.total_budget]
Ps2 = [(1.0/4.0)*args.total_budget, (3.0/4.0)*args.total_budget]
Ps3 = [(1.0/8.0)*args.total_budget, (1.0/8.0)*args.total_budget, (3.0/4.0)*args.total_budget]
Ps4 = [(1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (3.0/4.0)*args.total_budget]

n_l = np.linspace(1000, 3000, num=12)
err_nonpr = []
err_t1 = []
err_t2 = []
err_t3 = []
err_t4 = []

for i, n in enumerate(n_l):
    args.n = int(n)
    non_pr = []
    covs_t1 = []
    covs_t2 = []
    covs_t3 = []
    covs_t4 = []
    print(n)
    for i in range(100):
        if i % 50 == 0: print(i)
        X = torch.distributions.MultivariateNormal(dist_mean, dist_cov).sample((args.n,))
        non_pr.append(mahalanobis_dist(utils.cov(X.clone()), dist_cov))

        args.t = 1
        args.rho = Ps1
        covs_t1.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
        
        args.t = 2
        args.rho = Ps2
        covs_t2.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
        
        args.t = 3
        args.rho = Ps3
        covs_t3.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
          
        args.t = 4
        args.rho = Ps4
        covs_t4.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
                        
    err_nonpr.append(stats.trim_mean(non_pr,0.1))
    err_t1.append(stats.trim_mean(covs_t1,0.1))
    err_t2.append(stats.trim_mean(covs_t2,0.1))
    err_t3.append(stats.trim_mean(covs_t3,0.1))
    err_t4.append(stats.trim_mean(covs_t4,0.1))
    
np.savetxt("./results/synthetic_cov/n-3.txt", np.array(n_l))
np.savetxt("./results/synthetic_cov/nonpr-3.txt", np.array(err_nonpr))
np.savetxt("./results/synthetic_cov/t1-3.txt", np.array(err_t1))
np.savetxt("./results/synthetic_cov/t2-3.txt", np.array(err_t2))
np.savetxt("./results/synthetic_cov/t3-3.txt", np.array(err_t3))
np.savetxt("./results/synthetic_cov/t4-3.txt", np.array(err_t4))    
    
fig, ax = plt.subplots()
ax.plot(n_l, err_nonpr, marker="x", label='Non-private', color='#1f77b4')
ax.plot(n_l, err_t1, marker="x", label='t = 1', color='#9467bd')
ax.plot(n_l, err_t2, marker="x", label='t = 2', color='#d62728')
ax.plot(n_l, err_t3, marker="x", label='t = 3', color='#e377c2')
ax.plot(n_l, err_t4, marker="x", label='t = 4', color='#7f7f7f')

ax.set_xlabel('n')
ax.set_ylabel('Frobenius Error')
ax.set_title("Comparison")
ax.legend()


#High dimension, spherical. Vary n. Fix parameters d = 100, u = 10*sqrt(d), rho = 0.5. t = 1 through 4.
args = utils.parse_args()
args.d = 100
dist_mean = torch.zeros(args.d)
dist_cov = torch.eye(args.d)
args.u = 10*np.sqrt(args.d)
args.total_budget = .5

Ps1 = [args.total_budget]
Ps2 = [(1.0/4.0)*args.total_budget, (3.0/4.0)*args.total_budget]
Ps3 = [(1.0/8.0)*args.total_budget, (1.0/8.0)*args.total_budget, (3.0/4.0)*args.total_budget]
Ps4 = [(1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (3.0/4.0)*args.total_budget]

n_l = np.linspace(10000, 80000, num=12)
err_nonpr = []
err_t1 = []
err_t2 = []
err_t3 = []
err_t4 = []

for i, n in enumerate(n_l):
    args.n = int(n)
    non_pr = []
    covs_t1 = []
    covs_t2 = []
    covs_t3 = []
    covs_t4 = []
    print(n)
    for i in range(100):
        if i % 50 == 0: print(i)
        X = torch.distributions.MultivariateNormal(dist_mean, dist_cov).sample((args.n,))
        non_pr.append(mahalanobis_dist(utils.cov(X.clone()), dist_cov))

        args.t = 1
        args.rho = Ps1
        covs_t1.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
        
        args.t = 2
        args.rho = Ps2
        covs_t2.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
        
        args.t = 3
        args.rho = Ps3
        covs_t3.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
          
        args.t = 4
        args.rho = Ps4
        covs_t4.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
                        
    err_nonpr.append(stats.trim_mean(non_pr,0.1))
    err_t1.append(stats.trim_mean(covs_t1,0.1))
    err_t2.append(stats.trim_mean(covs_t2,0.1))
    err_t3.append(stats.trim_mean(covs_t3,0.1))
    err_t4.append(stats.trim_mean(covs_t4,0.1))
    
np.savetxt("./results/synthetic_cov/n-4.txt", np.array(n_l))
np.savetxt("./results/synthetic_cov/nonpr-4.txt", np.array(err_nonpr))
np.savetxt("./results/synthetic_cov/t1-4.txt", np.array(err_t1))
np.savetxt("./results/synthetic_cov/t2-4.txt", np.array(err_t2))
np.savetxt("./results/synthetic_cov/t3-4.txt", np.array(err_t3))
np.savetxt("./results/synthetic_cov/t4-4.txt", np.array(err_t4))       
    
fig, ax = plt.subplots()
ax.plot(n_l, err_nonpr, marker="x", label='Non-private', color='#1f77b4')
ax.plot(n_l, err_t1, marker="x", label='t = 1', color='#9467bd')
ax.plot(n_l, err_t2, marker="x", label='t = 2', color='#d62728')
ax.plot(n_l, err_t3, marker="x", label='t = 3', color='#e377c2')
ax.plot(n_l, err_t4, marker="x", label='t = 4', color='#7f7f7f')

ax.set_xlabel('n')
ax.set_ylabel('Frobenius Error')
ax.set_title("Comparison")
ax.legend()

#Effect of privacy, spherical. Vary rho. Fix parameters d = 10, u = 10*sqrt(d), n = 8000. t = 1 through 4.
args = utils.parse_args()
args.d = 10
dist_mean = torch.zeros(args.d)
dist_cov = torch.eye(args.d)
args.u = 10*np.sqrt(args.d)
args.n = 8000


r_l = np.geomspace(0.005, 0.5, num=10)
err_nonpr = []
err_t1 = []
err_t2 = []
err_t3 = []
err_t4 = []

for i, r in enumerate(r_l):
    non_pr = []
    covs_t1 = []
    covs_t2 = []
    covs_t3 = []
    covs_t4 = []
    
    args.total_budget = r
    Ps1 = [args.total_budget]
    Ps2 = [(1.0/4.0)*args.total_budget, (3.0/4.0)*args.total_budget]
    Ps3 = [(1.0/8.0)*args.total_budget, (1.0/8.0)*args.total_budget, (3.0/4.0)*args.total_budget]
    Ps4 = [(1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (1.0/12.0)*args.total_budget, (3.0/4.0)*args.total_budget]
    print(r)
    
    for i in range(100):
        if i % 50 == 0: print(i)
        X = torch.distributions.MultivariateNormal(dist_mean, dist_cov).sample((args.n,))
        non_pr.append(mahalanobis_dist(utils.cov(X.clone()), dist_cov))

        args.t = 1
        args.rho = Ps1
        covs_t1.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
        
        args.t = 2
        args.rho = Ps2
        covs_t2.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
        
        args.t = 3
        args.rho = Ps3
        covs_t3.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
          
        args.t = 4
        args.rho = Ps4
        covs_t4.append(mahalanobis_dist(cov_est(X.clone(), args), dist_cov))
                
        
    err_nonpr.append(stats.trim_mean(non_pr,0.1))
    err_t1.append(stats.trim_mean(covs_t1,0.1))
    err_t2.append(stats.trim_mean(covs_t2,0.1))
    err_t3.append(stats.trim_mean(covs_t3,0.1))
    err_t4.append(stats.trim_mean(covs_t4,0.1))
    
fig, ax = plt.subplots()
ax.plot(r_l, err_nonpr, marker="x", label='Non-private', color='#1f77b4')
ax.plot(r_l, err_t1, marker="x", label='t = 1', color='#9467bd')
ax.plot(r_l, err_t2, marker="x", label='t = 2', color='#d62728')
ax.plot(r_l, err_t3, marker="x", label='t = 3', color='#e377c2')
ax.plot(r_l, err_t4, marker="x", label='t = 4', color='#7f7f7f')

np.savetxt("./results/synthetic_cov/r-5.txt", np.array(r_l))
np.savetxt("./results/synthetic_cov/nonpr-5.txt", np.array(err_nonpr))
np.savetxt("./results/synthetic_cov/t1-5.txt", np.array(err_t1))
np.savetxt("./results/synthetic_cov/t2-5.txt", np.array(err_t2))
np.savetxt("./results/synthetic_cov/t3-5.txt", np.array(err_t3))
np.savetxt("./results/synthetic_cov/t4-5.txt", np.array(err_t4))

ax.set_xlabel('rho')
ax.set_ylabel('Frobenius Error')
ax.set_title("Comparison")
ax.legend()

