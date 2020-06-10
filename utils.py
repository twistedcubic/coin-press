# coding: utf-8
'''
Utilities functions
'''
import torch
import argparse
import os.path as osp
import numpy as np
import math

import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_budget', default=.5, type=float, help='Total privacy budget')
    parser.add_argument('--d', default=10, type=int, help='Feature dimension (dimension of synthetic data)')
    parser.add_argument('--n', default=3000, type=int, help='Number of samples to synthesize (for synthetic data)')
    parser.add_argument('--u', default=33, type=float, help='Initial upper bound for covariance')
    
    parser.add_argument('--fig_title', default=None, type=str, help='figure title')
    parser.add_argument('-f', default=None, type=str, help='needed for ipython starting')
    
    opt = parser.parse_args()
    return opt

def cov_nocenter(X):
    X = X
    cov = torch.mm(X.t(), X)/X.size(0)
    return cov

def cov(X):
    X = X - X.mean(0)
    cov = torch.mm(X.t(), X)/X.size(0)
    return cov

'''
PSD projection
'''
def psd_proj_symm(S):
    U, D, V_t = torch.svd(S)
    D = torch.clamp(D, min=0, max=None).diag()
    A = torch.mm(torch.mm(U, D), U.t()) 
    return A

'''
Mean Estimation Methods --------------------------------------------------------
'''

'''
Fine mean estimation algorithm 
 - list params are purely for graphing purposes and can be ignored if not needed
returns: fine DP estimate for mean
'''
def fineMeanEst(x, sigma, R, epsilon, epsilons=[], sensList=[], rounding_outliers=False):
    B = R+sigma*3
    sens = 2*B/(len(x)*epsilon) 
    epsilons.append([epsilon])
    sensList.append([sens])
    if rounding_outliers:
        for i in x:
            if i > B:
                i = B
            elif i < -1*B:
                i =  -1*B
    noise = np.random.laplace(loc=0.0, scale=sens)
    result = sum(x)/len(x) + noise 
    return result

'''
Coarse mean estimation algorithm with Private Histogram
returns: [start of intrvl, end of intrvl, freq or probability], bin number
- the coarse mean estimation would just be the midpoint of the intrvl (in case this is needed)
'''
def privateRangeEst(x, epsilon, delta, alpha, R, sd):
    # note alpha ∈ (0, 1/2)
    r = int(math.ceil(R/sd))
    bins = {}
    for i in range(-1*r,r+1):
        start = (i - 0.5)*sd # each bin is s ((j − 0.5)σ,(j + 0.5)σ]
        end = (i + 0.5)*sd 
        bins[i] = [start, end, 0] # first 2 elements specify intrvl, third element is freq
    # note: epsilon, delta ∈ (0, 1/n) based on https://arxiv.org/pdf/1711.03908.pdf Lemma 2.3
    # note n = len(x)
    # set delta here
    L = privateHistLearner(x, bins, epsilon, delta, r, sd)
    return bins[L], L


# helper function
# returns: max probability bin number
def privateHistLearner(x, bins, epsilon, delta, r, sd): # r, sd added to transmit info
    # fill bins
    max_prob = 0
    max_r = 0

    # creating probability bins
    for i in x:
        r_temp = int(round(i/sd))
        if r_temp in bins:
            bins[r_temp][2] += 1/len(x)
        
    for r_temp in bins:
        noise = np.random.laplace(loc=0.0, scale=2/(epsilon*len(x)))
        if delta == 0 or r_temp < 2/delta:
            # epsilon DP case
            bins[r_temp][2] += noise
        else:
            # epsilon-delta DP case
            if bins[r_temp][2] > 0:
                bins[r_temp][2] += noise
                t = 2*math.log(2/delta)/(epsilon*len(x)) + (1/len(x))
                if bins[r_temp][2] < t:
                    bins[r_temp][2] = 0
        
        if bins[r_temp][2] > max_prob:
            max_prob = bins[r_temp][2]
            max_r = r_temp
    return max_r


'''
Two shot algorithm
- may want to optimize distribution ratio between fine & coarse estimation

eps1 = epsilon for private histogram algorithm
eps2 = epsilon for fine mean estimation algorithm

returns: DP estimate for mean
'''
def twoShot(x, eps1, eps2, delta, R, sd):
    alpha = 0.5
    # coarse estimation
    [start, end, prob], r = privateRangeEst(x, eps1, delta, alpha, R, sd)
    for i in range(len(x)):
        if x[i] < start - 3*sd:
            x[i] = start - 3*sd
        elif x[i] > end + 3*sd:
            x[i] = end + 3*sd
    # fine estimation with smaller range (less sensitivity)
    est = fineMeanEst(x, sd, 3.5*sd, eps2)
    return est
