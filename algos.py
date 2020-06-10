
'''
Privately estimating covariance.
'''
import torch
import utils
import numpy as np
import math


def cov_est_step(X, A, rho, cur_iter, args):
    """
    One step of multivariate covariance estimation, scale cov a.
    """
    W = torch.mm(X, A)
    n = args.n
    d = args.d

    #Hyperparameters
    gamma = gaussian_tailbound(d, 0.1)
    eta = 0.5*(2*(np.sqrt(d/n)) + (np.sqrt(d/n))**2)
    
    #truncate points
    W_norm = np.sqrt((W**2).sum(-1, keepdim=True))
    norm_ratio = gamma / W_norm
    large_norm_mask = (norm_ratio < 1).squeeze()
    
    W[large_norm_mask] = W[large_norm_mask] * norm_ratio[large_norm_mask]
    #noise
    Y = torch.randn(d, d)
    noise_var = (gamma**4/(rho*n**2))
    Y *= np.sqrt(noise_var)    
    #can also do Y = torch.triu(Y, diagonal=1) + torch.triu(Y).t()
    Y = torch.triu(Y)
    Y = Y + Y.t() - Y.diagonal().diag_embed() #Don't duplicate diagonal entries
    Z = (torch.mm(W.t(), W))/n
    #add noise    
    Z = Z + Y
    #ensure psd of Z
    Z = utils.psd_proj_symm(Z)
    
    U = Z + eta*torch.eye(d)
    inv = torch.inverse(U)
    invU, invD, invV = inv.svd()
    inv_sqrt = torch.mm(invU, torch.mm(invD.sqrt().diag_embed(), invV.t()))
    A = torch.mm(inv_sqrt, A)
    return A, Z

def cov_est(X, args ):
    """
    Multivariate covariance estimation.
    Returns: zCDP estimate of cov.
    """
    A = torch.eye(args.d) / np.sqrt(args.u)
    assert len(args.rho) == args.t
    
    for i in range(args.t-1):
        A, Z = cov_est_step(X, A, args.rho[i], i, args)
    A_t, Z_t = cov_est_step(X, A, args.rho[-1], args.t-1, args)
    
    cov = torch.mm(torch.mm(A.inverse(), Z_t), A.inverse())
    return cov

def gaussian_tailbound(d,b):
    return ( d + 2*( d * math.log(1/b) )**0.5 + 2*math.log(1/b) )**0.5

def mahalanobis_dist(M, Sigma):
    Sigma_inv = torch.inverse(Sigma)
    U_inv, D_inv, V_inv = Sigma_inv.svd()
    Sigma_inv_sqrt = torch.mm(U_inv, torch.mm(D_inv.sqrt().diag_embed(), V_inv.t()))
    M_normalized = torch.mm(Sigma_inv_sqrt, torch.mm(M, Sigma_inv_sqrt))
    return torch.norm(M_normalized - torch.eye(M.size()[0]), 'fro')

''' 
Functions for mean estimation
'''

##    X = dataset
##    c,r = prior knowledge that mean is in B2(c,r)
##    t = number of iterations
##    Ps = 
def multivariate_mean_iterative(X, c, r, t, Ps):
    for i in range(t-1):
        c, r = multivariate_mean_step(X, c, r, Ps[i])
    c, r = multivariate_mean_step(X, c, r, Ps[t-1])
    return c

def multivariate_mean_step(X, c, r, p):
    n, d = X.shape

    ## Determine a good clipping threshold
    gamma = gaussian_tailbound(d,0.01)
    clip_thresh = min((r**2 + 2*r*3 + gamma**2)**0.5,r + gamma) #3 in place of sqrt(log(2/beta))
        
    ## Round each of X1,...,Xn to the nearest point in the ball B2(c,clip_thresh)
    x = X - c
    mag_x = np.linalg.norm(x, axis=1)
    outside_ball = (mag_x > clip_thresh)
    x_hat = (x.T / mag_x).T
    X[outside_ball] = c + (x_hat[outside_ball] * clip_thresh)
    
    ## Compute sensitivity
    delta = 2*clip_thresh/float(n)
    sd = delta/(2*p)**0.5
    
    ## Add noise calibrated to sensitivity
    Y = np.random.normal(0, sd, size=d)
    c = np.sum(X, axis=0)/float(n) + Y
    r = ( 1/float(n) + sd**2 )**0.5 * gaussian_tailbound(d,0.01)
    return c, r

def L1(est): # assuming 0 vector is gt
    return np.sum(np.abs(est))
    
def L2(est): # assuming 0 vector is gt
    return np.linalg.norm(est)

