
import utils
import algos as dp
import numpy as np
import torch

'''
Demos for private covariance estimation and private mean estimation.
'''

##############################################
# private multivariate covariance estimation #
##############################################

def demo_cov(args):
    dist_mean = torch.zeros(args.d)
    dist_cov = torch.eye(args.d)
    #sample data
    X = torch.distributions.MultivariateNormal(dist_mean, dist_cov).sample((args.n,))

    args.u = 10*np.sqrt(args.d)
    #specify how to spend the budget
    budget_l = [(1./4)*args.total_budget, (3./4)*args.total_budget]
    
    args.t = len(budget_l)
    args.rho = budget_l
    
    # estimate cov
    cov_est = dp.cov_est(X.clone(), args)
    est_err = dp.mahalanobis_dist(cov_est, dist_cov)
    print('The distance between the private covariance estimation and the true covariance is {}'.format(est_err))

###########################
# private mean estimation #
###########################
def demo_mean(args):
    d = args.d
    n = args.n
    mean = [0.0]*d
    cov = np.eye(d)
    c = [0]*d
    r = 10*np.sqrt(d)
    args.total_budget = 0.5
    eps = np.sqrt(2.0*args.total_budget)
    
    args.rho = [(1.0/4.0)*args.total_budget, (3.0/4.0)*args.total_budget]
    args.t = len(args.rho)
    X = np.random.multivariate_normal(mean, cov, int(n))
    
    non_private_mean = dp.L2(np.mean(X, axis=0)-mean)
    
    means_naive_coord = []
    means_kv_coord = []
    
    for j in range(d):
        
        means_naive_coord.append(utils.fineMeanEst(X[:,j].copy(), 1, r, eps/np.sqrt(d)) - mean[j])
        means_kv_coord.append(utils.twoShot(X[:,j].copy(), np.sqrt(1.0/2.0)*eps/np.sqrt(d), np.sqrt(1.0/2.0)*eps/np.sqrt(d), 0, r, 1) - mean[j])

    mean_naive = dp.L2(np.asarray(means_naive_coord))
    mean_kv = dp.L2(np.asarray(means_kv_coord))    
    iterative = dp.L2(dp.multivariate_mean_iterative(X.copy(), c, r, args.t, args.rho)-mean)
    print('non_private_mean {}, mean_naive {}, mean_kv {}, iterative estimate {}'.format(non_private_mean, mean_naive, mean_kv, iterative))
    

if __name__ == '__main__':
    args = utils.parse_args()
    #these can also be specified by command line arguments, but are shown here for illustration
    '''
    args.d = 10
    args.total_budget = .5
    '''
    print('~~ Begin mean estimation demo ~~') 
    demo_mean(args)

    print('~~ Begin covariance estimation demo ~~')
    demo_cov(args)
