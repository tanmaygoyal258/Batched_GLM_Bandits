import numpy as np
from tqdm import tqdm
from time import time
from utils import *

def LW_ArgMax(params , arm_set , scaling_factor , estimate_theta , eta , best_arm , theta , seed):
    '''
    returns the action which has value within alpha-multiplicative error
    of the best action
    '''
    # setting the parameters
    T = params["horizon"]
    W = 3 * np.log(T)
    N = W * np.log(T)**2
    s = 1 - 1/(6*np.log(T))
    eps_den = (1/T) ** (7 + 12*np.log(T))
    eps = ((1 - np.exp(-1)) / 12) * eps_den
    z = 2**W

    arms = []
    # for r in tqdm(range(int(np.ceil(N))+1)):
    for r in (range(int(np.ceil(N))+1)):
        new_theta_estimate = (1 + 1/W) * z * estimate_theta + (z ** (1 + 1/W)) * eta * theta
        theta_copy = new_theta_estimate.tolist()
        new_theta_estimate = []
        
        # ensuring all the components of theta donot become very small
        for e in theta_copy:
            if e > 0 and e < np.finfo(np.float32).eps:
                new_theta_estimate.append(np.finfo(np.float32).eps)
            elif e < 0 and e > -np.finfo(np.float32).eps:
                new_theta_estimate.append(-np.finfo(np.float32).eps)
            else:
                new_theta_estimate.append(e)

        new_theta_estimate = np.array(new_theta_estimate)
        a_i = approximate_additive_oracle(arm_set , new_theta_estimate / np.linalg.norm(new_theta_estimate) , eps , r)
        arms.append(a_i)
        z *= s
        
    values = [(np.dot(a , estimate_theta) * scaling_factor) / (1 + eta * scaling_factor * (np.dot(best_arm , estimate_theta) - np.dot(a , estimate_theta))) for a in arms]
    max_idx = np.argsort(values)[-1]
    return arms[max_idx]
    
    # DEBUG: Direct application of Multiplicative Oracle
    # phi_arms = [(scaling_factor * np.array(a)) / (1 + eta * scaling_factor * (np.dot(best_arm , estimate_theta) - np.dot(a , estimate_theta))) for a in arm_set]
    # best_phi_arm_idx , best_phi_arm = approximate_multiplicative_oracle(phi_arms , estimate_theta , alpha = np.exp(-3) , seed = seed)
    # return arm_set[best_phi_arm_idx]