import numpy as np
from tqdm import tqdm
from time import time
from utils import *

def LW_ArgMax_Reduction(horizon , reduced_armset , scaling_factor , estimate_theta , eta , best_arm , theta):
    '''
    returns the action which has value within alpha-multiplicative error
    of the best action
    '''

    # initialize the parameters
    T = horizon
    W = 3 * np.log(T)
    N = W
    s = 1 - 1/(6*np.log(T))
    z = 2**W
    q = np.finfo(np.float32).eps
    d = len(estimate_theta)

    arms = []
    thetas = []
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
        thetas.append(tuple(q * np.floor(new_theta_estimate * np.sqrt(d) / q) / np.sqrt(d)))
        z *= s

    # find the best scaled arm for each theta obtained    
    thetas = list(set(thetas))
    thetas = [np.array(t) for t in thetas]
    arms = [reduced_armset.calculate_fixed_arm(theta) for theta in (thetas)]
    values = [(np.dot(a , estimate_theta) * scaling_factor) / (1 + eta * scaling_factor * (np.dot(best_arm , estimate_theta) - np.dot(a , estimate_theta))) for a in arms]
    max_idx = np.argsort(values)[-1]
    return arms[max_idx] , thetas , arms
    
    # DEBUG: Direct application of Multiplicative Oracle
    # phi_arms = [(scaling_factor * a)/(1 + eta * scaling_factor * (np.dot(best_arm , estimate_theta) - np.dot(a , estimate_theta))) for a in arm_set]
    # phi_arms = [(scaling_factor * np.array(a)) / (1 + eta * scaling_factor * (np.dot(best_arm , estimate_theta) - np.dot(a , estimate_theta))) for a in arm_set]
    # best_phi_arm_idx , best_phi_arm = approximate_multiplicative_oracle(phi_arms , estimate_theta , alpha = np.exp(-3) , seed = seed)
    # for i in range(len(phi_arms)):
    #     if phi_arms[i].all() == best_phi_arm.all():
    #         return arm_set[i]