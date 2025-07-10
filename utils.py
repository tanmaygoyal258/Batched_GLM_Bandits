
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
from Logistic_Regression_Torch import LogisticRegression
from tqdm import tqdm

def mat_norm(x , A):
    return np.sqrt(np.dot(x , np.dot(A , x)))

def sigmoid(x):
    try:
        return 1 / (1 + np.exp(-x))
    except RuntimeWarning as e:
        print(e)
        print(x)
        assert False

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def sigmoid_inv(x):
    assert x > 0 and x < 1
    return np.log(x / (1-x))

def probit(x):
    return norm.cdf(x)

def dprobit(x):
    return (1.0 / np.sqrt(2*np.pi)) * np.exp(-x*x/2.0)

def calc_cov_matrix(dim , arms, dist):
    """
    calculates the covariance matrix of the form sum{pi(i) x_i x_i^T}
    
    Parameters:
        dim: dimension of the arms
        arms: set of the arms (not necessarily support of dist)
        dist: distribution over the set of arms
    """
    assert len(arms) == len(dist)

    V = np.zeros((dim , dim))
    for idx , arm in enumerate(arms):
        V += dist[idx] * np.outer(arm , arm)

    tol = 1e-12
    if np.linalg.det(V) < tol:
        V += 0.00001 * np.eye(dim)

    return V

def G_Optimal_Design(arms , dim):
    """
    calculates a G-optimal design using the Frank-Wolfe algorithm
    Implementation of the algorithm given in Bandit Algorithms.
    """

    dist = np.array([1/len(arms) for _ in range(len(arms))])

    objective = -1
    max_norm = 10
    max_iter = 1e5
    threshold = 1e-4
    
    for iter in range(int(max_iter)):
        cov_matrix = calc_cov_matrix(dim , arms , dist)
        inv_cov = np.linalg.inv(cov_matrix)
        norms = [arm.T @ inv_cov @ arm for arm in arms]
        max_norm_idx = np.argmax(norms)
        max_norm = norms[max_norm_idx]

        step_size = (max_norm / dim - 1) / (max_norm - 1)

        if np.abs(objective - max_norm) < threshold:
            break

        dist = (1 - step_size) * dist
        dist[max_norm_idx] += step_size
        objective = max_norm

        dist = dist / np.sum(dist)
        # assert sum(dist) == 1 , "The distribution is not normalized and has sum {}".format(sum(dist))
        while dist.sum()!=1:
            extra = 1 - dist.sum()
            dist[np.random.randint(len(dist))] += extra
    return dist



def solve_mle_sklearn(theta , arms , rewards , lmbda):
        """
        Solves the MLE problem using SGD implemented in scikit-learn
        """
        # print(rewards)
        # print(f"Number of 1s: {np.sum(rewards)} , Number of 0s: {len(rewards) - np.sum(rewards)}")
        if len(np.unique(rewards)) == 1:
            if rewards[0] == 1:
                return theta , True
            else:
                return -theta , True
        sgd = SGDClassifier(loss = "log_loss" , fit_intercept = False , alpha = lmbda/2)
        sgd.fit(arms , rewards)
        return sgd.coef_[0] , True


def approximate_additive_oracle(arms , theta, eps , seed):
    """
    A randomized oracle that returns an arm within {eps} of the best arm.
    Could have other implementations, such as LSH.
    """
    best_val = np.max([np.dot(arm , theta) for arm in arms])
    np.random.seed(seed)
    while True:
        arm_idx = np.random.randint(len(arms))
        arm = arms[arm_idx]
        if np.dot(arm , theta) >= best_val - eps:
            # print(f"The best value is {best_val} and returning an arm with value {np.dot(arm , theta)}")
            return arm
        
def approximate_multiplicative_oracle(arms , theta , alpha , seed):
    """
    A randomized oracle which returns an arm which has value atleast {alpha} times the best.
    """
    best_val = np.max([np.dot(arm , theta) for arm in arms])
    np.random.seed(seed)
    while True:
        arm_idx = np.random.randint(len(arms))
        arm = arms[arm_idx]
        if np.dot(arm , theta) >= alpha * best_val:
            return arm_idx , arm
        
# From Sawarni et. al (https://github.com/nirjhar-das/GLBandit_Limited_Adaptivity.git)
def log_loss(theta , arms , rewards , lmbda):
    return -np.sum(rewards * np.log(sigmoid(np.dot(arms , theta))) + (1 - rewards) * np.log(1 - sigmoid(np.dot(arms , theta)))) + lmbda/2 * np.linalg.norm(theta)**2

def log_loss_glm(theta, X, Y, lmbda, model):
    if model == 'Logistic':
        return - np.sum(Y * np.log(sigmoid(np.dot(X, theta))) + (1 - Y) * np.log(1 - sigmoid(np.dot(X, theta)))) + lmbda * np.sum(np.square(theta))
    elif model == 'Probit':
        return - np.sum(Y * np.log(probit(np.dot(X, theta))) + (1 - Y) * np.log(1 - probit(np.dot(X, theta)))) + lmbda * np.sum(np.square(theta))

def solve_glm_mle(theta_prev, X, Y, lmbda, model):
    # res = minimize(log_loss_glm, theta_prev,\
    #                jac=grad_log_loss_glm, hess=hess_log_loss_glm, \
    #                 args=(X, Y, lmbda, model), method='Newton-CG')
    res = minimize(log_loss_glm, theta_prev, args=(X, Y, lmbda, model))
    # if not res.success:
    #     print(res.message)

    theta_hat, succ_flag = res.x, res.success
    return theta_hat, succ_flag