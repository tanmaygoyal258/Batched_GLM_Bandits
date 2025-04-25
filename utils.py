
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import norm
from scipy.optimize import minimize

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))


def weighted_norm(x, A):
    return np.sqrt(np.dot(x, np.dot(A, x)))


def gaussian_sample_ellipsoid(center, design, radius):
    dim = len(center)
    sample = np.random.normal(0, 1, (dim,))
    res = np.real_if_close(center + np.linalg.solve(sqrtm(design), sample) * radius)
    return res

def probit(x):
    return norm.cdf(x)

def dprobit(x):
    return (1.0 / np.sqrt(2*np.pi)) * np.exp(-x*x/2.0)

def calc_cov_matrix(dim , arms, dist):

    assert len(arms) == len(dist)

    V = np.zeros((dim , dim))
    for idx , arm in enumerate(arms):
        V += dist[idx] * np.outer(arm , arm)

    tol = 1e-12
    if np.linalg.det(V) < tol:
        V += 0.00001 * np.eye(dim)

    return V



def G_Optimal_Design(arms , dim):

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
    return dist / np.sum(dist)

def mat_norm(x , A):
    return np.sqrt(np.dot(x , np.dot(A , x)))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def sigmoid_inv(x):
    assert x > 0 and x < 1
    return np.log(x / (1-x))

def log_loss(theta , arms , rewards , lmbda):
    return -np.sum(rewards * np.log(sigmoid(np.dot(arms , theta))) + (1 - rewards) * np.log(1 - sigmoid(np.dot(arms , theta)))) + lmbda/2 * np.linalg.norm(theta)**2


def solve_mle(theta , arms , rewards , lmbda):
        res = minimize(log_loss , theta , args = (arms , rewards , lmbda))
        new_theta , res_flag = res.x , res.success
        return new_theta, res_flag