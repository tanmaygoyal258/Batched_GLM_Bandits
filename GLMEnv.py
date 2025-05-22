import numpy as np
from time import time
from tqdm import tqdm
from utils import dsigmoid , sigmoid
from ada_OFU_ECOLog import ada_OFU_ECOLog
from RS_GLinCB import RS_GLinCB


class LogisticOracle:

    def __init__(self , theta_star , reward_rng , misspecification_dict):
        self.theta_star = theta_star
        self.reward_rng = reward_rng
        # self.misspecification_dict = misspecification_dict

    def expected_reward(self , arm):
        # misspecification = self.misspecification_dict[tuple(arm)]
        return min(sigmoid(np.dot(self.theta_star , arm)) , 1)
    
    def reward(self , arm):
        return self.reward_rng.binomial(1 , self.expected_reward(arm))
            
    
class GLMEnv:

    def __init__(self , params , theta_star , epsilon):
        
        self.alg_name = params["alg_name"]

        self.horizon = params["horizon"]
        self.dim = params["dimension"]
        self.contextual = params["contextual"]
        self.number_arms = params["number_arms"]
        self.arm_seed = params["arm_seed"]
        

        self.arm_rng = np.random.default_rng(params["arm_seed"])
        self.arms = self.create_arm_set(self.arm_rng)

        self.epsilon = epsilon
        self.reward_seed = params["reward_seed"]
        print(f"Reward Seed is {self.reward_seed}")
        self.reward_rng = np.random.default_rng(params["reward_seed"])
        self.epsilon_rng = np.random.default_rng(params["epsilon_seed"])
        self.misspecification_dict = {tuple(arm) : self.epsilon_rng.uniform() * 2 * self.epsilon - self.epsilon for arm in self.arms}
        self.oracle = LogisticOracle(theta_star , self.reward_rng , self.misspecification_dict)

        self.kappa = self.find_kappa(params["theta_star"])
        print(f"The value of kappa is {self.kappa}")

        if self.alg_name == "ada_ofu_ecolog":
            self.alg = ada_OFU_ECOLog(params , self.arms)
        elif self.alg_name == "RS_GLinCB":
            self.alg = RS_GLinCB(params , self.arms , self.kappa)        
        self.regret_arr = []
        self.time_arr = []

    def find_kappa(self , theta):
        if self.contextual:
            print(f"Calculating kappa")
            arm_rng = np.random.default_rng(self.arm_seed)
            kappa = -np.inf
            for _ in tqdm(range(self.horizon)):
                arms = self.create_arm_set(arm_rng)
                mu_dot = [dsigmoid(np.dot(theta , arm)) for arm in arms]
                kappa = max(kappa , 1.0/np.min(mu_dot))
            return kappa
        else:
            arm_rng = np.random.default_rng(self.arm_seed)
            arms = self.create_arm_set(arm_rng)
            mu_dot = [dsigmoid(np.dot(theta , arm)) for arm in arms]
            return 1.0 / np.min(mu_dot)
    
    def play_algorithm(self):
        for _ in tqdm(range(self.horizon)):
            
            # obtain the arms
            if self.contextual:
                self.arms = self.create_arm_set(self.arm_rng)
            arm_set = self.arms

            pull_start = time()
            # pull the arm
            picked_arm = self.alg.pull(arm_set)
            pull_end = time()

            # obtain the actual reward and expected regret
            best_arm_idx , best_arm , best_arm_reward = self.find_best_arm()
            actual_reward = self.oracle.reward(picked_arm)
            expected_regret = best_arm_reward - self.oracle.expected_reward(picked_arm)

            update_start = time()
            # update the parameters
            self.alg.update_parameters(picked_arm , actual_reward) 
            update_end = time()

            # store the regrets, rewards, and time
            self.regret_arr.append(expected_regret)
            self.time_arr.append(update_end + pull_end - update_start - pull_start)

        return self.regret_arr , self.time_arr

    def find_best_arm(self):
        expected_rewards = [self.oracle.expected_reward(arm) for arm in self.arms]
        best_arm_idx = np.argmax(expected_rewards)
        best_arm = self.arms[best_arm_idx]
        best_arm_expected_reward = expected_rewards[best_arm_idx]
        # print(f"DEBUG: Best arm is {best_arm} index {best_arm_idx} and expected reward {expected_rewards[best_arm_idx]}")
        return best_arm_idx , best_arm , best_arm_expected_reward
    
    def create_arm_set(self , arm_rng):
        arms = []
        for a in range(self.number_arms):
            arm = [arm_rng.random()*2 - 1 for i in range(self.dim)]
            arm = arm / np.linalg.norm(arm)
            arms.append(arm)
        return arms