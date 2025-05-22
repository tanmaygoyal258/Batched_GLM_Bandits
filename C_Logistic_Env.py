import numpy as np
from tqdm import tqdm
from utils import sigmoid , dsigmoid
from Batch_GLinCB import Batch_GLinCB
from SoftBatch_Contextual import SoftBatch

class LogisticOracle:

    def __init__(self , theta_star , reward_rng):
        self.theta_star = theta_star
        self.reward_rng = reward_rng

    def expected_reward(self , arm):
        return min(sigmoid(np.dot(self.theta_star , arm)) , 1)
    
    def reward(self , arm):
        return self.reward_rng.binomial(1 , self.expected_reward(arm))

class Contextual_Logistic:

    def __init__(self , theta_star , params):

        self.param_norm_ub = params["param_norm_ub"]
        self.horizon = params["horizon"]
        self.failure_level = params["failure_level"]  
        self.dim = params["dimension"]
        self.number_arms = params["number_arms"]
        
        self.arm_seed = params["arm_seed"]
        self.kappa = self.find_kappa(theta_star)
        # self.kappa = 150
        print(f"The value of kappa is {self.kappa}")
        self.arm_rng = np.random.default_rng(self.arm_seed) # reset the arm generator

        self.reward_rng = np.random.default_rng(params["reward_seed"])
        self.oracle = LogisticOracle(theta_star , self.reward_rng)

        self.regret_arr = []
        self.previous_arm_sets = []
        
        self.alg = Batch_GLinCB(params , self.kappa) if params["alg_name"] == Batch_GLinCB else SoftBatch(params , self.kappa)
        self.batch_endpoints = self.alg.batch_endpoints


    def find_best_arm(self  ,arms):
        expected_rewards = [self.oracle.expected_reward(arm) for arm in arms]
        best_arm_idx = np.argmax(expected_rewards)
        best_arm = arms[best_arm_idx]
        best_arm_expected_reward = expected_rewards[best_arm_idx]
        # print(f"DEBUG: Best arm is {best_arm} index {best_arm_idx} and expected reward {expected_rewards[best_arm_idx]}")
        return best_arm_idx , best_arm , best_arm_expected_reward
    
    def find_kappa(self , theta):
        print(f"Calculating kappa")
        arm_rng = np.random.default_rng(self.arm_seed)
        kappa = -np.inf
        for _ in tqdm(range(self.horizon)):
            arms = self.create_arm_set(arm_rng)
            # if _ <10:
                # print(arms)
            mu_dot = [dsigmoid(np.dot(theta , arm)) for arm in arms]
            kappa = max(kappa , 1.0/np.min(mu_dot))
        return kappa

    
    def run_algorithm(self):
        # print(f"Starting Batch 1")
        for t in range(1 , self.horizon + 1):
            arms = self.create_arm_set(self.arm_rng)
            self.previous_arm_sets.append(arms)
            self.best_arm_idx , self.best_arm , self.best_arm_expected_reward = self.find_best_arm(arms)
            
            recommendation , type = self.alg.play(t , arms)
            if type == "arm":
                arm_played = recommendation
            else:
                # returns a theta
                inner_products = [np.dot(arm , recommendation) for arm in arms]
                best_arm_idx = np.argmax(inner_products)
                arm_played = arms[best_arm_idx]
            
            reward = self.oracle.reward(arm_played)
            self.regret_arr.append(self.calculate_regret(arm_played , reward))
            if t != self.horizon:
                if_erase = self.alg.update_params(t , reward , self.previous_arm_sets)
                if if_erase:
                    self.previous_arm_sets = []
        assert len(self.regret_arr) == self.horizon

    def calculate_regret(self , arm_played , reward):
        return self.best_arm_expected_reward - self.oracle.expected_reward(arm_played)
    
    def create_arm_set(self , arm_rng):
        arms = []
        for a in range(self.number_arms):
            arm = [arm_rng.random()*2 - 1 for i in range(self.dim)]
            arm = arm / np.linalg.norm(arm)
            arms.append(arm)
        return arms