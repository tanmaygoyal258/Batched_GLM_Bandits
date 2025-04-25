import numpy as np
from utils import sigmoid , dsigmoid
from NC_Logistic import NC_Logistic_Alg

class LogisticOracle:

    def __init__(self , theta_star , epsilon , reward_rng):
        self.theta_star = theta_star
        self.misspecification = epsilon
        self.reward_rng = reward_rng

    def expected_reward(self , arm):
        return min(sigmoid(np.dot(self.theta_star , arm)) + self.misspecification , 1)
    
    def reward(self , arm):
        return self.reward_rng.binomial(1 , self.expected_reward(arm))

class Non_Contextual_Logistic:

    def __init__(self , theta_star , params , epsilon):

        self.param_norm_ub = params["param_norm_ub"]
        self.horizon = params["horizon"]
        self.epsilon = epsilon
        self.failure_level = params["failure_level"]  
        self.dim = params["dimension"]
        self.number_arms = params["number_arms"]
        
        self.arm_rng = np.random.default_rng(params["arm_seed"])
        self.arms = self.create_arm_set(self.arm_rng)
        
        self.reward_rng = np.random.default_rng(params["reward_seed"])
        self.oracle = LogisticOracle(theta_star , self.epsilon , self.reward_rng)

        self.lmbda = 1 
        self.kappa = self.find_kappa(theta_star)
        print(f"The value of kappa is {self.kappa}")
        
        self.regret_arr = []
        self.best_arm_idx , self.best_arm , self.best_arm_expected_reward = self.find_best_arm()
        
        self.alg = NC_Logistic_Alg(params , self.arms , epsilon , self.kappa)
        self.batch_endpoints = self.alg.batch_endpoints
        
    def find_best_arm(self):
        expected_rewards = [self.oracle.expected_reward(arm) for arm in self.arms]
        best_arm_idx = np.argmax(expected_rewards)
        best_arm = self.arms[best_arm_idx]
        best_arm_expected_reward = expected_rewards[best_arm_idx]
        print(f"DEBUG: Best arm is {best_arm} index {best_arm_idx} and expected reward {expected_rewards[best_arm_idx]}")
        return best_arm_idx , best_arm , best_arm_expected_reward
    
    def find_kappa(self , theta):
        mu_dot = [dsigmoid(np.dot(theta , arm)) for arm in self.arms]
        return 1.0/np.min(mu_dot)
    
    def run_algorithm(self):
        print(f"Starting Batch 1")
        for t in range(1 , self.horizon + 1):
            arm_played = self.alg.play(t)
            reward = self.oracle.reward(arm_played)
            self.regret_arr.append(self.calculate_regret(arm_played , reward))
            if t != self.horizon:
                self.alg.update_params(t , reward)

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