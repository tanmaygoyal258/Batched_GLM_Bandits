import numpy as np
from tqdm import tqdm
from utils import sigmoid , dsigmoid
from BatchGLinCB import BatchGLinCB
from SoftBatch_Contextual import SoftBatch

class LogisticOracle:
    """
    A class to implement a Logistic reward oracle.
    Initialization:
        theta_star: the optimal reward vector (unknown to the algorithm)
        reward_rng: a random generator for the rewards
    """

    def __init__(self , theta_star , reward_rng):
        self.theta_star = theta_star
        self.reward_rng = reward_rng

    def expected_reward(self , arm):
        return min(sigmoid(np.dot(self.theta_star , arm)) , 1)
    
    def reward(self , arm):
        return self.reward_rng.binomial(1 , self.expected_reward(arm))

class Contextual_Logistic:
    """
    A class to run algorithms for Contextual settings with a Logistic reward.
    """ 
    def __init__(self , theta_star , params):

        self.param_norm_ub = params["param_norm_ub"]
        self.horizon = params["horizon"]
        self.failure_level = params["failure_level"]  
        self.dim = params["dimension"]
        self.number_arms = params["number_arms"]
        
        self.arm_seed = params["arm_seed"]
        self.kappa = self.find_kappa(theta_star)
        
        # reset the arm generator, we will generate arms at each time round
        self.arm_rng = np.random.default_rng(self.arm_seed)

        self.reward_rng = np.random.default_rng(params["reward_seed"])
        self.oracle = LogisticOracle(theta_star , self.reward_rng)

        self.regret_arr = []
        self.previous_arm_sets = []
        
        self.alg = BatchGLinCB(params , self.kappa) if params["alg_name"] == "BatchGLinCB" else SoftBatch(params , self.kappa)

        self.batch_endpoints = self.alg.batch_endpoints

    def find_best_arm(self  ,arms):
        """
        returns the best arm index, the best arm, and the expected reward for this arm given an arm set
        """
        expected_rewards = [self.oracle.expected_reward(arm) for arm in arms]
        best_arm_idx = np.argmax(expected_rewards)
        best_arm = arms[best_arm_idx]
        best_arm_expected_reward = expected_rewards[best_arm_idx]
        # print(f"DEBUG: Best arm is {best_arm} index {best_arm_idx} and expected reward {expected_rewards[best_arm_idx]}")
        return best_arm_idx , best_arm , best_arm_expected_reward
    
    def find_kappa(self , theta):
        """
        finds kappa (with repsect to theta_star)
        """
        # sets the random generator for the arms
        arm_rng = np.random.default_rng(self.arm_seed)
        kappa = -np.inf
        # for _ in tqdm(range(self.horizon)):
        for _ in (range(self.horizon)):
            arms = self.create_arm_set(arm_rng)
            mu_dot = [dsigmoid(np.dot(theta , arm)) for arm in arms]
            kappa = max(kappa , 1.0/np.min(mu_dot))
        return kappa

    
    def run_algorithm(self):
        """
        runs the algorithm in the standard format: choose an arm, obtain reward, update params
        """
        for t in range(1 , self.horizon + 1):
            # create the arm set for that particular round
            arms = self.create_arm_set(self.arm_rng)
            # store the arm set
            self.previous_arm_sets.append(arms)     
            # find the true best arm from that arm_set
            self.best_arm_idx , self.best_arm , self.best_arm_expected_reward = self.find_best_arm(arms)
            
            # obtain the recommended play and the type of it (arm/theta)
            recommendation , type = self.alg.play(t , arms)
            if type == "arm":   # directly play the recommendation
                arm_played = recommendation
            else:       # obtained a theta, play the best arm with respect to it
                inner_products = [np.dot(arm , recommendation) for arm in arms]
                best_arm_idx = np.argmax(inner_products)
                arm_played = arms[best_arm_idx]
            
            # obtain the reward
            reward = self.oracle.reward(arm_played)
            
            self.regret_arr.append(self.calculate_regret(arm_played , reward))
            if t != self.horizon:
                if_erase = self.alg.update_params(t , reward , self.previous_arm_sets)
                if if_erase:     # parameters have been updated, erase all previous arm sets
                    self.previous_arm_sets = []
        assert len(self.regret_arr) == self.horizon

    def calculate_regret(self , arm_played , reward):
        """
        calculates the expected reward
        """
        return self.best_arm_expected_reward - self.oracle.expected_reward(arm_played)
    
    def create_arm_set(self , arm_rng):
        """
        creates an arm set using a random generator
        """
        arms = []
        for a in range(self.number_arms):
            arm = [arm_rng.random()*2 - 1 for i in range(self.dim)]
            arm = arm / np.linalg.norm(arm)
            arms.append(arm)
        return arms