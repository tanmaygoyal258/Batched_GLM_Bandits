import numpy as np
from utils import sigmoid , dsigmoid
from Batch_GLinCB_Fixed_1 import Batch_GLinCB_Fixed_1
from SoftBatch import SoftBatch
from SoftBatch_Updated import SoftBatch_Updated

class LogisticOracle:

    def __init__(self , theta_star , reward_rng , misspecification_dict):
        self.theta_star = theta_star
        self.reward_rng = reward_rng
        self.misspecification_dict = misspecification_dict

    def expected_reward(self , arm):
        misspecification = self.misspecification_dict[tuple(arm)]
        return min(sigmoid(np.dot(self.theta_star , arm)) + misspecification , 1)
    
    def reward(self , arm):
        return self.reward_rng.binomial(1 , self.expected_reward(arm))

class Non_Contextual_Logistic:

    def __init__(self , theta_star , params , epsilon):

        self.param_norm_ub = params["param_norm_ub"]
        self.horizon = params["horizon"]
        self.failure_level = params["failure_level"]  
        self.dim = params["dimension"]
        self.number_arms = params["number_arms"]
        self.theta_star = theta_star
        
        self.arm_rng = np.random.default_rng(params["arm_seed"])
        self.arms = self.create_arm_set(self.arm_rng)
        self.arms = self.correct_arms(0.01)
        print("After correction the number of arms is " , len(self.arms))
        
        self.epsilon = epsilon
        self.reward_rng = np.random.default_rng(params["reward_seed"])
        self.epsilon_rng = np.random.default_rng(params["epsilon_seed"])
        self.misspecification_dict = {tuple(arm) : self.epsilon_rng.uniform() * 2 * self.epsilon - self.epsilon for arm in self.arms}
        self.oracle = LogisticOracle(theta_star , self.reward_rng , self.misspecification_dict)

        self.best_arm_idx , self.best_arm , self.best_arm_expected_reward = self.find_best_arm()
        self.kappa = self.find_kappa(theta_star)
        print(f"The value of kappa is {self.kappa}")
        
        self.regret_arr = []
        self.theta_norms = []
        self.best_arm_gaps = []
        
        # self.best_rewards = []
        # for t in range(self.horizon):
        #     self.best_rewards.append(self.oracle.reward(self.best_arm))
        # # reset the generators and oracle
        # self.reward_rng = np.random.default_rng(params["reward_seed"])
        # self.epsilon_rng = np.random.default_rng(params["epsilon_seed"])
        # self.oracle = LogisticOracle(theta_star , self.epsilon , self.reward_rng , self.epsilon_rng)

        # self.alg = NC_Logistic_Alg(params , self.arms , epsilon , self.kappa) if params["alg_name"] == "NC_Logistic" else Soft_Elimination(params , self.arms , self.kappa , epsilon)
        # self.outfile = open(params["path"] + "/outfile.txt" , "a")
        if params["alg_name"] == "Fixed-1":
            self.alg = Batch_GLinCB_Fixed_1(params , self.arms , self.kappa , epsilon) 
        elif params["alg_name"] == "SoftBatch":
            self.alg = SoftBatch(params , self.arms , self.kappa , epsilon)
        elif params["alg_name"] == "SoftBatch_Updated":
            self.alg = SoftBatch_Updated(params , self.arms , self.kappa , epsilon)

        self.batch_endpoints = self.alg.batch_endpoints

        
    def find_best_arm(self):
        expected_rewards = [self.oracle.expected_reward(arm) for arm in self.arms]
        best_arm_idx = np.argmax(expected_rewards)
        best_arm = self.arms[best_arm_idx]
        best_arm_expected_reward = expected_rewards[best_arm_idx]
        print(f"DEBUG: Best arm is {best_arm} index {best_arm_idx} and expected reward {expected_rewards[best_arm_idx]}")
        expected_rewards.sort()
        print(f"DEBUG: Second best arm has expected_reward {expected_rewards[-2]}")
        return best_arm_idx , best_arm , best_arm_expected_reward
    
    def find_kappa(self , theta):
        mu_dot = [dsigmoid(np.dot(theta , arm)) for arm in self.arms]
        linear_gaps = [np.dot(self.best_arm - arm , self.theta_star) for arm in self.arms]
        non_linear_gaps = [sigmoid(np.dot(self.best_arm , self.theta_star)) - sigmoid(np.dot(arm , self.theta_star)) for arm in self.arms]
        linear_gaps.sort()
        non_linear_gaps.sort()
        print("Minimum linear gap is " , linear_gaps[1])
        print("Minimum non-linear gap is " , non_linear_gaps[1])
        return 1.0/np.min(mu_dot)
    
    def run_algorithm(self):
        print(f"Starting Batch 1")
        for t in range(1 , self.horizon + 1):
            arm_played , new = self.alg.play(t)
            try:
                reward = self.oracle.reward(arm_played)
            except:
                for a in self.arms:
                    if np.allclose(a , arm_played):
                        arm_played = a
                        break
                reward = self.oracle.reward(arm_played)

            if new:
                print(f"Associated gap is {self.oracle.expected_reward(arm_played)} and associated regret is {self.best_arm_expected_reward - self.oracle.expected_reward(arm_played)}")
                print(f"Associated linear gap is {np.dot(self.best_arm  - arm_played , self.theta_star)}")
            self.regret_arr.append(self.best_arm_expected_reward - self.oracle.expected_reward(arm_played))
            if t != self.horizon:
                theta , best_arm = self.alg.update_params(t , reward , arm_played)
                if theta is not None:
                    print(f"The new theta is {np.linalg.norm(self.theta_star - theta)} away")
                    self.theta_norms.append(np.linalg.norm(self.theta_star - theta))
                    self.best_arm_gaps.append(self.best_arm_expected_reward - self.oracle.expected_reward(best_arm))

        # assert len(self.regret_arr) == self.horizon

    def create_arm_set(self , arm_rng):
        arms = []
        for a in range(self.number_arms):
            arm = [arm_rng.random()*2 - 1 for i in range(self.dim)]
            arm = arm / np.linalg.norm(arm)
            arms.append(arm)
        return arms
    
    def correct_arms(self , threshold):
        new_arms = []
        # find best arm idx
        max_value = -1
        best_idx = -1
        for idx , arm in enumerate(self.arms):
            if sigmoid(np.dot(arm , self.theta_star)) > max_value:
                max_value = sigmoid(np.dot(arm , self.theta_star))
                best_idx = idx
        # eliminate arms that are too close to best arm
        for idx , arm in enumerate(self.arms):
            if idx == best_idx :
                new_arms.append(arm)
                continue
            if sigmoid(np.dot(self.arms[best_idx] , self.theta_star)) - sigmoid(np.dot(arm , self.theta_star)) >= threshold:
                new_arms.append(arm)

        return new_arms