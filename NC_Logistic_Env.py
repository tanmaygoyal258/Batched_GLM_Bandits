import numpy as np
from utils import sigmoid , dsigmoid
from BatchGLinCB_Fixed import BatchGLinCB_Fixed
from SoftBatch import SoftBatch

class LogisticOracle:
    """
    A class to implement a Logistic reward oracle.
    Initialization:
        theta_star: the optimal reward vector (unknown to the algorithm)
        reward_rng: a random generator for the rewards
        misspecification_dict: a dictionary containing the permissible misspecification values for each arm
    """

    def __init__(self , theta_star , reward_rng , misspecification_dict):
        self.theta_star = theta_star
        self.reward_rng = reward_rng
        self.misspecification_dict = misspecification_dict

    def expected_reward(self , arm):
        misspecification = self.misspecification_dict[tuple(arm)]
        # ensure expected reward is between 0 and 1
        return max(0 , min(sigmoid(np.dot(self.theta_star , arm)) + misspecification , 1))
    
    def reward(self , arm):
        reward = self.reward_rng.binomial(1 , self.expected_reward(arm))
        return reward

class Non_Contextual_Logistic:
    """
    A class to run algorithms for Non-Contextual settings with a Logistic reward.
    """

    def __init__(self , theta_star , params , epsilon):

        self.param_norm_ub = params["param_norm_ub"]
        self.horizon = params["horizon"]
        self.failure_level = params["failure_level"]  
        self.dim = params["dimension"]
        self.number_arms = params["number_arms"]
        self.theta_star = theta_star
        
        self.arm_rng = np.random.default_rng(params["arm_seed"])
        self.arms = self.create_arm_set(self.arm_rng)
        # self.arms = self.correct_arms(0.01)
        
        self.epsilon = epsilon
        self.reward_rng = np.random.default_rng(params["reward_seed"])
        self.epsilon_rng = np.random.default_rng(params["epsilon_seed"])
        
        # the permissible misspecification for any arm is between -eps and eps, where eps is randomly generated
        self.misspecification_dict = {tuple(arm) : self.epsilon_rng.uniform() * 2 * self.epsilon - self.epsilon for arm in self.arms}
        self.oracle = LogisticOracle(theta_star , self.reward_rng , self.misspecification_dict)

        self.best_arm_idx , self.best_arm , self.best_arm_expected_reward = self.find_best_arm()
        self.kappa = self.find_kappa(theta_star)
        # print(f"The value of kappa is {self.kappa}")
        
        self.regret_arr = []
        self.theta_norms = []
        self.best_arm_gaps = []        

        
        if params["alg_name"] == "BatchGLinCB-Fixed":
            self.alg = BatchGLinCB_Fixed(params , self.arms , self.kappa , epsilon) 
        elif params["alg_name"] == "SoftBatch":
            self.alg = SoftBatch(params , self.arms , self.kappa , epsilon)

        self.batch_endpoints = self.alg.batch_endpoints

        # DEBUG
        # self.outfile = open(params["path"] + "/outfile.txt" , "a")
        # self.outfile.write("STARTING\n")

    def find_best_arm(self):
        """
        returns the best arm index, the best arm, and the expected reward for this arm given an arm set
        """
        expected_rewards = [self.oracle.expected_reward(arm) for arm in self.arms]
        best_arm_idx = np.argmax(expected_rewards)
        best_arm = self.arms[best_arm_idx]
        best_arm_expected_reward = expected_rewards[best_arm_idx]
        # print(f"DEBUG: Best arm is {best_arm} index {best_arm_idx} and expected reward {expected_rewards[best_arm_idx]}")
        expected_rewards.sort()
        # print(f"DEBUG: Second best arm has expected_reward {expected_rewards[-2]}")
        return best_arm_idx , best_arm , best_arm_expected_reward
    
    def find_kappa(self , theta):
        """
        finds kappa for a given arm set (with repsect to theta_star)
        """
        mu_dot = [dsigmoid(np.dot(theta , arm)) for arm in self.arms]
        return 1.0/np.min(mu_dot)
        # DEBUG: to find the minimum linear and non-linear gaps
        # linear_gaps = [np.dot(self.best_arm - arm , self.theta_star) for arm in self.arms]
        # non_linear_gaps = [sigmoid(np.dot(self.best_arm , self.theta_star)) - sigmoid(np.dot(arm , self.theta_star)) for arm in self.arms]
        # linear_gaps.sort()
        # non_linear_gaps.sort()
        # print("Minimum linear gap is " , linear_gaps[1])
        # print("Minimum non-linear gap is " , non_linear_gaps[1])
    
    def run_algorithm(self):
        """
        runs the algorithm in the standard format: choose an arm, obtain reward, update params
        """
        for t in range(1 , self.horizon + 1):
            # new was used to ensure a new arm is being pulled and to print its information
            arm_played , new  = self.alg.play(t)    
            try:
                reward = self.oracle.reward(arm_played)
            except:
                # incase due to some floating point errors caused by scaling
                for a in self.arms:
                    if np.allclose(a , arm_played):
                        arm_played = a
                        break
                reward = self.oracle.reward(arm_played)
            
            # DEBUG
            # if new and batch==1:
            # if new:
                # print(f"Associated gap is {self.oracle.expected_reward(arm_played)} and associated regret is {self.best_arm_expected_reward - self.oracle.expected_reward(arm_played)}")
                # print(f"Associated linear gap is {np.dot(self.best_arm  - arm_played , self.theta_star)}")

            self.regret_arr.append(self.best_arm_expected_reward - self.oracle.expected_reward(arm_played))
            if t != self.horizon:
                theta , predicted_best_arm = self.alg.update_params(t , reward , arm_played)
                # DEBUG: captures how the norms of predicted theta and gaps of predicted best arm change with batches
                # if theta is not None:
                    # print(f"The new theta is {theta} with distance {np.linalg.norm(self.theta_star - theta)} away")
                    # self.theta_norms.append(np.linalg.norm(self.theta_star - theta))
                    # print(self.theta_norms)
                    # self.best_arm_gaps.append(self.best_arm_expected_reward - self.oracle.expected_reward(predicted_best_arm))
                    # self.outfile.write("NEW BATCH\n")
        # assert len(self.regret_arr) == self.horizon

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
    
    def correct_arms(self , threshold):
        """
        corrects the arm set genrated by remoiving arms which are within 
        {threshold} of best arm
        """
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