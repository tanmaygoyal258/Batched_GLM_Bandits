import numpy as np
import tqdm as tqdm
from utils import *
from Barycentric_Spanner import LWS

class SoftBatch:
    """
    Implements the SoftBatch algorithm from Osama Hanna's paper for Logistic Bandits
    """

    def __init__(self , params , arms , kappa , epsilon):
        self.params = params
        
        # variables set from params
        self.param_norm_ub = params["param_norm_ub"]
        self.horizon = params["horizon"]
        self.epsilon = epsilon
        self.failure_level = params["failure_level"]  
        self.dim = params["dimension"]

        # other variables
        self.current_batch = 0
        self.lmbda = 1
        self.kappa = kappa
        if self.epsilon != 0.0:
            self.gamma = self.param_norm_ub * np.sqrt(self.dim * np.log(self.horizon / self.failure_level)) * (self.kappa * self.epsilon**2)**0.5
        else:
            self.gamma = 0.01 * self.param_norm_ub * self.param_norm_ub * np.sqrt(self.dim * np.log(self.horizon / self.failure_level))
        self.eta = 1 / (8 * self.gamma * self.dim * np.exp(4))
        
        # calculate the batch_lengths
        self.batch_lengths , self.batch_endpoints = self.calculate_batch_endpoints()

        # initialize theta
        np.random.seed(0)
        self.theta = np.array([0.0 for i in range(self.dim)])

        # initialize arm set and predicted best arm
        self.arms = arms
        self.best_arm = self.arms[np.random.randint(len(self.arms))]
        
        # calculate the initial barycentric spanner
        self.ball_spanning_vectors = 1/self.horizon * np.identity(self.dim)
        self.barycentric_spanner = LWS(params , self.arms + self.ball_spanning_vectors.tolist() , 1 ,  self.eta , self.best_arm , self.theta , warmup_flag = False)
        self.final_barycentric_spanner = []
        for arm in self.barycentric_spanner:
            if arm in self.ball_spanning_vectors:
                continue
            self.final_barycentric_spanner.append(arm)

        # initialize a few variables
        self.arm_idx_being_played = 0
        self.current_arm_pulls = 0
        self.reward_arr = []    
        self.arms_arr = []
        self.scale = 1


    def calculate_batch_endpoints(self):
        """
        Calculates the batch lengths
        """
        total_batches = int(np.ceil(np.log(np.log(self.horizon)))) + 1
        warmup_length = min(np.sqrt(self.horizon) , 1000)
        # batch_lengths = [warmup_length]
        batch_lengths = []
        batch_lengths_post_warmup = [(self.horizon ** (1 - 2**(-i))) for i in range(1 , total_batches+1)]
        batch_lengths_post_warmup[-1] = self.horizon
        batch_lengths += batch_lengths_post_warmup
        batch_endpoints = np.cumsum(batch_lengths)
        batch_endpoints = np.clip(batch_endpoints , 0 , self.horizon)
        batch_endpoints = batch_endpoints.astype(int)
        final_batch_endpoints = []
        for element in batch_endpoints:
            final_batch_endpoints.append(element)
            if element == self.horizon:
                break
        return batch_lengths , final_batch_endpoints
    
    def play(self , time_round):
        
        # if arm belongs to barycentric spanner 
        if self.arm_idx_being_played < len(self.final_barycentric_spanner):
            arm = self.final_barycentric_spanner[self.arm_idx_being_played]
            current_gap = np.dot(self.best_arm , self.theta) - np.dot(arm , self.theta) if self.current_batch != 0 else self.param_norm_ub
            arm_scaling = 1
            upper_bound_on_arm_being_played = self.batch_lengths[self.current_batch] * self.scale**2 / (8 * arm_scaling**2 * self.dim * (1 + self.eta * self.scale * current_gap)**2)
            upper_bound_on_arm_being_played = np.ceil(upper_bound_on_arm_being_played)
            
            # if an arm has been played less than 1 times, it is a new arm
            new = False
            if self.current_arm_pulls <= 1:
                # print(f"DEBUG: Playing an arm {upper_bound_on_arm_being_played} times with estimated_gap {current_gap}")
                new = True
            if self.current_arm_pulls <= upper_bound_on_arm_being_played:
                self.current_arm_pulls += 1
                return arm , new 
            else:
                self.arm_idx_being_played += 1
                self.current_arm_pulls = 1
                if self.arm_idx_being_played == len(self.final_barycentric_spanner):
                    # completed barycentric spanner
                    # print(f"DEBUG: Completed Exploration in {time_round - self.batch_endpoints[self.current_batch-1]} rounds (fraction : {(time_round - self.batch_endpoints[self.current_batch-1])/self.batch_lengths[self.current_batch]})")
                    # print(f"DEBUG: Playing BEST arm {self.batch_endpoints[self.current_batch] - time_round} times with estimated_gap {current_gap}")
                    current_gap = 0
                    return self.best_arm , True 
                return self.final_barycentric_spanner[self.arm_idx_being_played] , new 
        
        # we have exhausted barycentric spanner, play best arm
        else:
            return self.best_arm , False 
                

    def update_params(self , time_round , reward , arm_played):
        """
        Updates the parameters at the end of the batch
        """

        self.reward_arr.append(reward)
        self.arms_arr.append(arm_played)
        
        if time_round in self.batch_endpoints:
            # find MLE estimate
            thth, succ_flag = solve_mle_sklearn(self.theta, np.array(self.arms_arr), np.array(self.reward_arr) , self.lmbda)
            if succ_flag:
                self.theta = thth 
            if np.linalg.norm(self.theta) > 0:
                self.theta /= np.linalg.norm(self.theta) / self.param_norm_ub

            # update self.best_arm
            self.best_arm = approximate_additive_oracle(self.arms , self.theta , 1/self.horizon  ,time_round)

            # find the next barycentric spanner
            self.eta = self.batch_lengths[self.current_batch] / (8 * self.gamma * self.dim * np.exp(4))
            self.barycentric_spanner = LWS(self.params , self.arms + self.ball_spanning_vectors.tolist() , self.scale , self.eta , self.best_arm , self.theta , warmup_flag = False)
            self.final_barycentric_spanner = []
            for arm in self.barycentric_spanner:
                if arm in self.ball_spanning_vectors:
                    continue
                self.final_barycentric_spanner.append(arm)

            # reset variables
            self.arms_arr = []
            self.reward_arr = []
            self.arm_idx_being_played = 0
            self.current_arm_pulls = 0
            self.current_batch += 1
            return self.theta , self.best_arm

        # time point is a part of batch and hence, no update takes place
        else:
            return None , None
        
