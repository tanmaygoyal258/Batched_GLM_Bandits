import numpy as np
import tqdm as tqdm
from utils import *
from Barycentric_Spanner import LWS

class BatchGLinCB_Fixed:
    """
    Runs the algorithm BatchGLinCB-Fixed
    """

    def __init__(self , params , arms , kappa , epsilon):
        self.params = params
        
        # setting variables from params
        self.param_norm_ub = params["param_norm_ub"]
        self.horizon = params["horizon"]
        self.epsilon = epsilon
        self.failure_level = params["failure_level"]  
        self.dim = params["dimension"]

        # setting other variables
        self.current_batch = 0
        self.lmbda = self.dim * np.log(self.horizon)
        self.kappa = kappa
        self.glm_upper_bound = 1
        if self.epsilon != 0.0:
            self.gamma = self.param_norm_ub * np.sqrt(self.dim * np.log(self.horizon / self.failure_level)) * (self.kappa * self.epsilon**2)**0.5
        else:
            self.gamma = 0.01 * self.param_norm_ub * self.param_norm_ub * np.sqrt(self.dim * np.log(self.horizon / self.failure_level))
        self.scale = 1
        self.eta = 0
        
        # finding the batch lengths
        self.batch_lengths , self.batch_endpoints = self.calculate_batch_endpoints()

        # setting the thetas used throughout the algorithm
        self.theta_rng = np.random.default_rng(params["theta_seed"])
        np.random.seed(0)
        self.theta = np.array([0.0 for i in range(self.dim)])
        self.theta_warmup = np.random.rand(self.dim)
        self.theta_warmup /= np.linalg.norm(self.theta_warmup) / self.param_norm_ub

        # setting the arm set and the (predicted) best arm
        self.arms = arms
        self.best_arm = self.arms[np.random.randint(len(self.arms))]
        
        # finding the barycentric spanner for the warmup batch
        self.ball_spanning_vectors = 1/self.horizon * np.identity(self.dim)
        self.warmup_barycentric_spanner = LWS(params , self.arms + self.ball_spanning_vectors.tolist() , self.scale ,  self.eta , self.best_arm , self.theta_warmup , warmup_flag = True)
        self.final_warmup_barycentric_spanner = []
        for arm in self.warmup_barycentric_spanner:
            if arm in self.ball_spanning_vectors:
                continue
            self.final_warmup_barycentric_spanner.append(arm)

        # initializing a few variables
        self.arm_idx_being_played = 0
        self.current_arm_pulls = 0
        self.reward_arr = []
        self.arms_arr = []
        
    def calculate_batch_endpoints(self):
        """
        Calculates the batch lengths as mentioned in the paper
        """
        total_batches = int(np.ceil(np.log(np.log(self.horizon)))) + 1
        # print(f"DEBUG: Total number of batches are {total_batches}")
        warmup_length = min(np.sqrt(self.horizon) , 1000)
        batch_lengths = [warmup_length]
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
        """
        plays arms for a batch given the barycentric spanner
        """

        # warmup batch
        if self.current_batch == 0:
            # if the arm belongs to the barycentric spanner
            if self.arm_idx_being_played < len(self.final_warmup_barycentric_spanner):
                arm = self.final_warmup_barycentric_spanner[self.arm_idx_being_played]
                upper_bound_on_arm_being_played =  self.batch_lengths[0] / self.dim     # play each arm equally
                
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
                    if self.arm_idx_being_played == len(self.final_warmup_barycentric_spanner):
                        # we have reached end of barycentric spanner
                        # print(f"DEBUG: Playing BEST arm {self.batch_endpoints[self.current_batch] - time_round} times with estimated_gap {current_gap}")
                        # print(f"DEBUG: Completed Exploration in {time_round - self.batch_endpoints[self.current_batch-1]} rounds (fraction : {(time_round - self.batch_endpoints[self.current_batch-1])/self.batch_lengths[self.current_batch]})")
                        current_gap = 0
                        return self.best_arm , True 
                    return self.final_warmup_barycentric_spanner[self.arm_idx_being_played] , new 

            # else play the best arm
            else:
                return self.best_arm , False 

        # non-warmup batches
        else:
            # arm is part of barycentric spanner
            if self.arm_idx_being_played < len(self.final_barycentric_spanner):
                arm = self.final_barycentric_spanner[self.arm_idx_being_played]
                current_gap = np.dot(self.best_arm , self.theta) - np.dot(arm , self.theta) if self.current_batch != 1 else self.param_norm_ub
                arm_scaling = np.sqrt(dsigmoid(np.dot(self.theta_warmup , arm)) / self.glm_upper_bound)
                upper_bound_on_arm_being_played = self.batch_lengths[self.current_batch] * self.scale**2 / (8 * arm_scaling**2 * self.dim * (1 + self.eta * self.scale * current_gap)**2)
                upper_bound_on_arm_being_played = min(upper_bound_on_arm_being_played , self.batch_lengths[self.current_batch]/self.dim)
                upper_bound_on_arm_being_played = upper_bound_on_arm_being_played / self.glm_upper_bound if self.glm_upper_bound >= 1 else upper_bound_on_arm_being_played
                upper_bound_on_arm_being_played = np.ceil(upper_bound_on_arm_being_played)
                
                # if an arm has been played for less than 1 times, it is a new arm
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
                        # barycentric spanner is complete
                        # print(f"DEBUG: Completed Exploration in {time_round - self.batch_endpoints[self.current_batch-1]} rounds (fraction : {(time_round - self.batch_endpoints[self.current_batch-1])/self.batch_lengths[self.current_batch]})")
                        # print(f"DEBUG: Playing BEST arm {self.batch_endpoints[self.current_batch] - time_round} times with estimated_gap {current_gap}")
                        current_gap = 0
                        return self.best_arm , True 
                    return self.final_barycentric_spanner[self.arm_idx_being_played] , new 

            # all arms from barycentric spanner have been played, return best arm
            else:
                return self.best_arm , False 
                    

    def update_params(self , time_round , reward , arm_played):
        """
        updates the parameters after each Batch
        """
        self.reward_arr.append(reward)
        self.arms_arr.append(arm_played)
        
        # warmup batch
        if time_round == self.batch_endpoints[0]:
            # calculate MLE estimate of theta_warmup
            thth, succ_flag = solve_mle_sklearn(self.theta_warmup, np.array(self.arms_arr), np.array(self.reward_arr) , self.lmbda)
            if succ_flag:
                self.theta_warmup = thth 
            if np.linalg.norm(self.theta_warmup) > 0:
                self.theta_warmup /= np.linalg.norm(self.theta_warmup) / self.param_norm_ub
            self.best_arm = approximate_additive_oracle(self.arms , self.theta , 1/self.horizon , time_round)
            self.scale = np.sqrt(dsigmoid(np.dot(self.theta_warmup , self.best_arm)) / self.glm_upper_bound)

            # find the next barycentric spanner
            self.eta = 1 / (50 * self.gamma * self.dim * np.exp(4))
            self.barycentric_spanner = LWS(self.params , self.arms + self.ball_spanning_vectors.tolist() , self.scale , self.eta , self.best_arm , self.theta , warmup_flag = False)
            self.final_barycentric_spanner = []
            for arm in self.barycentric_spanner:
                if arm in self.ball_spanning_vectors:
                    continue
                self.final_barycentric_spanner.append(arm)

            # reset the variables
            self.arms_arr = []
            self.reward_arr = []
            self.arm_idx_being_played = 0
            self.current_arm_pulls = 0
            self.current_batch += 1
            return self.theta_warmup, self.best_arm

        # non-warmup batches
        elif time_round in self.batch_endpoints:
            thth, succ_flag = solve_mle_sklearn(self.theta, np.array(self.arms_arr), np.array(self.reward_arr) , self.lmbda)
            if succ_flag:
                self.theta = thth 
            if np.linalg.norm(self.theta_warmup) > 0:
                self.theta_warmup /= np.linalg.norm(self.theta_warmup) / self.param_norm_ub

            # update predicted best_arm and the scaling factor
            self.best_arm = approximate_additive_oracle(self.arms , self.theta , 1/self.horizon  ,time_round)
            self.scale = np.sqrt(dsigmoid(np.dot(self.theta_warmup , self.best_arm)) / self.glm_upper_bound)        

            # find the next barycentric spanner
            self.eta = self.batch_lengths[self.current_batch] / (50 * self.gamma * self.dim * np.exp(4))
            self.barycentric_spanner = LWS(self.params , self.arms + self.ball_spanning_vectors.tolist() , self.scale , self.eta , self.best_arm , self.theta , warmup_flag = False)
            self.final_barycentric_spanner = []
            for arm in self.barycentric_spanner:
                if arm in self.ball_spanning_vectors:
                    continue
                self.final_barycentric_spanner.append(arm)
            
            # reset the variables
            self.arms_arr = []
            self.reward_arr = []
            self.arm_idx_being_played = 0
            self.current_arm_pulls = 0
            self.current_batch += 1
            return self.theta , self.best_arm

        # the time round is a part of batch and hence, no updates take place        
        else:
            return None , None
        
