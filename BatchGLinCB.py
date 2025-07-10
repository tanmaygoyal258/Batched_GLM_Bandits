import numpy as np
from tqdm import tqdm
from utils import *
import os
from Barycentric_Spanner_Reduction import LWS_Reduction
from Reduced_ArmSet import Reduced_ArmSet



class BatchGLinCB:
    """
    Runs the algorithm BatchGLinCB
    """
    def __init__(self , params , kappa):
        # setting variables from params
        self.params = params
        self.param_norm_ub = params["param_norm_ub"]
        self.horizon = params["horizon"]
        self.failure_level = params["failure_level"]  
        self.dim = params["dimension"]
        
        # setting other variables
        self.current_batch = 0
        self.lmbda = self.dim * np.log(self.horizon)
        self.glm_upper_bound = 1
        self.kappa = kappa
        # calculating batch lengths
        self.batch_lengths , self.batch_endpoints = self.calculate_batch_endpoints()

        # initializing variables
        self.reward_arr = []
        self.arms_arr = []
        self.arm_idx_being_played = 0
        self.current_arm_pulls = 0

        # initializing the thetas and armsets
        np.random.seed(0)
        self.theta = np.array([0.0 for i in range(self.dim)])
        self.theta_warmup = np.random.rand(self.dim)
        self.theta_warmup /= np.linalg.norm(self.theta_warmup) / self.param_norm_ub
        self.theta_rng = np.random.default_rng(params["theta_seed"])
        self.fixed_armset = Reduced_ArmSet()
        self.q = np.finfo(np.float32).eps

    def calculate_batch_endpoints(self):
        """
        Calculates the batch lengths as mentioned in the paper
        """
        total_batches = int(np.ceil(np.log(np.log(self.horizon)))) + 1
        warmup_length = np.sqrt(self.horizon)
        batch_lengths = [warmup_length , warmup_length] # to account for batch 0 and batch -1
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

    def play(self , time_round , arm_set):
        # warmup batch : play arms randomly
        if self.current_batch == 0: 
            arm_played = arm_set[np.random.randint(len(arm_set))]
            self.arms_arr.append(arm_played)
            return arm_played , "arm"
        
        # non-warmup batches
        else:
            # if the arm belongs to the barycentric spanner
            if self.arm_idx_being_played < len(self.final_barycentric_spanner):
                arm = self.final_barycentric_spanner[self.arm_idx_being_played]
                current_gap = np.dot(self.best_arm , self.theta) - np.dot(arm , self.theta) if self.current_batch != 1 else self.param_norm_ub
                arm_scaling = np.sqrt(dsigmoid(np.dot(self.theta_warmup , arm)) / self.glm_upper_bound)
                upper_bound_on_arm_being_played = self.batch_lengths[self.current_batch] * self.scale**2 / (8 * arm_scaling**2 * self.dim * (1 + self.eta * self.scale * current_gap)**2)
                upper_bound_on_arm_being_played = min(upper_bound_on_arm_being_played , self.batch_lengths[self.current_batch]/self.dim)
                upper_bound_on_arm_being_played = upper_bound_on_arm_being_played / self.glm_upper_bound if self.glm_upper_bound >= 1 else upper_bound_on_arm_being_played
                upper_bound_on_arm_being_played = np.ceil(upper_bound_on_arm_being_played)
                if self.current_arm_pulls <= upper_bound_on_arm_being_played:
                    self.current_arm_pulls += 1
                    self.arms_arr.append(arm)
                    # return the theta corresponding to that arm for the contextual alg
                    return np.array(self.dict_arms_thetas[tuple(arm)]) , "theta"
                else:
                    self.arm_idx_being_played += 1
                    self.current_arm_pulls = 1
                    if self.arm_idx_being_played == len(self.final_barycentric_spanner):
                        # we have reached end of barycentric spanner
                        self.arms_arr.append(self.best_arm)
                        return self.best_theta , "theta"
                    arm = self.final_barycentric_spanner[self.arm_idx_being_played]
                    self.arms_arr.append(arm)
                    
                    # return the theta corresponding to that arm for the contextual alg
                    return np.array(self.dict_arms_thetas[tuple(arm)]) , "theta"
            
            # else play the best arm
            else:
                self.arms_arr.append(np.array(self.dict_thetas_arms[tuple(self.best_theta)]))
                return self.best_theta , "theta"

    
    def update_params(self , time_round , reward , previous_arm_sets):
        """
        updates the parameters after each Batch
        """
        self.reward_arr.append(reward)
        
        # batch -1
        if time_round == self.batch_endpoints[0]:
            # update the set of previous arms for the fixed armset
            self.fixed_armset.update_sets(previous_arm_sets)
            
            # initialize the variables
            self.epsilon = np.sqrt(self.dim / self.batch_lengths[self.current_batch])
            self.gamma = self.param_norm_ub * np.sqrt(self.dim * np.log(self.horizon / self.failure_level)) * (self.kappa * self.epsilon**2)**0.5
            self.eta = 0
            self.scale = 1
            
            # find the next barycentric spanner
            self.best_theta = self.q * np.floor(self.theta_warmup * np.sqrt(self.dim)/self.q) / np.sqrt(self.dim)
            self.best_arm = self.fixed_armset.calculate_fixed_arm(self.best_theta)
            LWS_class = LWS_Reduction(self.params , self.fixed_armset)
            self.barycentric_spanner , self.dict_arms_thetas , self.dict_thetas_arms = LWS_class.LWS(self.scale , self.eta , self.best_arm , self.theta_warmup , warmup_flag = False)
            self.final_barycentric_spanner = self.barycentric_spanner

            # add the best theta and best arm to the dictionaries
            self.dict_arms_thetas[tuple(self.best_arm)] = tuple(self.best_theta)
            self.dict_thetas_arms[tuple(self.best_theta)] = tuple(self.best_arm)

            # reset the variables
            self.arms_arr = []
            self.reward_arr = []
            self.arm_idx_being_played = 0
            self.current_arm_pulls = 0
            self.current_batch += 1
            return True

        # batch 0
        elif time_round == self.batch_endpoints[1]:
            # calculate MLE estimate of theta_warmup
            thth, succ_flag = solve_mle_sklearn(self.theta_warmup, np.array(self.arms_arr), np.array(self.reward_arr) , self.lmbda)
            if succ_flag:
                self.theta_warmup = thth 
            if np.linalg.norm(self.theta_warmup) > 0:
                self.theta_warmup /= np.linalg.norm(self.theta_warmup) / self.param_norm_ub

            # update the set of previous arms for the fixed armset
            self.fixed_armset.update_sets(previous_arm_sets)
            self.best_theta = self.q * np.floor(self.theta_warmup * np.sqrt(self.dim)/self.q) / np.sqrt(self.dim)
            self.best_arm = self.fixed_armset.calculate_fixed_arm(self.best_theta)
            
            # update the variables
            self.epsilon = np.sqrt(self.dim / self.batch_lengths[self.current_batch])
            self.gamma = self.param_norm_ub * np.sqrt(self.dim * np.log(self.horizon / self.failure_level)) * (self.kappa * self.epsilon**2)**0.5
            self.eta = self.batch_lengths[self.current_batch] / (50 * self.gamma * self.dim * np.exp(4))
            self.scale = np.sqrt(dsigmoid(np.dot(self.theta_warmup , self.best_arm)) / self.glm_upper_bound)        
            
            # find the next barycentric spanner
            LWS_class = LWS_Reduction(self.params , self.fixed_armset)
            self.barycentric_spanner , self.dict_arms_thetas , self.dict_thetas_arms = LWS_class.LWS(self.scale , self.eta , self.best_arm , self.theta , warmup_flag = False)
            self.final_barycentric_spanner = self.barycentric_spanner

            # add the best theta and best arm to the dictionaries
            self.dict_arms_thetas[tuple(self.best_arm)] = tuple(self.best_theta)
            self.dict_thetas_arms[tuple(self.best_theta)] = tuple(self.best_arm)

            # reset the variables
            self.arms_arr = []
            self.reward_arr = []
            self.arm_idx_being_played = 0
            self.current_arm_pulls = 0
            self.current_batch += 1
            return True
        
        # non-warmup batch
        elif time_round in self.batch_endpoints:
            # calculate MLE estimate of theta
            thth, succ_flag = solve_mle_sklearn(self.theta, np.array(self.arms_arr), np.array(self.reward_arr) , self.lmbda)
            if succ_flag:
                self.theta = thth 
            if np.linalg.norm(self.theta) > 0:
                self.theta /= np.linalg.norm(self.theta) / self.param_norm_ub

            # update the set of previous arms for the fixed armset
            self.fixed_armset.update_sets(previous_arm_sets)
            self.best_theta = self.q * np.floor(self.theta * np.sqrt(self.dim)/self.q) / np.sqrt(self.dim)
            self.best_arm = self.fixed_armset.calculate_fixed_arm(self.best_theta)
            
            # update the variables
            self.epsilon = np.sqrt(self.dim / self.batch_lengths[self.current_batch])
            self.gamma = self.param_norm_ub * np.sqrt(self.dim * np.log(self.horizon / self.failure_level)) * (self.kappa * self.epsilon**2)**0.5
            self.eta = self.batch_lengths[self.current_batch] / (50 * self.gamma * self.dim * np.exp(4))
            self.scale = np.sqrt(dsigmoid(np.dot(self.theta_warmup , self.best_arm)) / self.glm_upper_bound)        
            
            # find the next barycentric spanner
            LWS_class = LWS_Reduction(self.params , self.fixed_armset)
            self.barycentric_spanner , self.dict_arms_thetas , self.dict_thetas_arms = LWS_class.LWS(self.scale , self.eta , self.best_arm , self.theta , warmup_flag = False)
            self.final_barycentric_spanner = self.barycentric_spanner

            # add the best theta and best arm to the dictionaries
            self.dict_arms_thetas[tuple(self.best_arm)] = tuple(self.best_theta)
            self.dict_thetas_arms[tuple(self.best_theta)] = tuple(self.best_arm)

            # reset the variables
            self.arms_arr = []
            self.reward_arr = []
            self.arm_idx_being_played = 0
            self.current_arm_pulls = 0
            self.current_batch += 1
            return True

        # time point is a part of a batch and hence, no updates
        else:
            return False