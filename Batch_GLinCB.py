import numpy as np
from tqdm import tqdm
from utils import *
import os
from Barycentric_Spanner_Reduction import LWS_Reduction
from Reduced_ArmSet import Reduced_ArmSet



class Batch_GLinCB:

    def __init__(self , params , kappa):
        self.param_norm_ub = params["param_norm_ub"]
        self.horizon = params["horizon"]
        self.failure_level = params["failure_level"]  
        self.dim = params["dimension"]
        self.epsilon = np.sqrt(self.dim / self.horizon**0.25)
        self.batch_lengths , self.batch_endpoints = self.calculate_batch_endpoints()
        self.current_batch = 0
        self.lmbda = 0.5
        self.kappa = kappa
        print(f"The value of kappa is {self.kappa}")
        self.gamma = self.param_norm_ub * np.sqrt(self.dim * np.log(self.horizon / self.failure_level)) * (self.kappa * self.epsilon**2)**0.5
        self.params = params

        self.reward_arr = []
        self.arms_arr = []
        self.arm_idx_being_played = 0
        self.current_arm_pulls = 0
        
        self.scale = 0.0075

        # we save the epsilon nets to repeat calculating them over and over
        folder = "epsilon_nets"
        if not os.path.exists(folder):
            os.makedirs(folder)
        if f"Epsilon_Net_10000_S={self.param_norm_ub}.npy" in os.listdir(folder):
            print(f"Loading Epsilon Net")
            self.discrete_theta_set = np.load(f"{folder}/Epsilon_Net_10000_S={self.param_norm_ub}.npy")
        else:
            assert False, "Epsilon Net not found"


        np.random.seed(0)
        self.theta = np.array([0.0 for i in range(self.dim)])
        self.theta_warmup = np.random.rand(self.dim)
        self.theta_warmup /= np.linalg.norm(self.theta_warmup) / self.param_norm_ub
        self.theta_rng = np.random.default_rng(params["theta_seed"])
        self.fixed_armset = Reduced_ArmSet(self.discrete_theta_set , self.theta_rng)

        print("Starting Warmup Batch = Batch 0")

    def calculate_batch_endpoints(self):
        total_batches = int(np.ceil(np.log(np.log(self.horizon)))) + 1
        print(f"Total number of batches are {total_batches}")
        warmup_length = np.sqrt(self.horizon)
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
        print(f"Batch Endpoints are {final_batch_endpoints} and Batch Lengths are {batch_lengths}")
        return batch_lengths , final_batch_endpoints

    def play(self , time_round , arm_set):
        
        if self.current_batch == 0:
            g_optimal_design = G_Optimal_Design(arm_set , self.dim)
            arm_idx = np.random.choice(len(arm_set) , p = g_optimal_design)        
            self.arms_arr.append(arm_set[arm_idx])
            return arm_set[arm_idx] , "arm"

        else:
            # print(self.fixed_armset.dict_arm_to_theta)
            if self.arm_idx_being_played < len(self.final_barycentric_spanner):
                arm = self.final_barycentric_spanner[self.arm_idx_being_played]
                scaling = np.sqrt(dsigmoid(np.dot(self.theta_warmup , arm)))
                current_gap = np.dot(self.best_arm , self.theta) - np.dot(arm , self.theta) if self.current_batch != 1 else self.param_norm_ub
                upper_bound_on_arm_being_played = np.ceil(self.batch_lengths[self.current_batch] * self.scale**2 / (8 * scaling**2 * self.dim * (1 + self.eta * self.scale * current_gap)**2))
                if self.current_arm_pulls <= upper_bound_on_arm_being_played:
                    self.current_arm_pulls += 1
                    self.arms_arr.append(arm)
                    return np.array(self.dict_arms_thetas[tuple(arm)]) , "theta"
                else:
                    self.arm_idx_being_played += 1
                    self.current_arm_pulls = 1
                    if self.arm_idx_being_played == len(self.final_barycentric_spanner):
                        self.arms_arr.append(np.array(self.dict_thetas_arms[tuple(self.best_theta)]))
                        return self.best_theta , "theta"
                    arm = self.final_barycentric_spanner[self.arm_idx_being_played]
                    self.arms_arr.append(arm)
                    return np.array(self.dict_arms_thetas[tuple(arm)]) , "theta"
            else:
                self.arms_arr.append(np.array(self.dict_thetas_arms[tuple(self.best_theta)]))
                return self.best_theta , "theta"

    
    def update_params(self , time_round , reward , previous_arm_sets):
        self.reward_arr.append(reward)
        
        if time_round == self.batch_endpoints[0]:
            print(f"Updating Parameters after Warmup Batch at round {time_round}")
            # update theta
            thth, succ_flag = solve_mle(self.theta_warmup, np.array(self.arms_arr), np.array(self.reward_arr) , self.lmbda)
            if succ_flag:
                self.theta_warmup = thth 
            self.theta_warmup /= np.linalg.norm(self.theta_warmup) / self.param_norm_ub

            self.fixed_armset.update_sets(self.discrete_theta_set , previous_arm_sets)
            self.best_theta = self.discrete_theta_set[np.random.randint(len(self.discrete_theta_set))]
            self.best_arm = self.fixed_armset.calculate_fixed_arm(self.best_theta)
            
            # self.epsilon = np.sqrt(self.dim / self.batch_lengths[self.current_batch - 1])
            self.gamma = self.param_norm_ub * np.sqrt(self.dim * np.log(self.horizon / self.failure_level)) * (self.kappa * self.epsilon**2)**0.5
            self.eta = 1 / (8 * self.gamma * self.dim * np.exp(4))
            
            # TODO: find next barycentric spanner
            print(f"Starting Batch {self.current_batch+1} with length {self.batch_lengths[self.current_batch+1]}...")
            self.fixed_armset = Reduced_ArmSet(self.discrete_theta_set , np.random.default_rng(0))
            self.fixed_armset.update_sets(self.discrete_theta_set , previous_arm_sets)
            LWS_class = LWS_Reduction(self.params , self.discrete_theta_set , previous_arm_sets , self.fixed_armset)
            self.barycentric_spanner , self.dict_arms_thetas , self.dict_thetas_arms = LWS_class.LWS(self.scale , self.eta , self.best_arm , self.theta , warmup_flag = False)
            self.final_barycentric_spanner = self.barycentric_spanner
            print(f"Final Barycentric Spanner chosen is \n{self.final_barycentric_spanner}")

            # add the best theta and best arm to the dictionaries
            self.dict_arms_thetas[tuple(self.best_arm)] = tuple(self.best_theta)
            self.dict_thetas_arms[tuple(self.best_theta)] = tuple(self.best_arm)

            self.arms_arr = []
            self.reward_arr = []
            self.arm_idx_being_played = 0
            self.current_arm_pulls = 0
            self.current_batch += 1
            return True

        elif time_round in self.batch_endpoints:
            print(f"Updating Parameters at round {time_round}")
            thth, succ_flag = solve_mle(self.theta, np.array(self.arms_arr), np.array(self.reward_arr) , self.lmbda)
            if succ_flag:
                self.theta = thth 
            self.theta /= np.linalg.norm(self.theta) / self.param_norm_ub
            self.theta_estimate_in_epsnet = self.discrete_theta_set[np.argmin([np.linalg.norm(self.theta - theta) for theta in self.discrete_theta_set])]

            # update self.best_arm
            self.fixed_armset.update_sets(self.discrete_theta_set , previous_arm_sets)
            self.best_theta = self.theta_estimate_in_epsnet
            self.best_arm = self.fixed_armset.calculate_fixed_arm(self.best_theta)

            self.epsilon = np.sqrt(self.dim / self.batch_lengths[self.current_batch])
            self.gamma = self.param_norm_ub * np.sqrt(self.dim * np.log(self.horizon / self.failure_level)) * (self.kappa * self.epsilon**2)**0.5
            self.eta = self.batch_lengths[self.current_batch] / (8 * self.gamma * self.dim * np.exp(4))

            # TODO: find next barycentric spanner
            print(f"Starting Batch {self.current_batch+1} with length {self.batch_lengths[self.current_batch+1]}...")
            self.fixed_armset = Reduced_ArmSet(self.discrete_theta_set , np.random.default_rng(0))
            self.fixed_armset.update_sets(self.discrete_theta_set , previous_arm_sets)
            LWS_class = LWS_Reduction(self.params , self.discrete_theta_set , previous_arm_sets , self.fixed_armset)
            self.barycentric_spanner , self.dict_arms_thetas , self.dict_thetas_arms = LWS_class.LWS(self.scale , self.eta , self.best_arm , self.theta , warmup_flag = False)
            self.final_barycentric_spanner = self.barycentric_spanner
            print(f"Final Barycentric Spanner chosen is \n{self.final_barycentric_spanner}")
            
            # add the best theta and best arm to the dictionaries
            self.dict_arms_thetas[tuple(self.best_arm)] = tuple(self.best_theta)
            self.dict_thetas_arms[tuple(self.best_theta)] = tuple(self.best_arm)

            self.arms_arr = []
            self.reward_arr = []
            self.arm_idx_being_played = 0
            self.current_arm_pulls = 0
            self.current_batch += 1
            return True

        else:
            return False