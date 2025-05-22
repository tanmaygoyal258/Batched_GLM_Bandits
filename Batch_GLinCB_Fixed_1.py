import numpy as np
import tqdm as tqdm
from utils import *
from Barycentric_Spanner import LWS

class Batch_GLinCB_Fixed_1:

    def __init__(self , params , arms , kappa , epsilon):
        self.params = params
        
        self.param_norm_ub = params["param_norm_ub"]
        self.horizon = params["horizon"]
        self.epsilon = epsilon
        self.failure_level = params["failure_level"]  
        self.dim = params["dimension"]

        self.current_batch = 0
        self.lmbda = 0.5
        self.kappa = kappa
        if self.epsilon != 0.0:
            self.gamma = self.param_norm_ub * np.sqrt(self.dim * np.log(self.horizon / self.failure_level)) * (self.kappa * self.epsilon**2)**0.5
        else:
            self.gamma = 0.01 * self.param_norm_ub * self.param_norm_ub * np.sqrt(self.dim * np.log(self.horizon / self.failure_level))

        self.batch_lengths , self.batch_endpoints = self.calculate_batch_endpoints()
        print(f"Batch Lengths are {self.batch_lengths}")

        self.reward_arr = []
        self.arms_arr = []
        
        self.theta_rng = np.random.default_rng(params["theta_seed"])

        np.random.seed(0)
        self.theta = np.array([0.0 for i in range(self.dim)])
        self.theta_warmup = np.random.rand(self.dim)
        self.theta_warmup /= np.linalg.norm(self.theta_warmup) / self.param_norm_ub

        # self.H = np.zeros((self.dim , self.dim))
        # self.V = np.zeros((self.dim , self.dim))
        
        self.arms = arms
        self.best_arm = self.arms[np.random.randint(len(self.arms))]
        
        self.ball_spanning_vectors = 1/self.horizon * np.identity(self.dim)
        self.warmup_barycentric_spanner = LWS(params , self.arms + self.ball_spanning_vectors.tolist() , 1,  0 , self.best_arm , self.theta , warmup_flag = True)
        self.final_warmup_barycentric_spanner = []
        for arm in self.warmup_barycentric_spanner:
            if arm in self.ball_spanning_vectors:
                continue
            self.final_warmup_barycentric_spanner.append(arm)
        # self.warmup_G_design = G_Optimal_Design(self.arms , self.dim)
        # self.upper_bound_warmup = [self.warmup_G_design[i] * self.batch_lengths[0] for i in range(len(self.warmup_G_design))]
        self.arm_idx_being_played = 0
        self.current_arm_pulls = 0

        # self.outfile = open(params["path"] + "/outfile.txt" , "a")
        print(f"Starting Warmup")
        # print(f"self.theta_warmup is {self.theta_warmup}")
        print(f"The barycentric spanner chosen is {self.final_warmup_barycentric_spanner}")

        print(f"Starting Warmup Batch...")

    def calculate_batch_endpoints(self):
        total_batches = int(np.ceil(np.log(np.log(self.horizon)))) + 1
        print(f"Total number of batches are {total_batches}")
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
        if self.current_batch == 0:
            # upper_bound_on_arm_being_played =  self.upper_bound_warmup[self.arm_idx_being_played]
            upper_bound_on_arm_being_played =  self.batch_lengths[0] / self.dim

            new = False
            if self.current_arm_pulls <= 1:
                    print(f"Playing an arm {upper_bound_on_arm_being_played} times")
                    new = True
            if self.current_arm_pulls <= upper_bound_on_arm_being_played:
                self.current_arm_pulls += 1
                return self.arms[self.arm_idx_being_played] , new
            else:
                self.arm_idx_being_played += 1
                self.current_arm_pulls = 1
                return self.arms[self.arm_idx_being_played] , new
        
        else:
            if self.arm_idx_being_played < len(self.final_barycentric_spanner):
                arm = self.final_barycentric_spanner[self.arm_idx_being_played]
                current_gap = np.dot(self.best_arm , self.theta) - np.dot(arm , self.theta) if self.current_batch != 1 else self.param_norm_ub
                upper_bound_on_arm_being_played = np.ceil(self.batch_lengths[self.current_batch] * self.scale**2 / (8 * self.scalings[tuple(arm)]**2 * self.dim * (1 + self.eta * self.scale * current_gap)**2))
                new = False
                if self.current_arm_pulls <= 1:
                    print(f"Playing an arm {upper_bound_on_arm_being_played} times with estimated_gap {current_gap}")
                    new = True
                if self.current_arm_pulls <= upper_bound_on_arm_being_played:
                    self.current_arm_pulls += 1
                    return arm , new
                else:
                    self.arm_idx_being_played += 1
                    self.current_arm_pulls = 1
                    if self.arm_idx_being_played == len(self.final_barycentric_spanner):
                        current_gap = 0
                        print(f"Completed Exploration in {time_round - self.batch_endpoints[self.current_batch-1]} rounds (fraction : {(time_round - self.batch_endpoints[self.current_batch-1])/self.batch_lengths[self.current_batch]})")
                        print(f"Playing BEST arm {self.batch_endpoints[self.current_batch] - time_round} times with estimated_gap {current_gap}")
                        return self.best_arm , True
                    return self.final_barycentric_spanner[self.arm_idx_being_played] , new
            else:
                return self.best_arm , False
                    

    def update_params(self , time_round , reward , arm_played):
        self.reward_arr.append(reward)
        self.arms_arr.append(arm_played)
        
        if time_round == self.batch_endpoints[0]:
            print(f"Updating Parameters after Warmup Batch")
            # update theta
            thth, succ_flag = solve_mle(self.theta_warmup, np.array(self.arms_arr), np.array(self.reward_arr) , self.lmbda)
            if succ_flag:
                self.theta_warmup = thth 
            self.theta_warmup /= np.linalg.norm(self.theta_warmup) / self.param_norm_ub
            # print(f"The new theta warmup is {self.theta_warmup}")        


            # # update V        
            # self.V = self.lmbda * np.eye(self.dim)
            # for arm in self.arms_arr:
            #     self.V += np.outer(arm , arm)

            # update H
            # self.H = self.lmbda * np.eye(self.dim)
            # for arm in self.arms_arr:
            #     self.H += dsigmoid(np.dot(self.theta_warmup , arm)) * np.outer(arm , arm)

            # update active arm_set
            # prev_arm_set = self.arms.copy()
            # maximum_LCB = -np.inf
            # for arm in prev_arm_set:
            #     # arm_score = np.dot(self.theta_warmup , arm) - self.gamma * np.sqrt(self.kappa) * np.sqrt(np.dot(arm , np.linalg.inv(self.V) @ arm))
            #     arm_score = np.dot(self.theta_warmup , arm) - self.gamma * np.sqrt(np.dot(arm , np.linalg.inv(self.H) @ arm))
            #     maximum_LCB = max(maximum_LCB , arm_score)
            # print("Maximum LCB: ", maximum_LCB)
            # # print("Minimum Score: ", np.min([np.dot(self.theta_warmup , arm) + self.gamma * np.sqrt(self.kappa) * np.sqrt(np.dot(arm , np.linalg.inv(self.V) @ arm)) for arm in prev_arm_set]))
            # print("Minimum Score: ", np.min([np.dot(self.theta_warmup , arm) + self.gamma * np.sqrt(np.dot(arm , np.linalg.inv(self.H) @ arm)) for arm in prev_arm_set]))

            # # self.arms = [arm for arm in prev_arm_set if np.dot(self.theta_warmup , arm) + self.gamma * np.sqrt(self.kappa) * np.sqrt(np.dot(arm , np.linalg.inv(self.V) @ arm)) >= maximum_LCB]
            # self.arms = [arm for arm in prev_arm_set if np.dot(self.theta_warmup , arm) + self.gamma * np.sqrt(np.dot(arm , np.linalg.inv(self.H) @ arm)) >= maximum_LCB]

            # print(f"Rewards")
            # zeros = 0
            # for r in self.reward_arr:
            #     if r == 0:
            #         zeros += 1
            # print(f"Zeros : {zeros} , Ones : {len(self.reward_arr) - zeros}")



            # print(f"Arms eliminated:")
            # for arm in prev_arm_set:
            #     if arm not in self.arms:
            #         print(f"{arm}")

            # find the minimum scaling
            self.scale = np.min([np.sqrt(dsigmoid(np.dot(self.theta_warmup , arm))) for arm in self.arms])
            # self.scale = 1e-8   
            print(f"The scaling factor is {self.scale}")

            # calculate scaled_arms
            # self.scaled_arms = [self.scale * arm for arm in self.arms]
        
            # maintain a lookup table for scalings for ease
            self.scalings = {tuple(arm) : np.sqrt(dsigmoid(np.dot(self.theta_warmup , arm))) for arm in self.arms}

            # print("Scalings are ", self.scalings)
            # print("True arms are ", self.arms)
            # print("Scaled arms are ", self.scaled_arms)
            # print(f"Starting True Batch {self.current_batch} with {len(self.arms)} arms")

            # find the next barycentric spanner
            self.eta = 1 / (8 * self.gamma * self.dim * np.exp(4))
            self.best_arm = approximate_additive_oracle(self.arms , self.theta , 1/self.horizon , time_round)
            self.barycentric_spanner = LWS(self.params , self.arms + self.ball_spanning_vectors.tolist() , self.scale , self.eta , self.best_arm , self.theta , warmup_flag = False)
            self.final_barycentric_spanner = []
            for arm in self.barycentric_spanner:
                if arm in self.ball_spanning_vectors:
                    continue
                self.final_barycentric_spanner.append(arm)
            print(f"Final Barycentric Spanner chosen is \n{self.final_barycentric_spanner}")
            # self.final_barycentric_spanner = self.scaled_arms

            self.arms_arr = []
            self.reward_arr = []
            self.arm_idx_being_played = 0
            self.current_arm_pulls = 0
            self.current_batch += 1
            print(f"Starting Batch {self.current_batch} with length {self.batch_lengths[self.current_batch]}...")\
            
            return None, None

        elif time_round in self.batch_endpoints:
            thth, succ_flag = solve_mle(self.theta, np.array(self.arms_arr), np.array(self.reward_arr) , self.lmbda)
            if succ_flag:
                self.theta = thth 
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
            print(f"Final Barycentric Spanner chosen is \n{self.final_barycentric_spanner}")
            

            self.arms_arr = []
            self.reward_arr = []
            self.arm_idx_being_played = 0
            self.current_arm_pulls = 0
            self.current_batch += 1
            print(f"Starting Batch {self.current_batch} with length {self.batch_lengths[self.current_batch]}...")

            return self.theta , self.best_arm

        
        else:
            return None , None
        
