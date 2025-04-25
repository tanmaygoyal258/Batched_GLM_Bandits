import numpy as np
from utils import G_Optimal_Design , sigmoid , dsigmoid , solve_mle

class NC_Logistic_Alg:
    
    def __init__(self , params , arms , epsilon , kappa):

        self.param_norm_ub = params["param_norm_ub"]
        self.horizon = params["horizon"]
        self.epsilon = epsilon
        self.failure_level = params["failure_level"]  
        self.dim = params["dimension"]

        self.batch_lengths , self.batch_endpoints = self.calculate_batch_endpoints()
        self.current_batch = 1
        self.lmbda = 0.5
        self.kappa = kappa
        print(f"The value of kappa is {self.kappa}")
        self.gamma = self.param_norm_ub* np.sqrt(self.dim*np.log(self.horizon / self.failure_level)) * (self.epsilon **0.5* self.kappa)
        
        self.active_arms = arms
        self.unscaled_G_design = G_Optimal_Design(self.active_arms , self.dim)
        self.scaled_G_design = None
        self.upper_bound_unscaled = [self.unscaled_G_design[i] * self.batch_lengths[0] for i in range(len(self.unscaled_G_design))]
        self.upper_bound_scaled = None

        self.reward_arr = []
        self.arms_arr = []
        self.arm_idx_being_played = 0
        self.current_arm_pulls = 0
        
        self.theta = np.random.rand(self.dim)
        self.theta = self.theta / np.linalg.norm(self.theta) * self.param_norm_ub
        self.H = np.zeros((self.dim , self.dim))

        # to ensure switch from scaled to unscaled design
        self.switch = False

    def calculate_batch_endpoints(self):
        total_batches = int(np.ceil(np.log(np.log(self.horizon)))) + 1
        print(f"Total number of batches are {total_batches}")
        batch_lengths = [(self.horizon ** (1 - 2**(-i))) for i in range(1 , total_batches+1)]
        batch_lengths[-1] = self.horizon
        batch_endpoints = np.cumsum(batch_lengths)
        batch_endpoints = np.clip(batch_endpoints , 0 , self.horizon)
        batch_endpoints = batch_endpoints.astype(int)
        assert batch_endpoints[-1] == self.horizon
        return batch_lengths , batch_endpoints

    def update_params(self , time_round , reward):
        self.reward_arr.append(reward)
        if time_round in self.batch_endpoints:
            # update theta
            thth, succ_flag = solve_mle(self.theta, np.array(self.arms_arr), np.array(self.reward_arr) , self.lmbda)
            if succ_flag:
                self.theta = thth 

            # update H
            self.H = self.lmbda * np.eye(self.dim)
            for arm in self.arms_arr:
                self.H += dsigmoid(np.dot(self.theta , arm)) * np.outer(arm , arm)

            # update active arm_set
            prev_arm_set = self.active_arms.copy()
            maximum_LCB = -np.inf
            for arm in prev_arm_set:
                arm_score = np.dot(self.theta , arm) - self.gamma * np.sqrt(np.dot(arm , np.linalg.inv(self.H) @ arm))
                maximum_LCB = max(maximum_LCB , arm_score)
            self.active_arms = [arm for arm in prev_arm_set if np.dot(self.theta , arm) + self.gamma * np.sqrt(np.dot(arm , np.linalg.inv(self.H) @ arm)) >= maximum_LCB]

            # reinitialize the arms, rewards, and batch parameters
            self.arms_arr = []
            self.reward_arr = []
            self.arm_idx_being_played = 0
            self.current_arm_pulls = 0
            self.current_batch += 1

            # learn the new optimal designs
            self.unscaled_G_design = G_Optimal_Design(self.active_arms , self.dim)
            self.scaled_G_design = G_Optimal_Design([dsigmoid(np.dot(self.theta , arm)) * arm for arm in self.active_arms] , self.dim)
            self.switch = False
            self.upper_bound_unscaled = [np.ceil(self.unscaled_G_design[i] * self.batch_lengths[self.current_batch - 1]/2) for i in range(len(self.unscaled_G_design))]
            self.upper_bound_scaled = [np.ceil(self.scaled_G_design[i] * self.batch_lengths[self.current_batch - 1]/2) for i in range(len(self.scaled_G_design))]    
            print(f"Starting Batch {self.current_batch}")

    def play(self , time_round):
        if self.current_batch == 1:
            upper_bound_on_arm_being_played =  self.upper_bound_unscaled[self.arm_idx_being_played]
            if self.current_arm_pulls <= upper_bound_on_arm_being_played:
                self.current_arm_pulls += 1
                self.arms_arr.append(self.active_arms[self.arm_idx_being_played])
                return self.active_arms[self.arm_idx_being_played]
            else:
                self.arm_idx_being_played += 1
                self.current_arm_pulls = 1
                self.arms_arr.append(self.active_arms[self.arm_idx_being_played])
                return self.active_arms[self.arm_idx_being_played]
            
        else:
            midpoint = (self.batch_lengths[self.current_batch-1]//2 + self.batch_endpoints[self.current_batch-2])
            # first half play with scaled optimal design
            if time_round <= midpoint:
                upper_bound_on_arm_being_played =  self.upper_bound_scaled[self.arm_idx_being_played]
                if self.current_arm_pulls <= upper_bound_on_arm_being_played:
                    self.current_arm_pulls += 1
                    self.arms_arr.append(self.active_arms[self.arm_idx_being_played])
                    return self.active_arms[self.arm_idx_being_played]
                else:
                    self.arm_idx_being_played += 1
                    self.current_arm_pulls = 1
                    self.arms_arr.append(self.active_arms[self.arm_idx_being_played])
                    return self.active_arms[self.arm_idx_being_played]
            else:
                if not self.switch:
                    self.switch = True
                    self.arm_idx_being_played = 0
                    self.current_arm_pulls = 1
                    self.arms_arr.append(self.active_arms[self.arm_idx_being_played])
                    return self.active_arms[self.arm_idx_being_played]
                
                upper_bound_on_arm_being_played =  self.upper_bound_unscaled[self.arm_idx_being_played]
                if self.current_arm_pulls <= upper_bound_on_arm_being_played:
                    self.current_arm_pulls += 1
                    self.arms_arr.append(self.active_arms[self.arm_idx_being_played])
                    return self.active_arms[self.arm_idx_being_played]
                else:
                    self.arm_idx_being_played += 1
                    self.current_arm_pulls = 1
                    self.arms_arr.append(self.active_arms[self.arm_idx_being_played])
                    return self.active_arms[self.arm_idx_being_played]