import numpy as np
from tqdm import tqdm
from utils import sigmoid , dsigmoid , sigmoid_inv , G_Optimal_Design , solve_mle

class Reduced_ArmSet:

    def __init__(self , theta_set , theta_rng):
        self.theta_set = theta_set
        self.arm_set = [theta * theta_rng.uniform() * 2 - 1 for theta in self.theta_set]

    def update_sets(self , theta_set , arm_sets):
        self.theta_set = theta_set
        self.arm_set = self.calculate_fixed_arms(arm_sets)
        return
    
    def calculate_fixed_arms(self , arm_sets):
        fixed_armset = []
        identity = np.eye(len(self.theta_set[0]))
        print("Updating the fixed arm sets")
        for theta in tqdm(self.theta_set):
            fixed_arm = [0 for _ in range(len(theta))]
            for arm_set in arm_sets:
                inner_products = [np.dot(arm , theta) for arm in arm_set]
                best_arm_idx = np.argmax(inner_products)
                best_arm = arm_set[best_arm_idx]
                for i , e in enumerate(identity):
                    fixed_arm[i] += sigmoid(np.dot(best_arm , e)) 
            fixed_arm = [x / len(arm_sets) for x in fixed_arm]
            fixed_arm = [sigmoid_inv(x) for x in fixed_arm]
            fixed_arm = np.array(fixed_arm) / np.linalg.norm(fixed_arm)
            fixed_armset.append(fixed_arm)
        return fixed_armset


class Reduction_Alg:

    def __init__(self , params , kappa):
        self.param_norm_ub = params["param_norm_ub"]
        self.horizon = params["horizon"]
        self.failure_level = params["failure_level"]  
        self.dim = params["dimension"]
        self.epsilon = np.sqrt(self.dim / self.horizon**0.25)

        self.batch_lengths , self.batch_endpoints = self.calculate_batch_endpoints()
        self.current_batch = 1
        self.lmbda = 0.5 
        self.kappa = kappa
        print(f"The value of kappa is {self.kappa}")
        self.gamma = self.param_norm_ub* np.sqrt(self.dim*np.log(self.horizon / self.failure_level)) * (self.epsilon **0.5* self.kappa)
        
        self.reward_arr = []
        self.arms_arr = []
        self.arm_idx_being_played = 0
        self.current_arm_pulls = 0
        
        self.theta_rng = np.random.default_rng(params["theta_seed"])
        self.discretized_theta = self.create_discretized_theta()
        self.active_theta = self.create_eps_net()

        self.theta = np.random.rand(self.dim)
        self.theta = self.theta / np.linalg.norm(self.theta) * self.param_norm_ub
        self.H = np.zeros((self.dim , self.dim))
        
        self.fixed_armset = Reduced_ArmSet(self.active_theta , self.theta_rng)
        self.unscaled_G_design = G_Optimal_Design(self.fixed_armset.arm_set , self.dim)
        self.scaled_G_design = None
        self.upper_bound_unscaled = [self.unscaled_G_design[i] * self.batch_lengths[0] for i in range(len(self.unscaled_G_design))]
        self.upper_bound_scaled = None
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


    def create_discretized_theta(self):
        print(f"Constructing a Discretized Set")
        discrete_theta = []
        for _ in tqdm(range(self.horizon)):
            theta = np.array([self.theta_rng.uniform()*2 - 1 for i in range(self.dim)])
            theta = theta / np.linalg.norm(theta) * self.param_norm_ub
            discrete_theta.append(theta)
        return discrete_theta
    
    def calculate_eps(self):
        factor = 2 * self.param_norm_ub * np.log(self.horizon) * np.log(np.log(self.horizon))
        init_eps , eps = factor , 1
        while init_eps != eps:
            init_eps = eps
            while eps <= 2 * self.param_norm_ub * self.horizon:
                eps *= factor
            eps /= factor
            factor  = factor ** 0.5
        return eps / self.horizon
    
    def create_eps_net(self):
        print(f"Creating Epsilon Net over discretized theta")
        eps_net = []    
        eps = self.calculate_eps()
        for theta1 in tqdm(self.discretized_theta):
            close_point = False
            for theta2 in eps_net:
                if np.linalg.norm(theta1 - theta2) <= eps:
                    close_point = True
                    break
            if not close_point:
                eps_net.append(theta1)
        print(f"Created an Epsilon (= {eps}) Net with length {len(eps_net)}")
        return eps_net
    
    def play(self , time_round):
        if self.current_batch == 1:
            upper_bound_on_arm_being_played =  self.upper_bound_unscaled[self.arm_idx_being_played]
            if self.current_arm_pulls <= upper_bound_on_arm_being_played:
                self.current_arm_pulls += 1
                self.arms_arr.append(self.fixed_armset.arm_set[self.arm_idx_being_played])
                return self.fixed_armset.theta_set[self.arm_idx_being_played]
            else:
                self.arm_idx_being_played += 1
                self.current_arm_pulls = 1
                self.arms_arr.append(self.fixed_armset.arm_set[self.arm_idx_being_played])
                return self.fixed_armset.theta_set[self.arm_idx_being_played]
            
        else:
            midpoint = (self.batch_lengths[self.current_batch-1]//2 + self.batch_endpoints[self.current_batch-2])
            # first half play with scaled optimal design
            if time_round <= midpoint:
                upper_bound_on_arm_being_played =  self.upper_bound_scaled[self.arm_idx_being_played]
                if self.current_arm_pulls <= upper_bound_on_arm_being_played:
                    self.current_arm_pulls += 1
                    self.arms_arr.append(self.fixed_armset.arm_set[self.arm_idx_being_played])
                    return self.fixed_armset.theta_set[self.arm_idx_being_played]
                else:
                    self.arm_idx_being_played += 1
                    self.current_arm_pulls = 1
                    self.arms_arr.append(self.fixed_armset.arm_set[self.arm_idx_being_played])
                    return self.fixed_armset.theta_set[self.arm_idx_being_played]
            else:
                if not self.switch:
                    self.switch = True
                    self.arm_idx_being_played = 0
                    self.current_arm_pulls = 1
                    self.arms_arr.append(self.fixed_armset.arm_set[self.arm_idx_being_played])
                    return self.fixed_armset.theta_set[self.arm_idx_being_played]
                
                upper_bound_on_arm_being_played =  self.upper_bound_unscaled[self.arm_idx_being_played]
                if self.current_arm_pulls <= upper_bound_on_arm_being_played:
                    self.current_arm_pulls += 1
                    self.arms_arr.append(self.fixed_armset.arm_set[self.arm_idx_being_played])
                    return self.fixed_armset.theta_set[self.arm_idx_being_played]
                else:
                    self.arm_idx_being_played += 1
                    self.current_arm_pulls = 1
                    self.arms_arr.append(self.fixed_armset.arm_set[self.arm_idx_being_played])
                    return self.fixed_armset.theta_set[self.arm_idx_being_played]
                
    
    def update_params(self , time_round , reward , previous_arm_sets):
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

            # update active theta_set
            prev_arm_set = self.fixed_armset.arm_set.copy()
            maximum_LCB = -np.inf
            for arm in prev_arm_set:
                arm_score = np.dot(self.theta , arm) - self.gamma * np.sqrt(np.dot(arm , np.linalg.inv(self.H) @ arm))
                maximum_LCB = max(maximum_LCB , arm_score)
            updated_theta_set = [theta for theta , arm in zip(self.fixed_armset.theta_set , self.fixed_armset.arm_set) \
                                 if np.dot(self.theta , arm) + self.gamma * np.sqrt(np.dot(arm , np.linalg.inv(self.H) @ arm)) >= maximum_LCB]
            self.fixed_armset.update_sets(updated_theta_set , previous_arm_sets)

            # update epislon and confidence radius
            self.epsilon = np.sqrt(self.dim / self.batch_lengths[self.current_batch - 1])
            self.gamma = 0.25 * self.param_norm_ub* np.sqrt(self.dim*np.log(self.horizon / self.failure_level)) * (self.epsilon **0.5* self.kappa)

            # reinitialize the arms, rewards, and batch parameters
            self.arms_arr = []
            self.reward_arr = []
            self.arm_idx_being_played = 0
            self.current_arm_pulls = 0
            self.current_batch += 1

            # learn the new optimal designs
            self.unscaled_G_design = G_Optimal_Design(self.fixed_armset.arm_set , self.dim)
            self.scaled_G_design = G_Optimal_Design([dsigmoid(np.dot(self.theta , arm)) * arm for arm in self.fixed_armset.arm_set] , self.dim)
            self.switch = False
            self.upper_bound_unscaled = [np.ceil(self.unscaled_G_design[i] * self.batch_lengths[self.current_batch - 1]/2) for i in range(len(self.unscaled_G_design))]
            self.upper_bound_scaled = [np.ceil(self.scaled_G_design[i] * self.batch_lengths[self.current_batch - 1]/2) for i in range(len(self.scaled_G_design))]    
            print(f"Starting Batch {self.current_batch}")
            return True
        
        return False