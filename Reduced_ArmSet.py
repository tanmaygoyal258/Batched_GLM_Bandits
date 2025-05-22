import numpy as np
from utils import *

class Reduced_ArmSet:

    def __init__(self , theta_set , theta_rng):
        self.theta_set = theta_set
        self.arm_set = [theta * theta_rng.uniform() * 2 - 1 for theta in self.theta_set]
        self.dict_arm_to_theta = {tuple(arm) : tuple(theta) for arm , theta in zip(self.arm_set , self.theta_set)}
        self.dict_theta_to_arm = {tuple(theta) : tuple(arm) for arm , theta in zip(self.arm_set , self.theta_set)}
        self.previous_arm_sets = None

    def update_sets(self , theta_set , arm_sets):
        self.theta_set = theta_set
        self.arm_set = []
        self.previous_arm_sets = arm_sets
        self.dict_arm_to_theta = {}
        self.dict_theta_to_arm = {}
        # self.arm_set = self.calculate_fixed_arms(arm_sets)
        return

    def calculate_fixed_arm(self , theta):
        if tuple(theta) in self.dict_theta_to_arm.keys():
            # print("HIT")
            return np.array(self.dict_theta_to_arm[tuple(theta)])
        
        identity = np.eye(len(self.theta_set[0]))
        fixed_arm = [0 for _ in range(len(theta))]
        for arm_set in self.previous_arm_sets:
            inner_products = [np.dot(arm , theta) for arm in arm_set]
            best_arm_idx = np.argmax(inner_products)
            best_arm = arm_set[best_arm_idx]
            for i , e in enumerate(identity):
                fixed_arm[i] += sigmoid(np.dot(best_arm , e)) 
        fixed_arm = [x / len(self.previous_arm_sets) for x in fixed_arm]
        fixed_arm = [sigmoid_inv(x) for x in fixed_arm]
        fixed_arm = np.array(fixed_arm) / np.linalg.norm(fixed_arm)
        
        # print("Updating length from ", len(self.dict_arm_to_theta))
        self.dict_theta_to_arm[tuple(theta)] = tuple(fixed_arm)
        self.dict_arm_to_theta[tuple(fixed_arm)] = tuple(theta)
        # print("Updating length to ", len(self.dict_arm_to_theta))
        return fixed_arm

