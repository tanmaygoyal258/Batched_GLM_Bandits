import numpy as np
from utils import *

class Reduced_ArmSet:
    """
    calculates the reduced arm set (a fixed arm for each parameter)
    """

    def __init__(self):
        self.default = True     # returns zeros for all arms

    def update_sets(self , arm_sets):
        """
        updates the set of {previous_arm_sets} and reinitializes the cached dictionaries
        """
        self.default = False
        self.previous_arm_sets = arm_sets
        self.dict_arm_to_theta = {}
        self.dict_theta_to_arm = {}
        return

    def calculate_fixed_arm(self , theta):
        if self.default == True:
            return np.zeros(len(theta))
        else:
            # checks the cached dictionary if the arm already exists
            if tuple(theta) in self.dict_theta_to_arm.keys():
                return np.array(self.dict_theta_to_arm[tuple(theta)])
            
            # calculates the arm by calculating each coordinate
            identity = np.eye(len(theta))
            fixed_arm = [0 for _ in range(len(theta))]
            for arm_set in self.previous_arm_sets:
                inner_products = [np.dot(arm , theta) for arm in arm_set]
                best_arm_idx = np.argmax(inner_products)
                best_arm = arm_set[best_arm_idx]
                for i , e in enumerate(identity):
                    fixed_arm[i] += sigmoid(np.dot(best_arm , e))             
            fixed_arm = [x / len(self.previous_arm_sets) for x in fixed_arm]
            fixed_arm = [sigmoid_inv(x) for x in fixed_arm]

            # stores the calculated arm in cache
            self.dict_theta_to_arm[tuple(theta)] = tuple(fixed_arm)
            self.dict_arm_to_theta[tuple(fixed_arm)] = tuple(theta)
            return fixed_arm

