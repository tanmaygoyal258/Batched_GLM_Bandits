import numpy as np
from LW_ArgMax_Reduction import LW_ArgMax_Reduction
from multiprocessing import Process , Queue
from utils import *

def compute_a_plus(q, horizon , reduced_armset , scaling_factor, estimate_theta, eta, best_arm, theta):
    """
    A proxy process to receive the queue and required arguments to calculate a_plus
    """
    a_plus , thetas_plus , arms_plus = LW_ArgMax_Reduction(horizon , reduced_armset , scaling_factor , estimate_theta / np.linalg.norm(estimate_theta), eta, best_arm, theta)
    q.put(("a_plus", (a_plus , thetas_plus , arms_plus)))  # Store result in the queue

def compute_a_minus(q, horizon , reduced_armset , scaling_factor , estimate_theta, eta, best_arm, theta):
    """
    A proxy process to receive the queue and required arguments to calculate a_plus
    """
    a_minus , thetas_minus , arms_minus = LW_ArgMax_Reduction(horizon , reduced_armset , scaling_factor , -estimate_theta / np.linalg.norm(estimate_theta), eta, best_arm, theta)
    q.put(("a_minus", (a_minus , thetas_minus , arms_minus)))  # Store result in the queue



class LWS_Reduction:
    def __init__(self , params ,fixed_armset):
        self.params= params
        self.reduced_armset = fixed_armset
        self.dict_arms_thetas = {}
        self.dict_thetas_arms = {}

    
    def LWS(self , scaling_factor ,eta , best_arm , theta , warmup_flag = False):
        """
        returns a C/alpha-approximate barycentric spanner assuming LW-ArgMax
        returns a alpha-approimate solution with multiplicative error
        """
        # sets the initial spanner to standard basis vectors
        barycentric_spanner = [arm for arm in np.identity(self.params["dimension"])]
        A = np.identity(self.params["dimension"])
        C = np.exp(1)

        for i in range(self.params["dimension"]):
            e_vec = [0 for _ in range(self.params["dimension"])]
            e_vec[i] = 1

            # in the unlikely case det(A) = 0
            try:
                estimate_theta = np.linalg.det(A) * (np.linalg.inv(A).T @ e_vec)
            except:
                A = A + 1e-5 * np.identity(self.params["dimension"])
                estimate_theta = np.linalg.det(A) * (np.linalg.inv(A).T @ e_vec)
            if np.linalg.norm(estimate_theta)== 0.0:
                print("CORRECTING ESTIMATE THETA")
                estimate_theta += np.finfo(np.float32).eps

            def weigh_the_arm(arm):
                return (scaling_factor * np.array(arm)) / (1 + eta * scaling_factor * (np.dot(best_arm , estimate_theta) - np.dot(arm , estimate_theta)))
                
            # Create a queue
            q = Queue()
            # Create processes
            process_a_plus = Process(target=compute_a_plus, args=(q, self.params["horizon"],  self.reduced_armset , scaling_factor , estimate_theta, eta, best_arm, theta ))
            process_a_minus = Process(target=compute_a_minus, args=(q, self.params["horizon"],  self.reduced_armset , scaling_factor , estimate_theta, eta, best_arm, theta ))
            # Start processes
            process_a_plus.start()
            process_a_minus.start()
            # Wait for both processes to complete
            process_a_plus.join()
            process_a_minus.join()  
            # Collect results
            results = {}
            while not q.empty():
                key, value = q.get()
                results[key] = value
            # Access results
            a_plus_results = results.get("a_plus")
            a_minus_results = results.get("a_minus")
            a_plus = a_plus_results[0]
            a_minus = a_minus_results[0]
            # Store the thetas and arms in cached dictionaries
            for t , a in zip(a_plus_results[1] , a_plus_results[2]):
                self.dict_arms_thetas[tuple(a)] = tuple(t)
                self.dict_thetas_arms[tuple(t)] = tuple(a)
            for t , a in zip(a_minus_results[1] , a_minus_results[2]):
                self.dict_arms_thetas[tuple(a)] = tuple(t)
                self.dict_thetas_arms[tuple(t)] = tuple(a)

            # print("DEBUG: A_plus" , a_plus)
            # print("DEBUG: A_minus" , a_minus)

            if abs(np.dot(weigh_the_arm(a_plus) , estimate_theta)) > abs(np.dot(weigh_the_arm(a_minus) , estimate_theta)):
                A[: , i] = weigh_the_arm(a_plus)
                barycentric_spanner[i] = a_plus
            else:
                A[: , i] = weigh_the_arm(a_minus)
                barycentric_spanner[i] = a_minus 

        replacement = True
        while replacement:
            replacement = False
            for i in range(self.params["dimension"]):
                e_vec = [0 for _ in range(self.params["dimension"])]
                e_vec[i] = 1

                # in the unlikely case det(A) = 0
                try:
                    estimate_theta = np.linalg.det(A) * (np.linalg.inv(A).T @ e_vec)
                except:
                    A = A + 1e-5 * np.identity(self.params["dimension"])
                    estimate_theta = np.linalg.det(A) * (np.linalg.inv(A).T @ e_vec)
                if np.linalg.norm(estimate_theta)== 0.0:
                    print("CORRECTING ESTIMATE THETA")
                    estimate_theta += np.finfo(np.float32).eps

                # Create a queue
                q = Queue()
                # Create processes
                process_a_plus = Process(target=compute_a_plus, args=(q, self.params["horizon"],  self.reduced_armset , scaling_factor ,  estimate_theta, eta, best_arm, theta))
                process_a_minus = Process(target=compute_a_minus, args=(q, self.params["horizon"],  self.reduced_armset , scaling_factor ,estimate_theta, eta, best_arm, theta))
                # Start processes
                process_a_plus.start()
                process_a_minus.start()
                # Wait for both processes to complete
                process_a_plus.join()
                process_a_minus.join()  
                # Collect results
                results = {}
                while not q.empty():
                    key, value = q.get()
                    results[key] = value
                # Access results
                a_plus_results = results.get("a_plus")
                a_minus_results = results.get("a_minus")
                a_plus = a_plus_results[0]
                a_minus = a_minus_results[0]
                # Store the thetas and arms in cached dictionaries
                for t , a in zip(a_plus_results[1] , a_plus_results[2]):
                    self.dict_arms_thetas[tuple(a)] = tuple(t)
                    self.dict_thetas_arms[tuple(t)] = tuple(a)
                for t , a in zip(a_minus_results[1] , a_minus_results[2]):
                    self.dict_arms_thetas[tuple(a)] = tuple(t)
                    self.dict_thetas_arms[tuple(t)] = tuple(a)

                # print("DEBUG: A_plus" , a_plus)
                # print("DEUBG: A_minus" , a_minus)
                if abs(np.dot(weigh_the_arm(a_plus) , estimate_theta)) > abs(np.dot(weigh_the_arm(a_minus) , estimate_theta)):
                    a = a_plus
                else:
                    a = a_minus

                A_temp = A.copy()
                A_temp[: , i] = weigh_the_arm(a)
                if abs(np.linalg.det(A_temp)) >= C * abs(np.linalg.det(A)):
                    # print("Found a better candidate. Restarting Loop")
                    barycentric_spanner[i] = a
                    A[: , i] = weigh_the_arm(a)
                    replacement = True
                    break
        return barycentric_spanner , self.dict_arms_thetas , self.dict_thetas_arms
