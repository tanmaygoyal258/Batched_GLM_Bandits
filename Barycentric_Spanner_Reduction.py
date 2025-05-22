import numpy as np
from LW_ArgMax_Reduction import LW_ArgMax_Reduction
from multiprocessing import Process , Queue
from utils import *
from Reduced_ArmSet import Reduced_ArmSet

def compute_a_plus(q, horizon, theta_set , reduced_armset , scaling_factor, estimate_theta, eta, best_arm, theta , seed):
    a_plus , thetas_plus , arms_plus = LW_ArgMax_Reduction(horizon, theta_set , reduced_armset , scaling_factor , estimate_theta / np.linalg.norm(estimate_theta), eta, best_arm, theta , seed)
    # verify_LW_ArgMax_Reduction(horizon, estimate_theta / np.linalg.norm(estimate_theta), eta, best_arm, theta, a_plus)
    q.put(("a_plus", (a_plus , thetas_plus , arms_plus)))  # Store result in the queue

def compute_a_minus(q, horizon, theta_set , reduced_armset , scaling_factor , estimate_theta, eta, best_arm, theta , seed):
    a_minus , thetas_minus , arms_minus = LW_ArgMax_Reduction(horizon, theta_set , reduced_armset , scaling_factor , -estimate_theta / np.linalg.norm(estimate_theta), eta, best_arm, theta , seed)
    # verify_LW_ArgMax_Reduction(horizon, estimate_theta / np.linalg.norm(estimate_theta), eta, best_arm, theta, a_minus)
    q.put(("a_minus", (a_minus , thetas_minus , arms_minus)))  # Store result in the queue



class LWS_Reduction:
    def __init__(self , params , theta_set , previous_arm_sets ,fixed_armset):
        self.params= params
        self.theta_set = theta_set
        self.previous_arm_sets = previous_arm_sets
        self.reduced_armset = fixed_armset
        self.dict_arms_thetas = {}
        self.dict_thetas_arms = {}

    
    def LWS(self , scaling_factor ,eta , best_arm , theta , warmup_flag = False):
        """
        returns a C/alpha-approximate barycentric spanner assuming LW-ArgMax
        returns a alpha-approimate solution with multiplicative error
        """
        barycentric_spanner = [arm for arm in np.identity(self.params["dimension"])]

        A = np.identity(self.params["dimension"])
        C = np.exp(1)

        for i in range(self.params["dimension"]):
            e_vec = [0 for _ in range(self.params["dimension"])]
            e_vec[i] = 1
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
                
            # a_plus = LW_ArgMax_Reduction(self.params , arm_set , estimate_theta/np.linalg.norm(estimate_theta) , eta , best_arm , theta ,seed = ((i+1) ** (i+1) + i))
            # # verify_LW_ArgMax_Reduction(self.params , arm_set , estimate_theta/np.linalg.norm(estimate_theta) , eta , best_arm , theta , a_plus)
            # a_minus = LW_ArgMax_Reduction(self.params , arm_set , -estimate_theta/np.linalg.norm(estimate_theta) , eta , best_arm , theta , seed = ((i+1) ** (i+1) + 10 * (i+1)))
            # verify_LW_ArgMax_Reduction(self.params , arm_set , estimate_theta/np.linalg.norm(estimate_theta) , eta , best_arm , theta , a_plus)        

            # Create a queue
            q = Queue()
            # Create processes
            process_a_plus = Process(target=compute_a_plus, args=(q, self.params["horizon"], self.theta_set , self.reduced_armset , scaling_factor , estimate_theta, eta, best_arm, theta , ((i+1) ** (i+1) + i)))
            process_a_minus = Process(target=compute_a_minus, args=(q, self.params["horizon"], self.theta_set , self.reduced_armset , scaling_factor , estimate_theta, eta, best_arm, theta , ((i+1) ** (i+1) + 10 * (i+1))))
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
            for t , a in zip(a_plus_results[1] , a_plus_results[2]):
                self.dict_arms_thetas[tuple(a)] = tuple(t)
                self.dict_thetas_arms[tuple(t)] = tuple(a)
            for t , a in zip(a_minus_results[1] , a_minus_results[2]):
                self.dict_arms_thetas[tuple(a)] = tuple(t)
                self.dict_thetas_arms[tuple(t)] = tuple(a)

            print("A_plus" , a_plus)
            print("A_minus" , a_minus)

            if abs(np.dot(weigh_the_arm(a_plus) , estimate_theta)) > abs(np.dot(weigh_the_arm(a_minus) , estimate_theta)):
                A[: , i] = weigh_the_arm(a_plus)
                barycentric_spanner[i] = a_plus
            else:
                A[: , i] = weigh_the_arm(a_minus)
                barycentric_spanner[i] = a_minus 

        print(f"Loop 1 of LWS done")

        replacement = True
        while replacement:
            replacement = False
            for i in range(self.params["dimension"]):
                e_vec = [0 for _ in range(self.params["dimension"])]
                e_vec[i] = 1
                
                try:
                    estimate_theta = np.linalg.det(A) * (np.linalg.inv(A).T @ e_vec)
                except:
                    A = A + 1e-5 * np.identity(self.params["dimension"])
                    estimate_theta = np.linalg.det(A) * (np.linalg.inv(A).T @ e_vec)

                if np.linalg.norm(estimate_theta)== 0.0:
                    print("CORRECTING ESTIMATE THETA")
                    estimate_theta += np.finfo(np.float32).eps

                # a_plus = LW_ArgMax_Reduction(self.params , arm_set , estimate_theta/np.linalg.norm(estimate_theta) , eta , best_arm , theta , seed = ((i+2) ** (i+2) - i))
                # verify_LW_ArgMax_Reduction(self.params , arm_set , estimate_theta/np.linalg.norm(estimate_theta) , eta , best_arm , theta , a_plus)
                # a_minus = LW_ArgMax_Reduction(self.params , arm_set , -estimate_theta/np.linalg.norm(estimate_theta) , eta , best_arm , theta , seed = ((i+2) ** (i+2) + 9 * (i+1)**i + i))
                # verify_LW_ArgMax_Reduction(self.params , arm_set , estimate_theta/np.linalg.norm(estimate_theta) , eta , best_arm , theta , a_plus)        
                
                # Create a queue
                q = Queue()
                # Create processes
                process_a_plus = Process(target=compute_a_plus, args=(q, self.params["horizon"], self.theta_set , self.reduced_armset , scaling_factor ,  estimate_theta, eta, best_arm, theta , ((i+2) ** (i+2) - i)))
                process_a_minus = Process(target=compute_a_minus, args=(q, self.params["horizon"], self.theta_set , self.reduced_armset , scaling_factor ,estimate_theta, eta, best_arm, theta , ((i+2) ** (i+2) + 9 * (i+1)**i + i)))
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
                for t , a in zip(a_plus_results[1] , a_plus_results[2]):
                    self.dict_arms_thetas[tuple(a)] = tuple(t)
                    self.dict_thetas_arms[tuple(t)] = tuple(a)
                for t , a in zip(a_minus_results[1] , a_minus_results[2]):
                    self.dict_arms_thetas[tuple(a)] = tuple(t)
                    self.dict_thetas_arms[tuple(t)] = tuple(a)

                print("A_plus" , a_plus)
                print("A_minus" , a_minus)
                if abs(np.dot(weigh_the_arm(a_plus) , estimate_theta)) > abs(np.dot(weigh_the_arm(a_minus) , estimate_theta)):
                    a = a_plus
                else:
                    a = a_minus

                A_temp = A.copy()
                A_temp[: , i] = weigh_the_arm(a)
                # print("After replacement: ", abs(np.linalg.det(A_temp)))
                # print("Original: ", abs(np.linalg.det(A)))
                if abs(np.linalg.det(A_temp)) >= C * abs(np.linalg.det(A)):
                    print("Found a better candidate. Restarting Loop")
                    barycentric_spanner[i] = a
                    A[: , i] = weigh_the_arm(a)
                    replacement = True
                    break
            print("Loop 2 of LWS done")     

        return barycentric_spanner , self.dict_arms_thetas , self.dict_thetas_arms

# def verify_LW_ArgMax_Reduction(self.params , arm_set , estimate_theta , eta , best_arm , theta , test_arm):
#     def weigh_the_arm(arm):
#         if self.params["gap"].lower() == "linear":
#             gap = np.dot(best_arm , theta) - np.dot(arm , theta)
#         else:
#             gap = sigmoid(np.dot(best_arm , theta)) - sigmoid(np.dot(arm , theta))
#         return np.array(arm) / (1 + eta * gap)

#     phi_arms = [weigh_the_arm(a) for a in arm_set]
#     best_val = np.max([np.dot(a , estimate_theta) for a in phi_arms])
#     assert np.dot(weigh_the_arm(test_arm) , estimate_theta) >= np.exp(-3) * best_val , "LW_ArgMax_Reduction failed"