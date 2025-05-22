import numpy as np
from LW_ArgMax import LW_ArgMax
from multiprocessing import Process , Queue
from utils import *

def compute_a_plus(q, params, arm_set, scaling_factor, estimate_theta, eta, best_arm, theta , seed):
    a_plus = LW_ArgMax(params, arm_set, scaling_factor , estimate_theta / np.linalg.norm(estimate_theta), eta, best_arm, theta , seed)
    # verify_LW_ArgMax(params, arm_set, estimate_theta / np.linalg.norm(estimate_theta), eta, best_arm, theta, a_plus)
    q.put(("a_plus", a_plus))  # Store result in the queue

def compute_a_minus(q, params, arm_set, scaling_factor , estimate_theta, eta, best_arm, theta , seed):
    a_minus = LW_ArgMax(params, arm_set, scaling_factor , -estimate_theta / np.linalg.norm(estimate_theta), eta, best_arm, theta , seed)
    # verify_LW_ArgMax(params, arm_set, estimate_theta / np.linalg.norm(estimate_theta), eta, best_arm, theta, a_minus)
    q.put(("a_minus", a_minus))  # Store result in the queue
    
def LWS(params , arm_set , scaling_factor ,eta , best_arm , theta , warmup_flag = False):
    """
    returns a C/alpha-approximate barycentric spanner assuming LW-ArgMax
    returns a alpha-approimate solution with multiplicative error
    """
    barycentric_spanner = [arm for arm in np.identity(params["dimension"])]

    A = np.identity(params["dimension"])
    C = np.exp(1)

    for i in range(params["dimension"]):
        e_vec = [0 for _ in range(params["dimension"])]
        e_vec[i] = 1
        try:
            estimate_theta = np.linalg.det(A) * (np.linalg.inv(A).T @ e_vec)
        except:
            A = A + 1e-5 * np.identity(params["dimension"])
            estimate_theta = np.linalg.det(A) * (np.linalg.inv(A).T @ e_vec)

        if np.linalg.norm(estimate_theta)== 0.0:
            print("CORRECTING ESTIMATE THETA")
            estimate_theta += np.finfo(np.float32).eps

        if warmup_flag:
                a_plus_idx = np.argmax([np.dot(arm , estimate_theta) for arm in arm_set])
                a_plus = arm_set[a_plus_idx]
                a_minus_idx = np.argmax([np.dot(arm , -estimate_theta) for arm in arm_set])
                a_minus = arm_set[a_minus_idx]
                if abs(np.dot(a_plus , estimate_theta)) > abs(np.dot(a_minus , estimate_theta)):
                    A[: , i] = a_plus
                    barycentric_spanner[i] = a_plus
                else:
                    A[: , i] = a_minus
                    barycentric_spanner[i] = a_minus
        
        else:
            def weigh_the_arm(arm):
                return (scaling_factor * np.array(arm)) / (1 + eta * scaling_factor * (np.dot(best_arm , estimate_theta) - np.dot(arm , estimate_theta)))
            
            # a_plus = LW_ArgMax(params , arm_set , estimate_theta/np.linalg.norm(estimate_theta) , eta , best_arm , theta ,seed = ((i+1) ** (i+1) + i))
            # # verify_LW_ArgMax(params , arm_set , estimate_theta/np.linalg.norm(estimate_theta) , eta , best_arm , theta , a_plus)
            # a_minus = LW_ArgMax(params , arm_set , -estimate_theta/np.linalg.norm(estimate_theta) , eta , best_arm , theta , seed = ((i+1) ** (i+1) + 10 * (i+1)))
            # verify_LW_ArgMax(params , arm_set , estimate_theta/np.linalg.norm(estimate_theta) , eta , best_arm , theta , a_plus)        

            # Create a queue
            q = Queue()
            # Create processes
            process_a_plus = Process(target=compute_a_plus, args=(q, params, arm_set, scaling_factor , estimate_theta, eta, best_arm, theta , ((i+1) ** (i+1) + i)))
            process_a_minus = Process(target=compute_a_minus, args=(q, params, arm_set, scaling_factor , estimate_theta, eta, best_arm, theta , ((i+1) ** (i+1) + 10 * (i+1))))
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
            a_plus = results.get("a_plus")
            a_minus = results.get("a_minus")

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
        for i in range(params["dimension"]):
            e_vec = [0 for _ in range(params["dimension"])]
            e_vec[i] = 1
            
            try:
                estimate_theta = np.linalg.det(A) * (np.linalg.inv(A).T @ e_vec)
            except:
                A = A + 1e-5 * np.identity(params["dimension"])
                estimate_theta = np.linalg.det(A) * (np.linalg.inv(A).T @ e_vec)

            if np.linalg.norm(estimate_theta)== 0.0:
                print("CORRECTING ESTIMATE THETA")
                estimate_theta += np.finfo(np.float32).eps

            if warmup_flag:
                a_plus_idx = np.argmax([np.dot(arm , estimate_theta) for arm in arm_set])
                a_plus = arm_set[a_plus_idx]
                a_minus_idx = np.argmax([np.dot(arm , -estimate_theta) for arm in arm_set])
                a_minus = arm_set[a_minus_idx]
                if abs(np.dot(a_plus , estimate_theta)) > abs(np.dot(a_minus , estimate_theta)):
                    A[: , i] = a_plus
                    barycentric_spanner[i] = a_plus
                else:
                    A[: , i] = a_minus
                    barycentric_spanner[i] = a_minus
            else:
                # a_plus = LW_ArgMax(params , arm_set , estimate_theta/np.linalg.norm(estimate_theta) , eta , best_arm , theta , seed = ((i+2) ** (i+2) - i))
                # verify_LW_ArgMax(params , arm_set , estimate_theta/np.linalg.norm(estimate_theta) , eta , best_arm , theta , a_plus)
                # a_minus = LW_ArgMax(params , arm_set , -estimate_theta/np.linalg.norm(estimate_theta) , eta , best_arm , theta , seed = ((i+2) ** (i+2) + 9 * (i+1)**i + i))
                # verify_LW_ArgMax(params , arm_set , estimate_theta/np.linalg.norm(estimate_theta) , eta , best_arm , theta , a_plus)        
                
                # Create a queue
                q = Queue()
                # Create processes
                process_a_plus = Process(target=compute_a_plus, args=(q, params, arm_set, scaling_factor ,estimate_theta, eta, best_arm, theta , ((i+2) ** (i+2) - i)))
                process_a_minus = Process(target=compute_a_minus, args=(q, params, arm_set, scaling_factor ,estimate_theta, eta, best_arm, theta , ((i+2) ** (i+2) + 9 * (i+1)**i + i)))
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
                a_plus = results.get("a_plus")
                a_minus = results.get("a_minus")
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
    return barycentric_spanner

# def verify_LW_ArgMax(params , arm_set , estimate_theta , eta , best_arm , theta , test_arm):
#     def weigh_the_arm(arm):
#         if params["gap"].lower() == "linear":
#             gap = np.dot(best_arm , theta) - np.dot(arm , theta)
#         else:
#             gap = sigmoid(np.dot(best_arm , theta)) - sigmoid(np.dot(arm , theta))
#         return np.array(arm) / (1 + eta * gap)

#     phi_arms = [weigh_the_arm(a) for a in arm_set]
#     best_val = np.max([np.dot(a , estimate_theta) for a in phi_arms])
#     assert np.dot(weigh_the_arm(test_arm) , estimate_theta) >= np.exp(-3) * best_val , "LW_ArgMax failed"