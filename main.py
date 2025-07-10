import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
import json

from C_Logistic_Env import Contextual_Logistic
from NC_Logistic_Env import Non_Contextual_Logistic
from GLMEnv import GLMEnv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg_name', type = str, help = 'algorithm name') 
    parser.add_argument('--horizon', type = int, default = '10000', help = 'time horizon')
    parser.add_argument('--failure_level', type = float, default = 0.05, help = 'delta')
    parser.add_argument('--dimension' , type = int , default = 5 , help = "dimension ")
    parser.add_argument('--number_arms' , type = int , default = 100 , help = 'number of arms')
    parser.add_argument('--reward_seed', type = int, default = 123, help = 'random seed for rewards')
    parser.add_argument('--arm_seed', type = int, default = 456, help = 'random seed for arms')
    parser.add_argument('--theta_seed', type = int, default = 789, help = 'random seed for generating theta star')
    parser.add_argument('--param_norm_ub', type = int, default = 1, help = 'desired norm for theta_star')
    parser.add_argument('--epsilon_seed', type = int, default = 1234, help = 'random seed for misspecification')
    parser.add_argument('--contextual' , action = "store_true")
    parser.add_argument('--lamda' , type = float, default = -1)
    parser.add_argument("--directory" , type = str , default = None)
    parser.add_argument("--epsilon" , type = float , default = -1)
    return parser.parse_args()


def main():
    # read the arguments
    args = parse_args()
    params = {}
    params["alg_name"] = args.alg_name
    params['horizon'] = args.horizon
    params['failure_level'] = args.failure_level
    params['dimension'] = args.dimension
    params['number_arms'] = args.number_arms
    params['reward_seed'] = args.reward_seed
    params['arm_seed'] = args.arm_seed
    params['theta_seed'] = args.theta_seed
    params["epsilon_seed"] = args.epsilon_seed
    params["param_norm_ub"] = args.param_norm_ub
    params["contextual"] = args.contextual
    params["lamda"] = args.lamda

    # Create the file name
    file_name = f"S={args.param_norm_ub}_T={args.horizon}_Num_Arms={args.number_arms}_Arm_seed={args.arm_seed}_Reward_seed={args.reward_seed}_eps={args.epsilon}"
    
    # creating the optimal parameter
    theta_rng = np.random.default_rng(params["theta_seed"])
    params["theta_star"] = np.array([theta_rng.uniform()*2 - 1 for i in range(params['dimension'])])
    params["theta_star"] = params["theta_star"] / np.linalg.norm(params["theta_star"]) * params["param_norm_ub"]
    params["theta_star"] = params["theta_star"].tolist()

    # Check for validity of folder    
    if args.directory is None:
        if not params["contextual"]:
            path = "Results/NC_Data_Files_{}/".format(params["alg_name"]) + file_name
        else:
            path = "Results/C_Data_Files_{}/".format(params["alg_name"]) + file_name
    else:
        path = args.directory
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + "/" + file_name
    params["path"] = path


    # set epislon for misclassification
    nc_epsilon = np.sqrt(params["dimension"]/params["horizon"]) if args.epsilon < 0 else args.epsilon
    
    # set the environment and run the algorithm
    if params["alg_name"] == "SoftBatch":
        if params["contextual"]:
            env = Contextual_Logistic(params["theta_star"] , params)
        else:
            env = Non_Contextual_Logistic(params["theta_star"] , params , nc_epsilon)
        env.run_algorithm()
        regret_arr = env.regret_arr
        batch_endpoints = env.batch_endpoints
        try:
            theta_norms = env.theta_norms
            best_arms = env.best_arm_gaps
        except:
            pass
    elif params["alg_name"] == "BatchGLinCB-Fixed":
        env = Non_Contextual_Logistic(params["theta_star"] , params , nc_epsilon)
        env.run_algorithm()
        regret_arr = env.regret_arr
        batch_endpoints = env.batch_endpoints
        theta_norms = env.theta_norms
        best_arms = env.best_arm_gaps
    elif params["alg_name"] == "BatchGLinCB":
        env = Contextual_Logistic(params["theta_star"] , params)
        env.run_algorithm()
        regret_arr = env.regret_arr
        batch_endpoints = env.batch_endpoints
    elif params["alg_name"] == "RS_GLinCB":
        env = GLMEnv(params, params["theta_star"] , nc_epsilon)
        env.play_algorithm()
        regret_arr = env.regret_arr
        time_arr = env.time_arr
    else:
        print("Incorrect Algorithm name")
        return

    # Dump the JSON and save the results retrieved
    if not os.path.exists(path):
            os.makedirs(path)
    with open(path+"/config.json", "w") as file:
            json.dump(params, file) 
    np.save(path + "/regret_array" , regret_arr)
    try:
        np.save(path + "/batch_endpoints" , batch_endpoints)
    except:
        pass
    try:
        np.save(path + "/time_arr" , time_arr)
    except:
        pass
    try:
        np.save(path + "/theta_norms" , theta_norms)
    except:
        pass
    try:
        np.save(path + "/best_arms" , best_arms)
    except:
        pass

    print(f"Reward Seed:{params['reward_seed']}: Kappa:{env.kappa} , Regret: {np.sum(regret_arr)}")

if __name__ == "__main__":
    main()
