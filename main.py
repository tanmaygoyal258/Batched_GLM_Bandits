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
    parser.add_argument('--contextual' , action = "store_true")
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
    params["param_norm_ub"] = args.param_norm_ub
    params["contextual"] = args.contextual

    # Store config as a JSON file
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%H-%M-%S")
    print(timestamp)

    # creating the optimal parameter
    theta_rng = np.random.default_rng(params["theta_seed"])
    params["theta_star"] = np.array([theta_rng.uniform()*2 - 1 for i in range(params['dimension'])])
    params["theta_star"] = params["theta_star"] / np.linalg.norm(params["theta_star"]) * params["param_norm_ub"]
    params["theta_star"] = params["theta_star"].tolist()
    
    nc_epsilon = np.sqrt(params["dimension"]/params["horizon"])
    if params["alg_name"] == "NC_Logistic":
        env = Non_Contextual_Logistic(params["theta_star"] , params , nc_epsilon)
        env.run_algorithm()
        regret_arr = env.regret_arr
        batch_endpoints = env.batch_endpoints
    elif params["alg_name"] == "ada_ofu_ecolog":
        env = GLMEnv(params, params["theta_star"] , nc_epsilon)
        env.play_algorithm()
        regret_arr = env.regret_arr
        time_arr = env.time_arr
    elif params["alg_name"] == "RS_GLinCB":
        env = GLMEnv(params, params["theta_star"] , nc_epsilon)
        env.play_algorithm()
        regret_arr = env.regret_arr
        time_arr = env.time_arr
    elif params["contextual"]:
        env = Contextual_Logistic(params["theta_star"] , params)
        params["alg_name"] = "Reduction_Alg"
        env.run_algorithm()
        regret_arr = env.regret_arr
        batch_endpoints = env.batch_endpoints

    if not params["contextual"]:
        path = "NC_Data_Files_{}/".format(params["alg_name"]) + timestamp
    else:
        path = "C_Data_Files_{}/".format(params["alg_name"]) + timestamp
    if not os.path.exists(path):
        os.makedirs(path)
    params["path"] = path
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

if __name__ == "__main__":
    main()
