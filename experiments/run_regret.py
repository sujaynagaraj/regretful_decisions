import os
import time
import sys
import pickle

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0,'..')

import argparse
from random import SystemRandom


from src.models import *
from src.loss_functions import *
from src.noise import *
from src.metrics import *
from src.plotting import *
from src.generate_data import *
from src.helper import *
from src.real_data import *

from operator import xor

import pickle as pkl
import timeit

parser = argparse.ArgumentParser('regret')  

parser.add_argument('--n_models', type =int, default=100, help="number of models to train")
parser.add_argument('--n_draws', type =int, default=5, help="number of noise draws")
parser.add_argument('--noise_type', type=str, default="class_independent", help="specify type of label noise")
parser.add_argument('--noise_level', type =float, default=0.2, help="noise level")
parser.add_argument('--max_iter', type =int, default=10000, help="max iterations to check for typical vec")
parser.add_argument('--dataset_size', type =int, default=5000, help="max iterations to check for typical vec")
parser.add_argument('--model_type', type =str, default="LR", help="LR or NN")
parser.add_argument('--dataset', type =str, default="cshock_mimic", help="dataset choice")
parser.add_argument('--epsilon', type =float, default=0.1, help="number of models to train")

# Add a boolean argument that defaults to False, but sets to True when specified
parser.add_argument('--misspecify', type=str, default = "correct" ,help="over or under-estimate T")


args = parser.parse_args()

#####################################################################################################

if __name__ == '__main__':
    
    start_time = timeit.default_timer()

    print('Starting Regret')
    print("Noise Type: ", args.noise_type)
    print("Noise Level: ", args.noise_level)
    print("Model Type: ", args.model_type)
    print("Dataset: ", args.dataset)
    print("Models: ", args.n_models)
    print("N Draws: ", args.n_draws)

    n_models = args.n_models
    n_draws = args.n_draws
    max_iter = args.max_iter
    noise_type = args.noise_type
    noise_level = args.noise_level
    model_type = args.model_type
    dataset_size = args.dataset_size
    dataset = args.dataset
    epsilon = args.epsilon
    misspecify = args.misspecify


    if dataset == "cshock_eicu":
        batch_size = 512
    elif dataset == "lungcancer":
        batch_size = 2048
    else:
        batch_size = 1024
    

    # Parent directory for saving figures
    #parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    parent_dir = "/scratch/hdd001/home/snagaraj/"
    files_path = os.path.join(parent_dir, "results", "regret", dataset, args.model_type, args.noise_type, args.misspecify)

    if not os.path.exists(files_path):
        os.makedirs(files_path)

    X_train, X_test, y_train, y_test, group_train, group_test = load_dataset_splits(dataset, group="age")


    if noise_type == "class_independent":
        _, T_true = generate_class_independent_noise(y_train, noise_level) #Fixed noise draw
        
        if misspecify == "over": #Estimate more noise than true T
            T_est = adjust_transition_matrix(T_true, 0.1)
        elif misspecify == "under": #Estimate less noise than true T
            assert(noise_level!=0)
            T_est = adjust_transition_matrix(T_true, -0.1)
        elif misspecify == "correct": #Correct T
            T_est = T_true
        else:
            print("invalid type")

        for training_loss in ["None", "backward"]:
            vectors = run_experiment(dataset=dataset, noise_type=noise_type, model_type=model_type, n_models=n_models, max_iter=max_iter, training_loss=training_loss, T=T_est, n_draws = n_draws, batch_size = batch_size)
        
            path = os.path.join(files_path, f"{training_loss}_{noise_level}_{epsilon}_vectors.pkl")

            # Open a file for writing in binary mode
            with open(path, 'wb') as file:
                # Use pickle to write the dictionary to the file
                pkl.dump(vectors, file)
            print(timeit.default_timer() - start_time)

    elif noise_type == "class_conditional":
        fixed_classes = [0]
        fixed_noises = [0.0]

        for fixed_class in fixed_classes:
            for i, fixed_noise in enumerate(fixed_noises):
                
                _, T_true = generate_class_conditional_noise(y_train, noise_level, fixed_class, fixed_noise)


                if misspecify == "correct": #Estimate more noise than true T
                    T_est = T_true
                elif misspecify == "flipped" : 
                    T_est = np.array([[T[1, 1], T[1, 0]], 
                                    [T[0, 1], T[0, 0]]]) 
                else:
                    print("invalid type")

                for training_loss in ["None", "backward"]:
                    print(training_loss)
                    vectors = run_experiment(dataset=dataset, noise_type=noise_type, model_type=model_type, n_models=n_models, max_iter=max_iter, training_loss=training_loss, T=T_est, n_draws = n_draws, batch_size = batch_size)
        
                    path = os.path.join(files_path, f"{training_loss}_{noise_level}_{fixed_class}_{fixed_noise}_{epsilon}_vectors.pkl")

                    # Open a file for writing in binary mode
                    with open(path, 'wb') as file:
                        # Use pickle to write the dictionary to the file
                        pkl.dump(vectors, file)
                    print(timeit.default_timer() - start_time)

    print("DONE")