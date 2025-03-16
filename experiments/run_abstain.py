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
from src.abstain import *
from src.loss_functions import *
from src.noise import *
from src.metrics import *
from src.plotting import *
from src.generate_data import *
from src.helper import *
from src.real_data import *

from operator import xor

import pickle as pkl

parser = argparse.ArgumentParser('abstain')

parser.add_argument('--n_models', type =int, default=100, help="number of models to train")
parser.add_argument('--noise_type', type=str, default="class_independent", help="specify type of label noise")
parser.add_argument('--max_iter', type =int, default=10000, help="max iterations to check for typical vec")
parser.add_argument('--model_type', type =str, default="LR", help="LR or NN")
parser.add_argument('--dataset', type =str, default="cshock_mimic", help="dataset choice")
parser.add_argument('--epsilon', type =float, default=0.1, help="number of models to train")

# Add a boolean argument that defaults to False, but sets to True when specified
parser.add_argument('--misspecify', type=str, default = "correct" ,help="over or under-estimate T")

args = parser.parse_args()

#####################################################################################################

if __name__ == '__main__':
    start_time = timeit.default_timer()

    print('Starting Abstention')
    print("Noise Type: ", args.noise_type)
    print("Model Type: ", args.model_type)
    print("Dataset: ", args.dataset)

    n_models = args.n_models
    max_iter = args.max_iter
    noise_type = args.noise_type
    model_type = args.model_type
    dataset = args.dataset
    epsilon = args.epsilon
    misspecify = args.misspecify

    # Parent directory for saving figures
    #parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    parent_dir = "/scratch/hdd001/home/snagaraj/"
    files_path = os.path.join(parent_dir, "results", "abstain", dataset, args.model_type, args.noise_type, args.misspecify)

    if not os.path.exists(files_path):
        os.makedirs(files_path)

    X_train, X_test, y_train, y_test, group_train, group_test = load_dataset_splits(dataset, group = "age")

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    p_y_x_dict =  calculate_prior(y_train, noise_type = noise_type, group=group_train) #Clean prior

    batch_size = 512

    if dataset == "cshock_eicu":
        batch_size = 512
    elif dataset == "lungcancer":
        batch_size = 2048
    else:
        batch_size = 1024


    if noise_type == "class_independent":
        noise_levels = []
        losses = []

        amb_train = []
        unanticipated_train= []
        unanticipated_retrain =[]
        actual_train = []
        probs_train = []


        amb_test = []
        probs_test = []
        draw_ids = []

        for noise_level in [0.05, 0.2,  0.4]:
            
            _, T_true = generate_class_independent_noise(y_train, noise_level) #Fixed noise draw
        
            if misspecify == "over": #Estimate more noise than true T
                T_est = adjust_transition_matrix(T_true, 0.1)
                misspecify_flag = True
            if misspecify == "under": #Estimate less noise than true T
                assert(noise_level!=0)
                T_est = adjust_transition_matrix(T_true, -0.1)
                misspecify_flag = True
            elif misspecify == "correct": #Correct T
                T_est = T_true
                misspecify_flag = False
            else:
                continue

            for seed in range(5):
                u_vec = get_u(y_train, T = T_true, seed= seed, noise_type = noise_type)
                yn_train = flip_labels(y_train, u_vec) #XOR

                (ambiguity_train, 
                ambiguity_test,
                unanticipated) = run_procedure(n_models, max_iter, X_train, yn_train, X_test, y_test, p_y_x_dict, group_train = group_train, group_test = group_test, noise_type = noise_type, model_type = model_type, T = T_est, epsilon = epsilon, misspecify = misspecify_flag)

                
                model, (train_acc,
                                test_acc,
                                train_probs,
                                test_probs,
                                train_loss,
                                test_loss,
                                train_preds,
                                test_preds) = train_model_ours_regret(X_train, yn_train, X_test, y_test, seed = 2024, model_type=model_type)
                
                actual_mistakes, unanticipated_mistakes = get_uncertainty(n_models, max_iter, train_preds, yn_train, p_y_x_dict,group_train = group_train, group_test = group_test, noise_type=noise_type, model_type=model_type, T=T_est, epsilon=epsilon, misspecify=misspecify_flag)
    
                noise_levels.append(noise_level)
                losses.append("BCE")

                amb_train.append(ambiguity_train)
                unanticipated_train.append(unanticipated_mistakes)
                unanticipated_retrain.append(unanticipated)
                actual_train.append(actual_mistakes)
                probs_train.append(train_probs)


                amb_test.append(ambiguity_test)
                probs_test.append(test_probs)
                draw_ids.append(seed)

                
                for loss in ["backward", "forward"]:
                    model, results = train_model(X_train, y_train, yn_train, X_test, y_test, T_est,  seed=2024, num_epochs=100, batch_size = 256, correction_type=loss, model_type = model_type)
                    
                    train_probs = results[4]
                    test_probs = results[7]

                    noise_levels.append(noise_level)
                    losses.append(loss)

                    amb_train.append(ambiguity_train)
                    unanticipated_train.append(unanticipated_mistakes)
                    unanticipated_retrain.append(unanticipated)
                    actual_train.append(actual_mistakes)
                    probs_train.append(train_probs)


                    amb_test.append(ambiguity_test)
                    probs_test.append(test_probs)
                    draw_ids.append(seed)

        data = {'noise': noise_levels, 'loss': losses, "actual_train": actual_train , "unanticipated_retrain": unanticipated_retrain , "unanticipated_train": unanticipated_train , "ambiguity_train":amb_train, "ambiguity_test":amb_test, "test_probs":probs_test, "train_probs":probs_train,  "draw_id":draw_ids}

        path = os.path.join(files_path, f"{epsilon}.pkl")

            # Open a file for writing in binary mode
        with open(path, 'wb') as file:
            # Use pickle to write the dictionary to the file
            pkl.dump(data, file)

        print(timeit.default_timer() - start_time)

    elif noise_type == "class_conditional":
        classes = [0]
        noises = [0.0]
        noise_levels, fixed_classes, fixed_noises, losses, draw_ids = [], [], [], [], []

        plausible_labels_train_all, plausible_labels_test_all  = [], []
        preds_train_all, preds_test_all =  [], []
        probs_train_all, probs_test_all = [], []

        final_model_probs_train_all, final_model_probs_test_all = [], []

        # Iterate over different noise conditions
        for fixed_class in classes:
            for fixed_noise in noises:
                for noise_level in [0.05, 0.2, 0.4]:
                    
                    # Generate noise transition matrix
                    _, T_train = generate_class_conditional_noise(y_train, noise_level, fixed_class, fixed_noise)
                    _, T_test = generate_class_conditional_noise(y_test, noise_level, fixed_class, fixed_noise)

                    # Determine misspecified noise matrix
                    if misspecify == "correct":
                        T_est = T_train
                        misspecify_flag = False
                    else:
                        T_est = np.array([[T_train[1, 1], T_train[1, 0]], 
                                         [T_train[0, 1], T_train[0, 0]]]) 
                        misspecify_flag = True

                    # Loop over multiple draws
                    for seed in range(5):
                        
                        # Step 1: Merge y_train and y_test
                        y_all = np.concatenate([y_train, y_test])  # Shape: (total_instances,)

                        # Step 2: Generate a single noise vector for all instances
                        u_all = get_u(y_all, T=T_train, seed=seed, noise_type=noise_type)  # Using T_train for now

                        # Step 3: Flip labels using the generated noise vector
                        yn_all = flip_labels(y_all, u_all)

                        # Step 4: Split back into train and test sets
                        u_train, u_test = u_all[:len(y_train)], u_all[len(y_train):]
                        yn_train, yn_test = yn_all[:len(y_train)], yn_all[len(y_train):]

                        # Run main procedure
                        plausible_labels_train, plausible_labels_test, preds_train, preds_test, probs_train, probs_test = run_procedure_abstain(
                            n_models, max_iter, X_train, yn_train, X_test, yn_test, y_test, p_y_x_dict, 
                            group_train=group_train, group_test=group_test, noise_type=noise_type, 
                            model_type=model_type, T=T_est, epsilon=epsilon, misspecify=misspecify_flag
                        )
                        
                        # Train final model on noisy labels
                        model, (train_acc, test_acc, train_probs, test_probs, 
                                train_loss, test_loss, train_preds, test_preds) = train_model_ours_regret(
                            X_train, yn_train, X_test, y_test, seed=2024, model_type=model_type
                        )

                        # Store results
                        noise_levels.append(noise_level)
                        fixed_classes.append(fixed_class)
                        fixed_noises.append(fixed_noise)
                        losses.append("BCE")
                        draw_ids.append(seed)
                        
                        plausible_labels_train_all.append(plausible_labels_train)
                        plausible_labels_test_all.append(plausible_labels_test)
                        preds_train_all.append(preds_train)
                        preds_test_all.append(preds_test)
                        probs_train_all.append(probs_train)
                        probs_test_all.append(probs_test)
                        
                        final_model_probs_train_all.append(train_probs)
                        final_model_probs_test_all.append(test_probs)

        # Convert results into a structured dictionary
        data = {
            "noise": noise_levels,
            "loss": losses,
            "plausible_labels_train": plausible_labels_train_all,
            "plausible_labels_test": plausible_labels_test_all,
            "probs_train": probs_train_all,
            "probs_test": probs_test_all,
            "preds_train": preds_train_all,
            "preds_test": preds_test_all,
            "final_model_probs_train": final_model_probs_train_all,
            "final_model_probs_test": final_model_probs_test_all,
            "fixed_noise": fixed_noises,
            "fixed_class": fixed_classes,
            "draw_id": draw_ids
        }

        pkl_path = os.path.join(files_path, f"{epsilon}.pkl")
        csv_path = os.path.join(files_path, "results.csv")

            # Open a file for writing in binary mode
        with open(pkl_path, 'wb') as file:
            # Use pickle to write the dictionary to the file
            pkl.dump(data, file)

        results_df = metrics_active_learning(dataset, noise_type, model_type, data)

        results_df.to_csv(csv_path, index=False)


        print(timeit.default_timer() - start_time)