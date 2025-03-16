import sys
sys.path.insert(0,'..')

from src.models import *
from src.loss_functions import *
from src.noise import *
from src.metrics import *
from src.plotting import *
from src.generate_data import *
from src.abstain import *

import sklearn
import pandas as pd

from operator import xor
from collections import defaultdict

from scipy.stats import bernoulli

import random
import itertools


def logistic(x):
    return 1 / (1 + np.exp(-x))

def calculate_weights(prob_labels):
    # Create the system of equations
    A = np.array([
        [1, 0, 0],  # For p(0,0)
        [1, 0, 1],  # For p(0,1)
        [1, 1, 0],  # For p(1,0)
        [1, 1, 1]   # For p(1,1)
    ])
    
    # Calculate the logit for each probability
    b = np.array([np.log(p/(1-p)) if p not in [0, 1] else -np.inf if p == 0 else np.inf for p in prob_labels.values()])
    
    # Solve the system of equations
    weights = np.linalg.lstsq(A, b, rcond=None)[0]
    return weights

def generate_probabilistic_labels(features, weights, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)  # Set seed for reproducibility

    # Calculate probabilities using logistic function
    probs = logistic(np.dot(features, weights))
    
    # Sample labels based on probabilities
    labels = np.random.binomial(1, probs)
    return labels


def generate_dataset(true_labels, instances_counts, probabilistic = False, weights=None, seed=None, p_y_x_dict=None):
    """
    Generate a dataset with binary features
    
    :param true_labels: Dictionary with keys as (x1, x2) and values as the deterministic labels.
    :param instances_counts: Dictionary with keys as (x1, x2) and values as the number of instances.
    :param weights: Coefficients for the logistic model (bias, w1, w2).
    :param seed: Seed for the random number generator.
    :return: Shuffled features and probabilistic labels.
    """
    features, labels = [], []
    for (x1,x2,label) in true_labels:

        n_instances = instances_counts[(x1,x2,label)]
        #features.extend([(1, x1, x2)] * n_instances)  # Add a 1 for the bias term in logistic regression
        features.extend([(x1,x2)] * n_instances) 
        if not probabilistic:
            labels.extend([label] * n_instances)

    # Convert to numpy array for matrix operations
    features = np.array(features)
    
    if probabilistic:
    # Generate probabilistic labels
        #labels = generate_probabilistic_labels(features, weights, random_seed=seed)
        for i, X in enumerate(features):
            np.random.seed(i)
            p_y_x = p_y_x_dict[X]
            label =  bernoulli.rvs(p=p_y_x[1], size=1)[0]
            labels.append(label)

    # Shuffle the dataset
    #return shuffle(features[:, 1:], labels, random_state=seed)  # Exclude the bias term from the returned features
    return shuffle(features, np.array(labels), random_state=seed)  # Exclude the bias term from the returned features

def generate_noisy_labels(y, random_seeds, noise_transition_matrix= None, instance_dependent=False, X=None, noise_transition_dict=None):
    # Generate noisy labels for given random seeds
    noisy_labels = []
    for seed in random_seeds:
        np.random.seed(seed)
        if instance_dependent:
            noisy_labels.append(add_label_noise(y, instance_dependent=instance_dependent, X=X, noise_transition_dict=noise_transition_dict))
        else:
            noisy_labels.append(add_label_noise(y, noise_transition_matrix=noise_transition_matrix))
    return noisy_labels

def train_models(X, y, noisy_ys, X_test, methods, random_seeds, noise_transition_matrix = None):
    # Train models and collect predictions
    predictions = {}
    for method in methods:
        if method == "clean":
            model = train_LR_no_test(X, y, seed=42, num_epochs=100, correction_type="None", noise_transition_matrix=None)
            predictions[(method, 'clean')] = get_predictions_LR(model, X_test)
        else:
            for ny, seed in zip(noisy_ys, random_seeds):
                model = train_LR_no_test(X, ny, seed=42, num_epochs=100, correction_type=method, noise_transition_matrix=noise_transition_matrix)
                predictions[(method, f'seed={seed}')] = get_predictions_LR(model, X_test)
    return predictions

def compile_predictions(predictions, instances_counts, true_labels):
    # Compile predictions into a DataFrame
    rows = []
    for i,(x1, x2) in enumerate(list(true_labels.keys())):
        row = {
            'n': instances_counts[(x1, x2)],
            'x1': x1,
            'x2': x2,
            'y': true_labels[(x1, x2)]
        }
        for (method, label_type), preds in predictions.items():
            row[f'{method} {label_type}'] = preds[i].item()
        rows.append(row)
    return pd.DataFrame(rows)

def coefs_pm1_to_01(coefs):
    """
    :param coefs: coefficient vector of a linear classifier with features x[j] \in {-1,+1}
    :return: coefficient vector of a linear classifier with features x[j] \in {0,+1}
    """
    coefs = np.array(coefs).flatten()
    t = coefs[0] - sum(coefs[1:])
    w = 2.0 * coefs[1:]
    w = np.insert(w, 0, t)
    return w

def linear_model_batch(coefs, X):
    """
    Apply a linear classifier to a batch of input features X.
    
    :param coefs: Coefficients of the linear model (bias, w1, w2, ..., wd).
    :param X: Input features as a 2D array where each row is a set of features.
    :return: The raw output of the linear model for each input set, before applying the threshold.
    """
    # Add a column of 1s to the beginning of X to account for the bias term
    bias = np.ones((X.shape[0], 1))
    
    X_with_bias = np.hstack((bias, X))
    
    # Compute the linear combination of inputs and weights
    z = np.dot(X_with_bias, coefs)

    return z


def output_01(coefs, X):
    """
    Convert the linear model output to the 0, 1 space for a batch of inputs.
    
    :param coefs: Coefficients of the linear model.
    :param X: Batch of input features as a 2D array.
    :return: Array of 0 or 1 depending on the raw model output for each input set.
    """
    z = linear_model_batch((coefs), X)
    
    return (z > 0).astype(int)

# Function to load the CSV file with the coefficients
def load_coefficients(file_path):
    """
    Load coefficients from a CSV file.

    :param file_path: Path to the CSV file.
    :return: Coefficients as a NumPy array.
    """
    converted = []
    for value in pd.read_csv(file_path, header=None).values:
        converted.append(coefs_pm1_to_01(value))
    return converted


def zero_one_loss(y_true, y_pred):
    """
    Calculate the mean 0-1 loss for a set of predictions.
    :param y_true: The true labels.
    :param y_pred: The predicted labels.
    :return: The mean 0-1 loss.
    """
    # Convert y_true and y_pred to numpy arrays if they are not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Calculate the loss
    loss = np.mean(np.where(y_true == y_pred, 0, 1))
    return loss


def backward_zero_one_loss(y_true, y_pred, noise_transition_matrix=None, instance_dependent=False, noise_transition_dict=None, X=None):
    """
    Calculate the noise-corrected 0-1 loss for a set of predictions.

    :param y_true: The observed (noisy) labels.
    :param y_pred: The predicted labels.
    :param noise_transition_matrix: The noise transition matrix.
    :return: The noise-corrected 0-1 loss.
    """
    # Convert y_true and y_pred to numpy arrays if they are not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if instance_dependent:
        
        # Initialize the loss
        corrected_loss = 0
        # Compute the 0-1 loss for each prediction
        for i in range(len(y_true)):
            instance = X[i]
            x0 = instance[0]
            x1 = instance[1]
            tup = (x0,x1)

            # Calculate the inverse noise transition matrix
            noise_transition_matrix = noise_transition_dict[tup]
            inv_noise_transition_matrix = np.linalg.inv(noise_transition_matrix)

            for true_label in range(noise_transition_matrix.shape[0]):
                # Compute the uncorrected 0-1 loss for the current assumed true label
                uncorrected_loss = 1 if y_pred[i] != true_label else 0
                
                # Weight the uncorrected loss by the probability of the true label given the observed label
                corrected_loss += inv_noise_transition_matrix[true_label, y_true[i]] * uncorrected_loss

        # Normalize the corrected loss by the number of samples
        corrected_loss /= len(y_true)
    
    else:

        # Calculate the inverse noise transition matrix
        inv_noise_transition_matrix = np.linalg.inv(noise_transition_matrix)

        # Initialize the loss
        corrected_loss = 0
        # Compute the 0-1 loss for each prediction
        for i in range(len(y_true)):

            for true_label in range(noise_transition_matrix.shape[0]):
                # Compute the uncorrected 0-1 loss for the current assumed true label
                uncorrected_loss = 1 if y_pred[i] != true_label else 0
                
                # Weight the uncorrected loss by the probability of the true label given the observed label
                corrected_loss += inv_noise_transition_matrix[true_label, y_true[i]] * uncorrected_loss

        # Normalize the corrected loss by the number of samples
        corrected_loss /= len(y_true)

    return corrected_loss


def bayes_model(d, X, y, linear_only = True, loss_type = "0-1", noise_transition_matrix = None, noise_transition_dict=None):
    
    if linear_only:
        file_path = "/h/snagaraj/noise_multiplicity/src/linear_classifier_coefficients/coefs_complete_sym_d_0"+str(d)+".csv"

    else:
    # Example file path, replace with actual path pattern
        file_path = "/h/snagaraj/noise_multiplicity/src/linear_classifier_coefficients/coefs_complete_sym_d_0"+str(d)+".csv"

    # Load the coefficients from the CSV file
    # The file path will need to be adjusted to the actual path where your file is stored
    coefficients = (load_coefficients(file_path))

    #random.shuffle(coefficients) ##Shuffling in case of ties, we have different orders we check
    
    best_loss = 100
    
    for i in range(len(coefficients)):
        coefs = (coefficients[i])
        y_pred = output_01(coefs, X)
        
        if loss_type == "0-1":
            loss = zero_one_loss(y, y_pred)
        else:
            if noise_transition_dict!=None:
                loss = backward_zero_one_loss(y, y_pred, instance_dependent=True, X=X, noise_transition_dict=noise_transition_dict)
            
            else:
                loss = backward_zero_one_loss(y, y_pred, noise_transition_matrix)
            
        if loss < best_loss:
            best_loss = loss
            best_model = coefs
    return best_model, best_loss

def compile_bayes_predictions(d, X, y, noisy_ys, random_seeds, instances_counts, true_labels, noise_transition_matrix):
    X_test = np.array(list(true_labels.keys()))
    
    predictions = {}
    
    methods = ["clean", "noisy"]
    losses = ["0-1", "corrected_0-1"]
    
    for method in methods:
        for loss in losses:
            if (method == "clean" and loss == "corrected_0-1"):
                continue
            elif method == "clean" and loss == "0-1":
                best_model, best_loss = bayes_model(d, X, y, loss_type = "0-1")
                predictions[method+"_"+loss] = output_01((best_model), X_test)
            else: #Noisy
                for ny, seed in zip(noisy_ys, random_seeds):
                    best_model_noisy, best_loss = bayes_model(d, X, ny, loss_type = loss, noise_transition_matrix = noise_transition_matrix)
                    predictions[(method+"_"+loss + "_" + f'seed={seed}')] = output_01((best_model_noisy), X_test)
                
    # Compile predictions into a DataFrame
    rows = []
    for i,(x1, x2) in enumerate(list(true_labels.keys())):
        row = {
            'n': instances_counts[(x1, x2)],
            'x1': x1,
            'x2': x2,
            'y': true_labels[(x1, x2)]
        }
        for (key), preds in predictions.items():
            row[f'{key}'] = int(preds[i].item())
        rows.append(row)
    return pd.DataFrame(rows)

# def generate_metrics_toy(noise_levels, m,d , X, y, true_labels, instances_counts, loss_type, noise_type = "class_independent", fixed_class = None, fixed_noise = None, feature_weights=None):
#     """
#     Generate a dictionary of ambiguity rates across a range of noise levels.
    
#     :param noise_levels: A list or array of noise levels to test.
#     :param X: The input features for the dataset.
#     :param y: The true labels for the dataset.
#     :param true_labels: The true labels dictionary for generating the dataset.
#     :param instances_counts: The instances counts for generating the dataset.
#     :return: A dictionary with keys as instances and values as lists of ambiguity rates.
#     """
    
#     X_test = np.array(list(true_labels.keys()))
#     labels = np.array(list(true_labels.values()))

#     error_rates = {str(list(instance)): [] for instance in X_test}
#     disagreement_rates = {str(list(instance)): [] for instance in X_test}
    
#     for flip_p in tqdm(noise_levels):
#         # Generate noisy labels
#         if noise_type == "class_independent":
#             noise_transition_matrix = np.array([[1-flip_p, flip_p], [flip_p, 1-flip_p]])
#             noisy_ys = generate_noisy_labels(y, range(1, m+1), noise_transition_matrix=noise_transition_matrix)
#         elif noise_type == "class_conditional":
#             if fixed_class == 0:
#                 noise_transition_matrix = np.array([[1-fixed_noise, fixed_noise], [flip_p, 1-flip_p]])
#             elif fixed_class == 1:
#                 noise_transition_matrix = np.array([[1-flip_p, flip_p], [fixed_noise, 1-fixed_noise]])
#             noisy_ys = generate_noisy_labels(y, range(1, m+1), noise_transition_matrix=noise_transition_matrix)
#         elif noise_type == "instance_dependent":

#             # flip_00 = flip_p
#             # flip_01 = 0.1
#             # flip_10 = 0.1
#             # flip_11 = 0.1

#             # noise_transition_dict = {(0,0): np.array([[1-flip_00, flip_00], [flip_00, 1-flip_00]]),
#             #                         (0,1): np.array([[1-flip_01, flip_01], [flip_01, 1-flip_01]]),
#             #                         (1,0): np.array([[1-flip_10, flip_10], [flip_10, 1-flip_10]]),
#             #                         (1,1): np.array([[1-flip_11, flip_11], [flip_11, 1-flip_11]])}

#             noise_transition_dict = {}
            
#             for instance in np.unique(X, axis=0):
#                 flip_instance = instance_noise_level(instance, flip_p, feature_weights)
#                 # Use flip_instance to build the noise transition matrix for this instance
#                 noise_transition_matrix = np.array([[1-flip_instance, flip_instance],
#                                                     [flip_instance, 1-flip_instance]])
#                 noise_transition_dict[tuple(instance)] = noise_transition_matrix

            

#             noisy_ys = generate_noisy_labels(y, range(1, m+1), instance_dependent=True, X=X, noise_transition_dict=noise_transition_dict)
        
        
#         predictions = []

#         for noisy_y in noisy_ys:
#             # Train the model and make predictions
#             if noise_type == "instance_dependent":
#                 best_model_noisy, loss = bayes_model(d, X, noisy_y, loss_type=loss_type, noise_transition_dict=noise_transition_dict)
#             else:
#                 best_model_noisy, loss = bayes_model(d, X, noisy_y, loss_type=loss_type, noise_transition_matrix=noise_transition_matrix)
                
#             preds = output_01(best_model_noisy, X_test)
#             predictions.append(preds)

#         predictions = np.array(predictions)
#         error_rate = calculate_error_rate(predictions, labels)
#         disagreement_rate = estimate_disagreement(predictions)

#         for i, item in enumerate(X_test):
#             error_rates[str(list(item))].append(error_rate[i])
#             disagreement_rates[str(list(item))].append(disagreement_rate[i])

#     return error_rates, disagreement_rates

def generate_metrics_toy(noise_levels, m, max_iter, d , X_train, y_train, X_test, y_test, true_labels, instances_counts, noise_type = "class_independent",  p_y_x_dict = None, probabilistic = False, fixed_class = 0, fixed_noise = 0.3, feature_weights=None):
    """
    Generate a dictionary of ambiguity rates across a range of noise levels.
    
    :param noise_levels: A list or array of noise levels to test.
    :param X: The input features for the dataset.
    :param y: The true labels for the dataset.
    :param true_labels: The true labels dictionary for generating the dataset.
    :param instances_counts: The instances counts for generating the dataset.
    :return: A dictionary with keys as instances and values as lists of ambiguity rates.
    """


    #error_rates = {str(list(instance)): [] for instance in X_test}
    #disagreement_rates = {str(list(instance)): [] for instance in X_test}

    error_rates = {noise_level: [] for noise_level in noise_levels}
    disagreement_rates = {noise_level: [] for noise_level in noise_levels}
    
    for flip_p in tqdm(noise_levels):

        if noise_type == "class_independent":
            yn_train, noise_transition_matrix = generate_class_independent_noise(y_train, flip_p)
        elif noise_type == "class_conditional":
            yn_train, noise_transition_matrix = generate_class_conditional_noise(y_train, flip_p, fixed_class, fixed_noise)
        elif noise_type == "instance_dependent":
            yn_train, noise_transition_dict = generate_instance_dependent_noise(y_train, X_train, flip_p, feature_weights)

        predictions = []
        typical_count = 0
        
        for iteration in range(1, max_iter+1):

            u_vec = []
            for seed, (yn, instance) in enumerate(zip(yn_train, X_train)):
                
                if noise_type == "instance_dependent":
                    noise_transition_matrix = noise_transition_dict[tuple(instance)]

                p_y_x = p_y_x_dict[tuple(instance)]

                u = infer_u(yn, noise_transition_matrix, p_y_x, seed = seed+iteration)
                u_vec.append(u)

            #Check if typical
            
            u_vec = np.array(u_vec)
            
            if noise_type == "instance_dependent":
                bool_flag = True
                for instance in np.unique(X_train, axis=0):
                    p_y_x = p_y_x_dict[tuple(instance)]
                    
                    indices = [idx for idx, elem in enumerate(X_train) if np.array_equal(elem, instance)]
                    
                    if not is_typical(u_vec[indices], noise_transition_matrix, yn_train[indices], p_y_x, noise_type = noise_type):
                        bool_flag = False
                        break
                if not bool_flag:
                    continue
                
            else:
                if not is_typical(u_vec, noise_transition_matrix, yn_train, p_y_x, noise_type = noise_type):
                    continue

            cleaned_labels = flip_labels(yn_train, u_vec)

            best_model_noisy, loss = bayes_model(d, X_train, cleaned_labels, loss_type="0-1")

            preds = output_01(best_model_noisy, X_test)

            predictions.append(preds)
            typical_count+=1

            if typical_count==m:
                break

        predictions = np.array(predictions)

        try:
            error_rate = calculate_error_rate(predictions, y_test)
            disagreement_rate = estimate_disagreement(predictions)
        except:
            continue
        for i, item in enumerate(X_test):
            
            error_rates[flip_p].append(error_rate[i])
            disagreement_rates[flip_p].append(disagreement_rate[i])

    
        print(typical_count/iteration)
    return error_rates, disagreement_rates


def generate_metrics_toy_est_noise(noise_level, m,d , X, y, true_labels, instances_counts, loss_type):
    """
    Generate a dictionary of ambiguity rates across a range of estimated noise levels.
    
    :param noise_levels: A list or array of noise levels to test.
    :param X: The input features for the dataset.
    :param y: The true labels for the dataset.
    :param true_labels: The true labels dictionary for generating the dataset.
    :param instances_counts: The instances counts for generating the dataset.
    :return: A dictionary with keys as instances and values as lists of ambiguity rates.
    """
    
    X_test = np.array(list(true_labels.keys()))
    labels = np.array(list(true_labels.values()))

    error_rates = {str(list(instance)): [] for instance in X_test}
    disagreement_rates = {str(list(instance)): [] for instance in X_test}
    
    flip_p = noise_level
    noise_levels = np.linspace(-noise_level, 0.49-noise_level, num=20)
    
    # Generate noisy labels
    if noise_type == "class_independent":
        noise_transition_matrix = np.array([[1-flip_p, flip_p], [flip_p, 1-flip_p]])

    noisy_ys = generate_noisy_labels(y, noise_transition_matrix, range(1, m+1))
    
    
    for delta in tqdm(noise_levels):
        
        noise_transition_matrix_est = np.array([[1-flip_p-delta, flip_p+delta], [flip_p+delta, 1-flip_p-delta]])
        
        predictions = []
        for noisy_y in noisy_ys:
            # Train the model and make predictions
            best_model_noisy, _ = bayes_model(d, X, noisy_y, loss_type=loss_type, noise_transition_matrix=noise_transition_matrix_est)
            preds = output_01(best_model_noisy, X_test)
            predictions.append(preds)

        predictions = np.array(predictions)
        error_rate = calculate_error_rate(predictions, labels)
        disagreement_rate = estimate_disagreement(predictions)

        for i, item in enumerate(X_test):
            error_rates[str(list(item))].append(error_rate[i])
            disagreement_rates[str(list(item))].append(disagreement_rate[i])

    return error_rates, disagreement_rates




def generate_losses_accuracies(noise_levels, m,d , X, y, true_labels, instances_counts, loss_types, noise_type = "class_independent", fixed_class = None, fixed_noise = None):
    """
    Generate a dictionary of ambiguity rates across a range of noise levels.
    
    :param noise_levels: A list or array of noise levels to test.
    :param X: The input features for the dataset.
    :param y: The true labels for the dataset.
    :param true_labels: The true labels dictionary for generating the dataset.
    :param instances_counts: The instances counts for generating the dataset.
    :return: A dictionary with keys as instances and values as lists of ambiguity rates.
    """
    
    X_test = np.array(list(true_labels.keys()))
    labels = np.array(list(true_labels.values()))
    
    loss_dict = {loss_type: [] for loss_type in loss_types}
    accuracy_dict = {loss_type: [] for loss_type in loss_types}

    for flip_p in tqdm(noise_levels):
        # Generate noisy labels
        if noise_type == "class_independent":
            noise_transition_matrix = np.array([[1-flip_p, flip_p], [flip_p, 1-flip_p]])
            noisy_ys = generate_noisy_labels(y, range(1, m+1), noise_transition_matrix=noise_transition_matrix)
        elif noise_type == "class_conditional":
            if fixed_class == 0:
                noise_transition_matrix = np.array([[1-fixed_noise, fixed_noise], [flip_p, 1-flip_p]])
            elif fixed_class == 1:
                noise_transition_matrix = np.array([[1-flip_p, flip_p], [fixed_noise, 1-fixed_noise]])
            noisy_ys = generate_noisy_labels(y, range(1, m+1), noise_transition_matrix=noise_transition_matrix)
        elif noise_type == "instance_dependent":

            flip_00 = flip_p
            flip_01 = 0.1
            flip_10 = 0.1
            flip_11 = 0.1

            noise_transition_dict = {(0,0): np.array([[1-flip_00, flip_00], [flip_00, 1-flip_00]]),
                                    (0,1): np.array([[1-flip_01, flip_01], [flip_01, 1-flip_01]]),
                                    (1,0): np.array([[1-flip_10, flip_10], [flip_10, 1-flip_10]]),
                                    (1,1): np.array([[1-flip_11, flip_11], [flip_11, 1-flip_11]])}

            noisy_ys = generate_noisy_labels(y, range(1, m+1), instance_dependent=True, X=X, noise_transition_dict=noise_transition_dict)
        
        noisy_y = noisy_ys[0]

        for loss_type in loss_types:
            if loss_type == "0-1 Clean":
                best_model, loss = bayes_model(d, X, y, loss_type="0-1")
                
                predictions = np.array(output_01(best_model, X))
                accuracy = np.mean(predictions == y)
                
                loss_dict[loss_type].append(loss)
                accuracy_dict[loss_type].append(accuracy)
                
                
            elif loss_type == "0-1 Noisy":
                # Train the model and make predictions
                best_model, loss = bayes_model(d, X, noisy_y, loss_type="0-1")
                predictions = np.array(output_01(best_model, X))
                accuracy = np.mean(predictions == y)


                loss_dict[loss_type].append(loss)
                accuracy_dict[loss_type].append(accuracy)
            
            elif loss_type == "Corrected 0-1 Noisy":

                if noise_type == "instance_dependent":
                    # Train the model and make predictions
                    best_model, loss = bayes_model(d, X, noisy_y, loss_type="Corrected 0-1", noise_transition_dict=noise_transition_dict)

                else:
                    # Train the model and make predictions
                    best_model, loss = bayes_model(d, X, noisy_y, loss_type="Corrected 0-1", noise_transition_matrix=noise_transition_matrix)

                predictions = output_01(best_model, X)
                accuracy = np.array(np.mean(predictions == y))


                loss_dict[loss_type].append(loss)
                accuracy_dict[loss_type].append(accuracy)

    return loss_dict, accuracy_dict


def generate_noisy_label(y, noise_transition_matrix= None, instance_dependent=False, X=None, noise_transition_dict=None):
    # Generate a single realization of noisy labels
    np.random.seed(2024)
    if instance_dependent:
        return add_label_noise(y, instance_dependent=instance_dependent, X=X, noise_transition_dict=noise_transition_dict)

    else:
        return add_label_noise(y, noise_transition_matrix=noise_transition_matrix)



def calculate_priors_toy(true_labels, instances_counts):
    """
    Calculate prior probabilities based on the observed frequencies from true_labels and instances_counts.

    Parameters:
    - true_labels: Dictionary mapping (x1, x2) pairs to their deterministic labels.
    - instances_counts: Dictionary mapping (x1, x2) pairs to their instance counts.

    Returns:
    - p_y_x_dict: Dictionary mapping (x1, x2) pairs to numpy arrays representing prior probabilities.
    """
    # Initialize counters for each label
    total_instances = sum(instances_counts.values())
    label_counts = {0: 0, 1: 0}
    
    # Count the number of instances for each label
    for (x1, x2), count in instances_counts.items():
        label = true_labels[(x1, x2)]
        label_counts[label] += count

    # Calculate prior probabilities
    p_y_x_dict = {}
    for (x1, x2), _ in true_labels.items():
        label = true_labels[(x1, x2)]
        # Calculate the prior for the current label and the complementary label
        prior = label_counts[label] / total_instances
        complementary_prior = 1 - prior  # Assuming binary labels (0 and 1)
        p_y_x_dict[(x1, x2)] = np.array([complementary_prior, prior]) if label == 1 else np.array([prior, complementary_prior])
    
    return p_y_x_dict


def get_value_counts(arr):
    value_counts = defaultdict(int)
    
    if arr.ndim == 2:
        # 2D array
        for row in arr:
            # Convert row to a tuple
            row_tuple = tuple(row)
            # Increment the count for this tuple in the dictionary
            value_counts[row_tuple] += 1
    elif arr.ndim == 1:
        # 1D array
        for item in arr:
            # Increment the count for this item in the dictionary
            value_counts[item] += 1
    
    return dict(value_counts)



def generate_all_pairs(d):
    """
    Generate all possible binary tuples of length d.

    Args:
        d (int): The dimension of the binary tuple.

    Returns:
        list: A list of tuples representing all possible pairs.
    """
    return list(itertools.product([0, 1], repeat=d))

def complete_binary_dict(d, input_dict):
    # Define all possible 2-value binary combinations
    all_pairs = generate_all_pairs(d)
    
    # Check and add missing pairs
    for pair in all_pairs:
        if pair not in input_dict:
            input_dict[pair] = 0
    
    return input_dict



def regret_toy(d, X, y, noise_type, T, loss_type="0-1", n_draws=10, epsilon=0.1):
    """
    Generate a dictionary of ambiguity rates across a range of noise levels.
    
    :param noise_levels: A list or array of noise levels to test.
    :param X: The input features for the dataset.
    :param y: The true labels for the dataset.
    :param true_labels: The true labels dictionary for generating the dataset.
    :param instances_counts: The instances counts for generating the dataset.
    :return: A dictionary with keys as instances and values as lists of ambiguity rates.
    """
    
    group = np.repeat(0, len(y))
    
    p_y_x_dict = calculate_prior(y, group, noise_type=noise_type)  # Clean prior

    typical_count = 0
    preds_vec = []
    
    
    results = {
               "seed": [],
               "typical":[],
              "noisy_risk": [],
              "clean_risk": [],
              "regret": [],
              "fpr/underreliance": [],
              "fnr/overreliance": []}
    
    
    max_iter = 10*n_draws
    
    instance_metrics = {
        "instance": [],
        "typical":[],
        "seed": [],
        "noisy_risk": [],
        "clean_risk": [],
        "regret": [],
        "fpr/underreliance": [],
        "fnr/overreliance": []
    }
    
    class_metrics = {
        "class": [],
        "typical":[],
        "seed": [],
        "noisy_risk": [],
        "clean_risk": [],
        "regret": [],
        "fpr/underreliance": [],
        "fnr/overreliance": []
    }
    
    for seed in tqdm(range(1, n_draws)):
        u_vec = get_u(y, T=T, seed=seed, noise_type=noise_type)
        
        typical_flag, difference = is_typical(u_vec, p_y_x_dict,  T = T, y_vec = y, noise_type = noise_type, uncertainty_type = "forward", epsilon = epsilon)
        
        if typical_flag:    
            typical_count+=1
        
        yn = flip_labels(y, u_vec)
        
        best_model_noisy, loss = bayes_model(d, X, yn, loss_type="0-1")
        
        preds = output_01(best_model_noisy, X)
        
        error_noisy, err_anticipated = instance_01loss(yn, preds)
        error_clean, err_true = instance_01loss(y, preds)

        fp_vec = ((err_anticipated == 1) & (err_true == 0))
        fn_vec = ((err_anticipated == 0) & (err_true == 1))
        
        
        _, regret_vec = instance_01loss(err_anticipated, err_true)
        
        results["typical"].append(str(typical_flag))
        results["seed"].append(seed) 
        results["noisy_risk"].append(np.mean(err_anticipated)*100)
        results["clean_risk"].append(np.mean(err_true)*100)
        results["fpr/underreliance"].append(np.mean(fp_vec)*100)
        results["fnr/overreliance"].append(np.mean(fn_vec)*100)
        results["regret"].append(np.mean(regret_vec)*100)
        
        # Calculate instance-level metrics
        for instance in np.unique(X, axis=0):
            mask = (X == instance).all(axis=1)
            instance_metrics["instance"].append(str(instance))
            instance_metrics["typical"].append(str(typical_flag))
            instance_metrics["seed"].append(seed)
            instance_metrics["noisy_risk"].append(np.mean(err_anticipated[mask]) * 100)
            instance_metrics["clean_risk"].append(np.mean(err_true[mask]) * 100)
            instance_metrics["regret"].append(np.mean(regret_vec[mask]) * 100)
            instance_metrics["fpr/underreliance"].append(np.mean(fp_vec[mask]) * 100)
            instance_metrics["fnr/overreliance"].append(np.mean(fn_vec[mask]) * 100)
        
        # Calculate class-level metrics
        for class_val in np.unique(y):
            mask = (y == class_val)
            class_metrics["class"].append(class_val)
            class_metrics["typical"].append(str(typical_flag))
            class_metrics["seed"].append(seed)
            class_metrics["noisy_risk"].append(np.mean(err_anticipated[mask]) * 100)
            class_metrics["clean_risk"].append(np.mean(err_true[mask]) * 100)
            class_metrics["regret"].append(np.mean(regret_vec[mask]) * 100)
            class_metrics["fpr/underreliance"].append(np.mean(fp_vec[mask]) * 100)
            class_metrics["fnr/overreliance"].append(np.mean(fn_vec[mask]) * 100)
        
        if typical_count == n_draws:
            break
        
    metrics_df = pd.DataFrame(results)
    instance_metrics_df = pd.DataFrame(instance_metrics)
    class_metrics_df = pd.DataFrame(class_metrics)

    return metrics_df, instance_metrics_df, class_metrics_df


def toy_data_helper(d, X, y, noise_type, noise_level, T, n_draws, loss_type, epsilon, metrics_dfs, instance_metrics_dfs, class_metrics_dfs):
    metrics_df, instance_metrics_df, class_metrics_df = regret_toy(d, X, y, noise_type=noise_type, n_draws=n_draws, T=T, loss_type=loss_type, epsilon=epsilon)

    metrics_df["noise"], instance_metrics_df["noise"], class_metrics_df["noise"] = noise_level, noise_level, noise_level

    metrics_dfs.append(metrics_df)
    instance_metrics_dfs.append(instance_metrics_df)
    class_metrics_dfs.append(class_metrics_df)



def plot_regret_toy(metrics_df, instance_metrics_df, class_metrics_df):
    # Overall metrics plot for typical == True
    melted_df = metrics_df.melt(id_vars=['seed', 'typical', 'noise'], 
                                value_vars=['noisy_risk', 'clean_risk', 'fpr/underreliance', 'fnr/overreliance'], 
                                var_name='metric', value_name='value')
    melted_df['style'] = melted_df['metric'].apply(lambda x: 'risk' if 'risk' in x else 'other')
    
    # Defining a color palette that forces similar colors within each style
    palette = {
        'noisy_risk': '#4287f5',
        'clean_risk': '#7142c2',
        'fpr/underreliance': '#ebe534',
        'fnr/overreliance': '#c73326'
    }

    plt.figure(figsize=(7, 4))
    sns.lineplot(data=melted_df[melted_df['typical'] == "True"], x='noise', y='value', hue='metric', style = 'style', palette=palette)
    plt.xlabel('Noise')
    plt.ylabel('Value')
    plt.legend(title='Metric', loc='upper left', bbox_to_anchor=(1, 1))
    plt.title('Metrics vs Noise (Typical == True)')

    # Removing the style from the legend
    handles, labels = plt.gca().get_legend_handles_labels()
   
    plt.legend(handles[1:-3], labels[1:-3], title='Metric', loc='upper left', bbox_to_anchor=(1, 1))
    
    # Define the metrics to plot
    metrics = ["noisy_risk", "clean_risk", "regret", "fpr/underreliance", "fnr/overreliance"]

    # Instance-level metrics plot
    # Ensure the hue column is correct and free of any formatting issues
    instance_metrics_df['instance'] = instance_metrics_df['instance'].str.strip()

    fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=(20, 5), sharex=True)

    for i, metric in enumerate(metrics):
        sns.lineplot(
            data=instance_metrics_df[instance_metrics_df["typical"] == "True"], 
            x='noise', 
            y=metric, 
            hue='instance', 
            ax=axes[i], 
            marker='o', 
            legend='full'  # Force the creation of the legend
        )
        axes[i].legend().remove()  # Remove the subplot legends
        axes[i].set_title(metric)
        axes[i].set_xlabel('Noise')
        axes[i].set_ylabel(metric)

    # Adjust layout to make room for the shared legend
    plt.tight_layout()
    plt.suptitle('Instance-level Metrics vs Noise', y=1.02)

    # Now get handles and labels
    handles, labels = axes[0].get_legend_handles_labels()

    # Create a shared legend
    fig.legend(handles=handles, labels=labels, loc='upper center', ncol=10, bbox_to_anchor=(0.5, -0.0), frameon=False)

    # Adjust the space to accommodate the shared legend
    plt.subplots_adjust(top=0.85, bottom=0.15)
    plt.show()

    # Class-level metrics plot
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=(20, 5), sharex=True)
    for i, metric in enumerate(metrics):
        sns.lineplot(data=class_metrics_df[class_metrics_df["typical"]=="True"], x='noise', y=metric, hue='class', ax=axes[i], marker='o', legend = "full")
        axes[i].legend().remove()  # Remove the subplot legends
        axes[i].set_title(metric)
        axes[i].set_xlabel('Noise')
        axes[i].set_ylabel(metric)
    plt.tight_layout()
    plt.suptitle('Class-level Metrics vs Noise', y = 1.02)
    
     # Now get handles and labels
    handles, labels = axes[0].get_legend_handles_labels()

    # Create a shared legend
    fig.legend(handles=handles, labels=labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.0), frameon=False)

    # Adjust the space to accommodate the shared legend
    plt.subplots_adjust(top=0.85, bottom=0.15)
    plt.show()


def run_procedure_toy(d, m, max_iter, X, yn, p_y_x_dict, group = None,noise_type = "class_independent",  T = None, epsilon = 0.10, misspecify = False):
    
    typical_count = 0
    preds_all = []
    errors = []
    
    y_vec = yn
    
    
    

    for seed in (range(1, max_iter+1)):
        
        u_vec = infer_u(y_vec, group = group, noise_type = noise_type, p_y_x_dict = p_y_x_dict,  T = T , seed=seed)
        
        typical_flag, _ = is_typical(u_vec, p_y_x_dict, group = group,  T = T, y_vec = y_vec, noise_type = noise_type, uncertainty_type = "backward", epsilon = epsilon)

        if misspecify:
            typical_flag = True
            
        if not typical_flag:
            continue
            
        flipped_labels = flip_labels(y_vec, u_vec)

        
        model, loss = bayes_model(d, X, flipped_labels, loss_type="0-1")

        
        preds = output_01(model, X)

        
        preds_all.append(preds)


        error = [preds[i]!= flipped_labels[i] for i in range(len(preds))]
        
        errors.append(error)
        typical_count += 1

        if typical_count == m:
            break
    errors = np.array(errors)

    ambiguity = np.mean(errors, axis=0)*100
    predictions = np.array(preds_all)
    #disagreement = estimate_disagreement(predictions)

    return ambiguity



def abstain_toy(d, m, max_iter, X, y, noise_type, T, loss_type="0-1", n_draws=10, epsilon=0.1):
    """
    Generate a dictionary of ambiguity rates across a range of noise levels.
    
    :param noise_levels: A list or array of noise levels to test.
    :param X: The input features for the dataset.
    :param y: The true labels for the dataset.
    :param true_labels: The true labels dictionary for generating the dataset.
    :param instances_counts: The instances counts for generating the dataset.
    :return: A dictionary with keys as instances and values as lists of ambiguity rates.
    """
    
    group = np.repeat(0, len(y))
    
    p_y_x_dict = calculate_prior(y, group, noise_type=noise_type)  # Clean prior

    typical_count = 0
    preds_vec = []
    
    
    results = {
                "seed": [],
                "susceptible": [],
                "posterior": [],
              "ambiguity": [],
                "preds": [],
                  "yn": [],
                  "y": [],
                "X":[]}
    

    for seed in tqdm(range(1, n_draws*10+1)):
        u_vec = get_u(y, T=T, seed=seed, noise_type=noise_type)

        typical_flag, difference = is_typical(u_vec, p_y_x_dict,  T = T, y_vec = y, noise_type = noise_type, uncertainty_type = "forward", epsilon = epsilon)
        
        if typical_flag:    
            typical_count+=1
            
        else:
            continue
        
        yn = flip_labels(y, u_vec)
        
        posterior = calculate_posterior(yn, T, p_y_x_dict[0])
        results["posterior"].append(posterior)
        susceptible = np.sum(posterior > 0)/len(X)
        
        #TRAIN MODEL ON NOISY DATASET
        best_model_noisy, loss = bayes_model(d, X, yn, loss_type="0-1")
        preds = output_01(best_model_noisy, X)
    
        
        #COMPUTE REGRET METRICS
        error_noisy, err_anticipated = instance_01loss(yn, preds)
        error_clean, err_true = instance_01loss(y, preds)

        fp_vec = ((err_anticipated == 1) & (err_true == 0))
        fn_vec = ((err_anticipated == 0) & (err_true == 1))
        
        _, regret_vec = instance_01loss(err_anticipated, err_true)
        
        #RUN OUR PROCEDURE
        ambiguity = run_procedure_toy(d, m, max_iter, X, yn, p_y_x_dict, group = group, noise_type = noise_type,  T = T, epsilon = epsilon)
        
        results["seed"].append(seed) 
        results["susceptible"].append(susceptible)
        #results["disagreement"].append(disagreement)
        results["ambiguity"].append(ambiguity)
        results["preds"].append(preds)
        results["yn"].append(yn)
        results["y"].append(y)
        results["X"].append(X)  
        
        if typical_count == n_draws:
            break
        
    metrics_df = pd.DataFrame(results)
    return metrics_df





def compute_abstain_metrics_toy(abstain_percentage, preds, criteria, y_vec, yn_vec):
    n = len(preds)
    
    abstain_count = int(abstain_percentage * n)
                           
    abstain = abstain_order(criteria, abstain_count)

    non_abstain = (1 - abstain).astype(bool)  # abstention vector

    coverage = np.sum(non_abstain)/n

    err_true = abs(preds - y_vec)  # full err_true
    
    subset_err_true = err_true[non_abstain]
    
    clean_risk = (np.mean(err_true * non_abstain)) / coverage if coverage > 0 else 0.0
    
    err_anticipated = abs(preds - yn_vec)  # full err_anticipated
    subset_err_anticipated = err_anticipated[non_abstain]

    regret = (1/n)*np.sum(abs(subset_err_anticipated - subset_err_true))/ coverage if coverage > 0 else 0.0

    # Calculate False Positives (FP) and False Negatives (FN)
    fp = ((subset_err_anticipated == 1) & (subset_err_true == 0))
    fn = ((subset_err_anticipated == 0) & (subset_err_true == 1))

    tp = ((subset_err_anticipated == 1) & (subset_err_true == 1))
    tn = ((subset_err_anticipated == 0) & (subset_err_true == 0))

    fpr = (1/n) * (np.sum(fp)) / coverage if coverage > 0 else 0.0
    fnr = (1/n) * (np.sum(fn)) / coverage if coverage > 0 else 0.0

    tpr = (1/n) * (np.sum(tp)) / coverage if coverage > 0 else 0.0
    tnr = (1/n) * (np.sum(tn)) / coverage if coverage > 0 else 0.0

    return {
        'fpr': 100*fpr,
        'fnr': 100*fnr,
        'tpr': 100*tpr,
        'tnr': 100*tnr,
        'regret': 100*regret,
        'non_regret': 100*(1-regret),
        'coverage': 100*coverage,
        'clean_risk': 100*clean_risk
    }, non_abstain


def calculate_metrics_abstain_toy(df, noise_type="class_conditional", fixed_class=0, fixed_noise=0.0):
    splits = []
    metrics = []
    values = []
    coverages = []
    thresholds = []
    draw_ids = []
    methods = []
    Xs = []
    ys = []
    yns = []



    for draw_id in df.seed.unique():
        

        sub_df = df[(df["seed"] == draw_id)]

        metric_lis = ['clean_risk','regret','non_regret' ,'fpr', 'fnr', 'tpr', 'tnr'] 


        ambiguity = np.clip(sub_df.ambiguity.values[0] / 100, 0, 1)
        #disagreement = np.clip(sub_df.disagreement.values[0] / 100, 0, 1)
        preds = sub_df.preds.values[0]
        
        y = sub_df.y.values[0]
        yn = sub_df.yn.values[0]
        X = sub_df.X.values[0]
        
        u = abs(y-yn)

        d = len(X[0])
        
        
        for method in ["ambiguity", "random"]:
            if method == "ambiguity":
                criteria = ambiguity
            elif method == "disagreement":
                criteria = disagreement
            elif method == "1-ambiguity":
                criteria = 1-ambiguity
            else: #
                criteria = np.random.uniform(0, 1, len(X))

            for abstain_percentage in np.linspace(0, 0.99, 100):

                abstain_metrics, non_abstain = compute_abstain_metrics_toy(abstain_percentage, preds, criteria, y_vec = y, yn_vec = yn)

                for metric in metric_lis:

                    metrics.append(metric)

                    values.append(abstain_metrics[metric])
                    coverages.append(abstain_metrics['coverage'])

                    thresholds.append(abstain_percentage)

                    draw_ids.append(draw_id)
                    methods.append(method)

                metrics.append("X_counts")
                values.append(complete_binary_dict(d, get_value_counts(X[non_abstain])))
                coverages.append(abstain_metrics['coverage'])
                thresholds.append(abstain_percentage)
                draw_ids.append(draw_id)
                methods.append(method)

                metrics.append("y_counts")
                values.append(get_value_counts(y[non_abstain]))
                coverages.append(abstain_metrics['coverage'])
                thresholds.append(abstain_percentage)
                draw_ids.append(draw_id)
                methods.append(method)

                metrics.append("u_counts")
                values.append(get_value_counts(u[non_abstain]))
                coverages.append(abstain_metrics['coverage'])
                thresholds.append(abstain_percentage)
                draw_ids.append(draw_id)
                methods.append(method)
            
            
    # Create a DataFrame from the arrays
    data = pd.DataFrame({
        #'train': splits,
        'metric': metrics,
        'value': values,
        'coverage': coverages,
        'threshold': thresholds,
        'draw_id': draw_ids,
        'method': methods
    })
    return data


def plot_abstain_toy(data, noise_level):
    data["abstention"] = 100 - data["coverage"]

    # Set the font style to sans-serif
    plt.rcParams["font.family"] = "sans-serif"

    metrics = ["regret", "fpr", "fnr", "clean_risk"]
    count_metrics = ["X_counts", "y_counts", "u_counts"]
    
    # Define your custom color palette for each method
    method_colors = {
        "ambiguity": "#8896FB",   # Purple
        "disagreement": "#808080",  # Gray
         "1-ambiguity": "#00ff00",  # Green
        "random": "black"
    }

    # Plot total level metrics
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    sub_data = data

    for i, metric in enumerate(metrics):
        method_data = sub_data[sub_data['metric'] == metric]
        #color = "#8896FB"
        sns.lineplot(data=method_data, x="abstention", y="value", hue = "method", ax=axes[i], linewidth=1, palette = method_colors)
#         axes[i].scatter(method_data["abstention"].values, method_data["value"].values, c = method_data["method"].values, s=5, alpha=0.5)

        axes[i].set_xlabel("Abstention Rate", fontsize=14)
        axes[i].set_ylabel(metric, fontsize=14)
        axes[i].set_title(f"{metric} (Noise Level: {noise_level})")
        axes[i].grid(True, which='both', color='grey', linestyle='-', linewidth=0.5)
        axes[i].tick_params(axis='both', which='major', labelsize=12)
        axes[i].tick_params(axis='both', which='minor', labelsize=12)

    
    
    # Removing the style from the legend
    handles, labels = plt.gca().get_legend_handles_labels()
   
    plt.legend(handles, labels, title='', loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()

    metrics = ['non_regret', 'tpr', 'tnr',  "clean_risk"]
    count_metrics = ["X_counts", "y_counts", "u_counts"]
    
    # Define your custom color palette for each method
    method_colors = {
        "ambiguity": "#8896FB",   # Purple
        "disagreement": "#808080",  # Gray
         "1-ambiguity": "#00ff00",  # Green
        "random": "black"
    }

    # Plot total level metrics
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    sub_data = data

    for i, metric in enumerate(metrics):
        method_data = sub_data[sub_data['metric'] == metric]
        #color = "#8896FB"
        sns.lineplot(data=method_data, x="abstention", y="value", hue = "method", ax=axes[i], linewidth=1, palette = method_colors)
#         axes[i].scatter(method_data["abstention"].values, method_data["value"].values, c = method_data["method"].values, s=5, alpha=0.5)

        axes[i].set_xlabel("Abstention Rate", fontsize=14)
        axes[i].set_ylabel(metric, fontsize=14)
        axes[i].set_title(f"{metric} (Noise Level: {noise_level})")
        axes[i].grid(True, which='both', color='grey', linestyle='-', linewidth=0.5)
        axes[i].tick_params(axis='both', which='major', labelsize=12)
        axes[i].tick_params(axis='both', which='minor', labelsize=12)

    
    
    # Removing the style from the legend
    handles, labels = plt.gca().get_legend_handles_labels()
   
    plt.legend(handles, labels, title='', loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()

    # Plot the distribution of X_counts, y_counts, yn_counts vs abstention rate using catplot
    for count_metric in count_metrics:
        distribution_data = []

        metric_data = data[data['metric'] == count_metric]
        for index, row in metric_data.iterrows():
            abstention = row['abstention']
            counts = row['value']  # assuming the counts are stored in the 'value' column
            draw_id = row['draw_id']
            method = row['method']
            for key, value in counts.items():
                distribution_data.append({'abstention': abstention, 'instance': key, 'count': value, 'count_type': count_metric, 'draw_id':draw_id, 'method':method})

        distribution_df = pd.DataFrame(distribution_data)
        
        # Create instances_counts dictionary based on abstention == 0.0
        instances_counts = distribution_df[(distribution_df['abstention'] == 0.0)& (distribution_df['draw_id'] == distribution_df['draw_id'].min())].groupby('instance')['count'].sum().to_dict()
        
        distribution_df['normalized_count'] = distribution_df.apply(lambda row: row['count'] / instances_counts[row['instance']]*len(distribution_df.method.unique()), axis=1)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        
        g = sns.lineplot(
            data=distribution_df, x="abstention", y="normalized_count", hue="instance", style ="method", palette="pastel", ax=ax)
        ax.set_title(f"{count_metric} vs Abstention Rate")
        ax.set_xlabel("Abstention Rate")
        ax.set_ylabel("Normalized Count")
        ax.legend(title='Instance')
        
            # Removing the style from the legend
        handles, labels = plt.gca().get_legend_handles_labels()

        
        plt.legend(handles[1:], labels[1:], title="", loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()