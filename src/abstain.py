import sys
sys.path.insert(0,'..')

from src.models import *
from src.loss_functions import *
from src.noise import *
from src.metrics import *
from src.plotting import *
from src.generate_data import *
from src.real_data import *

import sklearn
import pandas as pd

from scipy.stats import bernoulli

from operator import xor

import os


def run_procedure_abstain(m, max_iter, X_train, yn_train, X_test, yn_test, y_test, p_y_x_dict, 
                  group_train=None, group_test=None, noise_type="class_independent", 
                  model_type="LR", T=None, epsilon=0.1, misspecify=False):
    """
    Runs the procedure for training and evaluating models under noisy labels.

    Args:
        m (int): Number of models to train.
        max_iter (int): Maximum number of iterations.
        X_train (numpy.ndarray): Training feature matrix.
        yn_train (numpy.ndarray): Noisy training labels.
        X_test (numpy.ndarray): Test feature matrix.
        y_test (numpy.ndarray): Test labels.
        p_y_x_dict (dict): Dictionary mapping class probabilities for noise modeling.
        group_train (numpy.ndarray, optional): Training group labels (default: None).
        group_test (numpy.ndarray, optional): Test group labels (default: None).
        noise_type (str, optional): Type of noise ("class_independent" or "group").
        model_type (str, optional): Type of model ("LR" for logistic regression, etc.).
        T (numpy.ndarray, optional): Noise transition matrix.
        epsilon (float, optional): Epsilon value for typicality checking.
        misspecify (bool, optional): Whether to force a misspecified noise model.

    Returns:
        tuple: Arrays containing plausible labels, training predictions, test predictions, 
               training probabilities, and test probabilities.
    """
    
    preds_test, preds_train = [], []
    probs_test, probs_train = [], []
    plausible_labels_train, plausible_labels_test = [], []

    #y_vec = yn_train

    yn_all = np.concatenate([yn_train, yn_test]) 
    group_all = np.concatenate([group_train, group_test]) 

    for seed in tqdm(range(1, min(m, max_iter) + 1)):  # Ensure we don't exceed both `m` and `max_iter`
        
        # Infer noisy label distribution
        u_all = infer_u(yn_all, group=group_all, noise_type=noise_type, 
                        p_y_x_dict=p_y_x_dict, T=T, seed=seed)

        # Typicality Check
        if not misspecify and noise_type != "group":
            typical_flag, _ = is_typical(u_all, p_y_x_dict, group=group_all,  
                                         T=T, y_vec=yn_all, noise_type=noise_type, 
                                         uncertainty_type="backward", epsilon=epsilon)
        else:
            typical_flag = True  # Default to True when misspecified or using "group" noise
        
        if not typical_flag:
            continue
        
        # Flip labels based on noise model
        yhat_all = flip_labels(yn_all, u_all)

        # Step 4: Split back into train and test sets
        u_train, u_test = u_all[:len(yn_train)], u_all[len(yn_train):]
        yhat_train, yhat_test = yhat_all[:len(yn_train)], yhat_all[len(yn_train):]

        # Train model with noisy labels
        _, metrics = train_model_ours(X_train, yhat_train, X_test, yhat_test, 
                                          seed=2024, model_type=model_type)
        
        # Unpack metrics
        train_acc, test_acc, train_probs, test_probs, train_loss, test_loss, train_preds, test_preds = metrics

        # Store results
        preds_test.append(test_preds)
        preds_train.append(train_preds)
        probs_test.append(test_probs)
        probs_train.append(train_probs)
        plausible_labels_train.append(yhat_train)
        plausible_labels_test.append(yhat_test)

        # Break when `m` models are trained
        if len(preds_train) >= m:
            break

    return (
        np.array(plausible_labels_train), 
        np.array(plausible_labels_test), 
        np.array(preds_train), 
        np.array(preds_test), 
        np.array(probs_train), 
        np.array(probs_test)
    )

def load_abstain(dataset, model_type, noise_type, epsilon=0.1, fixed_class = 0, fixed_noise = 0.0, misspecify = "correct"):
    path = "/scratch/hdd001/home/snagaraj/results/abstain/"
    
    file_path = os.path.join(path, dataset, model_type, noise_type, misspecify ,f"{epsilon}.pkl")
    
    with open(file_path, 'rb') as file:
            # Use pickle to write the dictionary to the file
            data = pkl.load(file)
            
    return data
    

def abstain_order(uncertainty_measures, abstain_count):
    # Sort the indices of the uncertainty measures in descending order
    sorted_indices = np.argsort(uncertainty_measures)[::-1]
    
    # Create a mask vector initialized to 0
    mask = np.zeros_like(uncertainty_measures, dtype=int)
    
    # Set the first 'abstain_count' elements to 1
    mask[sorted_indices[:abstain_count]] = 1
    
    return mask

def compute_abstain_group_stats(abstain_percentage, group_val, group_vec, preds, criteria, y_vec, train = True, yn_vec = None):
    """
    Computes false positive rate (FPR), false negative rate (FNR), and regret rate for a given group value.

    Parameters:
    - group_val: The group value to compute statistics for.
    - group: Group label vector
    - err_true: True errors
    - (Optional) instance_err_pred_test: Anticipated errors 

    Returns:
    - Dictionary containing FPR, FNR, and regret rates for the specified group and/or Risk.

    """
    n = len(preds)
    
    abstain_count = int(abstain_percentage * n)
                           
    abstain = abstain_order(criteria, abstain_count)

    non_abstain = (1 - abstain).astype(bool)  # abstention vector

    coverage = np.sum(non_abstain)/n

    err_true = abs(preds - y_vec)  # full err_true

    subset_err_true = err_true[non_abstain]
    subset_group_vec = group_vec[non_abstain]
    
    group_indices = np.where(group_vec == group_val)[0]
    
    risk_in_group = np.sum(err_true[group_indices])
    total_in_group = len(group_indices)
    
    if train:
        
        err_anticipated = abs(preds - yn_vec)
        subset_err_anticipated = err_anticipated[non_abstain]
        
        # Calculate rates for training and test sets
        
        err_true_group = err_true[group_indices]
        err_anticipated_group = err_anticipated[group_indices]

        fp = np.sum((err_anticipated_group == 1) & (err_true_group == 0))
        fn = np.sum((err_anticipated_group == 0) & (err_true_group == 1))

        total_in_group = len(group_indices)

        fpr = fp / total_in_group if total_in_group > 0 else 0.0
        fnr = fn / total_in_group if total_in_group > 0 else 0.0
        
        regret = abs(err_true - err_anticipated)
        regret_in_group = np.sum(regret[group_indices])
        total_in_group = len(group_indices)
        
    
        return {
            'fpr': 100*fpr,
            'fnr': 100*fnr,
            'regret': 100*regret_in_group / total_in_group if total_in_group > 0 else 0.0,
            'coverage': 100*coverage,
            'clean_risk': 100*risk_in_group / total_in_group if total_in_group > 0 else 0.0
        }
    else:
        return {
            'coverage': 100*coverage,
            'clean_risk': 100*risk_in_group / total_in_group if total_in_group > 0 else 0.0
        }

def compute_abstain_metrics(abstain_percentage, preds, criteria, y_vec,  yn_vec):
    n = len(preds)
    
    abstain_count = int(abstain_percentage * n)
                           
    abstain = abstain_order(criteria, abstain_count)

    non_abstain = (1 - abstain).astype(bool)  # abstention vector

    coverage = np.sum(non_abstain)/n

    err_true = abs(preds - y_vec)  # full err_true
    
    subset_err_true = err_true[non_abstain]
    
    risk = (1/n)*np.sum((subset_err_true))/ coverage if coverage > 0 else 0.0

    err_anticipated = abs(preds - yn_vec)  # full err_anticipated
    subset_err_anticipated = err_anticipated[non_abstain]
    
    regret = (1/n)*np.sum(abs(subset_err_anticipated - subset_err_true))/ coverage if coverage > 0 else 0.0
    
    # Calculate False Positives (FP) and False Negatives (FN)
    fp = ((subset_err_anticipated == 1) & (subset_err_true == 0))
    fn = ((subset_err_anticipated == 0) & (subset_err_true == 1))

    fpr = (1/n) * (np.sum(fp)) / coverage if coverage > 0 else 0.0
    fnr = (1/n) * (np.sum(fn)) / coverage if coverage > 0 else 0.0

    return {
        'fpr': 100*fpr,
        'fnr': 100*fnr,
        'regret': 100*regret,
        'coverage': 100*coverage,
        'risk': 100*risk
    }

    

def load_dataset_splits(dataset, group=""):

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))  


    filepath = os.path.join(parent_dir, "data", dataset , f"{dataset}_{group}_processed.pkl")
    with open(filepath, 'rb') as file:

        # Use pickle to write the dictionary to the file
        [X_train, X_test, y_train, y_test, group_train, group_test] = pkl.load(file)

    return X_train, X_test, y_train, y_test, group_train, group_test


def metrics_active_learning(dataset, noise_type, model_type, data, fixed_class=0, fixed_noise=0.0):
    """
    Runs an experiment with three abstention handling strategies:
      - "abstain": compute metrics without retraining,
      - "drop": drop abstained instances and retrain,
      - "relabel": replace abstained labels with clean labels and retrain.
    
    Args:
        dataset (str): Name of the dataset.
        noise_type (str): Noise type ("class_independent" or "class_conditional").
        model_type (str): Model type to train.
        data (pd.DataFrame): DataFrame containing abstain-related results.
        fixed_class (int): Fixed class for noise generation.
        fixed_noise (float): Fixed noise level.
    
    Returns:
        pd.DataFrame: Results with abstention metrics and retrained model performance.
    """

    results = []

    # Load dataset splits
    X_train, X_test, y_train, y_test, _, _ = load_dataset_splits(dataset, group="age")
    y_train, y_test = y_train.astype(int), y_test.astype(int)
    
    df = pd.DataFrame(data)
    unique_noise_levels = df["noise"].unique()
    unique_losses = df["loss"].unique()
    unique_draw_ids = df["draw_id"].unique()

    metric_lis = ["risk","regret", "fpr", "fnr"]
    abstain_percentages = np.linspace(0, 0.99, 100)

    for noise_level in unique_noise_levels:
        # Generate noise transition matrix
        T = (generate_class_independent_noise(y_train, noise_level)[1]
             if noise_type == "class_independent"
             else generate_class_conditional_noise(y_train, noise_level, fixed_class, fixed_noise)[1])
        
        for loss in unique_losses:
            for draw_id in tqdm(unique_draw_ids, desc="Processing Draws"):

                # Step 1: Merge y_train and y_test
                y_all = np.concatenate([y_train, y_test])  # Shape: (total_instances,)
                # Step 2: Generate a single noise vector for all instances
                u_draw = get_u(y_all, T=T, seed=draw_id, noise_type=noise_type)  # Using T_train for now
                # Step 3: Flip labels using the generated noise vector
                yn_all = flip_labels(y_all, u_draw)
                # Step 4: Split back into train and test sets
                u_train, u_test = u_draw[:len(y_train)], u_draw[len(y_train):]
                yn_train, yn_test = yn_all[:len(y_train)], yn_all[len(y_train):]

                # Filter and retrieve one row from the DataFrame
                sub_df = df[(df["loss"] == loss) & (df["noise"] == noise_level) & (df["draw_id"] == draw_id)].iloc[0]
                plausible_labels_train = sub_df.plausible_labels_train
                plausible_labels_test = sub_df.plausible_labels_test
                probs_train, probs_test = sub_df.probs_train, sub_df.probs_test
                preds_train, preds_test = sub_df.preds_train, sub_df.preds_test
                model_probs_train, model_probs_test = sub_df.final_model_probs_train, sub_df.final_model_probs_test
                
                model_preds_train = np.argmax(model_probs_train, axis=1)
                model_preds_test = np.argmax(model_probs_test, axis=1)
                
                # Compute training abstention criteria
                ambiguity_train = calculate_ambiguity(preds_train, plausible_labels_train)
                #uncertainty_train = 1 - np.max(model_probs_train, axis=1)
                mean_probs_plausible = np.mean(probs_train, axis=0)
                uncertainty_train = 1 - (np.abs(mean_probs_plausible - 0.5) * 2)
                
                # Compute test abstention criteria
                ambiguity_test = calculate_ambiguity(preds_test, plausible_labels_test)
                uncertainty_test = 1 - np.max(model_probs_test, axis=1)
                #mean_probs_plausible = np.mean(probs_test, axis=0)
                #uncertainty_plausible = 1 - (np.abs(mean_probs_plausible - 0.5) * 2)
                #majority = compute_majority_vote(preds_test)
                #LOO = compute_loo(probs_test)
                # confidence_scores, prediction_sets, conformal_ambiguity = conformal_prediction_plausible(
                #     probs_test, probs_train, plausible_labels, alpha=0.01
                # )
                # entropy = compute_entropy_score(probs_test)
                # disagreement_test = estimate_disagreement(probs_test)
                
                # Loop over different abstention percentages
                for abstain_pct in tqdm(abstain_percentages, desc="Processing Abstain Levels", leave=False):
                    
                    # Loop over test criteria methods
                    for method_name, crit in [("confidence", uncertainty_test),
                                                ("ambiguity", ambiguity_test)]:

                        # Compute abstain metrics on TEST set
                        abstain_metrics = compute_abstain_metrics(
                            abstain_pct, model_preds_test, crit, y_vec=y_test, yn_vec=yn_test)
                            
                        for metric in metric_lis:
                            results.append({
                                "metric": metric,
                                "value": abstain_metrics[metric],
                                "coverage": abstain_metrics["coverage"],
                                "threshold": abstain_pct,
                                "noise": noise_level,
                                "loss": loss,
                                "draw_id": draw_id,
                                "method": method_name,
                                "experiment_type": "abstain"
                            })
                    # Loop over training criteria methods
                    for method_name, crit in [("confidence", uncertainty_train),
                                                ("ambiguity", ambiguity_train)]:
                        for exp_type in ["drop"]:
                            
                            abstain_count = int(abstain_pct * len(X_train))
                            abstain = abstain_order(crit, abstain_count)
                            non_abstain = (1 - abstain).astype(bool)
                            coverage = np.mean(non_abstain) * 100

                            if exp_type == "drop":
                                X_train_subset = X_train[non_abstain]
                                yn_train_subset = yn_train[non_abstain]
                            elif exp_type == "relabel":
                                yn_train_relabel = yn_train.copy()
                                yn_train_relabel[abstain.astype(bool)] = y_train[abstain.astype(bool)]
                                X_train_subset = X_train
                                yn_train_subset = yn_train_relabel
                            
                            try:
                                # Retrain model using the selected strategy
                                model, (train_acc, test_acc, _, _, train_loss, test_loss, train_preds, _) = train_model_ours_regret(
                                    X_train_subset, yn_train_subset, X_test, y_test, 2024, model_type=model_type
                                )
                                
                                
                                    

                                results.append({
                                    "metric": "risk",
                                    "value": 100 * (1 - test_acc),
                                    "coverage": coverage,
                                    "threshold": abstain_pct,
                                    "noise": noise_level,
                                    "loss": loss,
                                    "draw_id": draw_id,
                                    "method": method_name,
                                    "experiment_type": exp_type
                                })
                                
                                 
                            except:
                                
                                break

    return pd.DataFrame(results)



def metrics_abstain(data, dataset, model_type="LR", noise_type="class_conditional", misspecify="correct", fixed_class=0, fixed_noise=0.0):
    results = []

    # Load dataset splits
    X_train, X_test, y_train, y_test, group_train, group_test = load_dataset_splits(dataset, group="age")
    y_train, y_test = y_train.astype(int), y_test.astype(int)

    # Load abstain data
    #df = pd.DataFrame(load_abstain(dataset, model_type, noise_type, misspecify=misspecify))
    df = pd.DataFrame(data)
    # Precompute unique values to avoid repeated calls
    unique_noise_levels = [0.2] #df["noise"].unique()
    unique_losses =  ["BCE"]#df["loss"].unique()
    unique_draw_ids = df["draw_id"].unique()

    metric_lis = ['regret', 'fpr', 'fnr']
    abstain_percentages = np.linspace(0, 0.50, 100)

    for noise_level in unique_noise_levels:
        # Precompute noise transformation matrix
        T = generate_class_independent_noise(y_train, noise_level)[1] if noise_type == "class_independent" else \
            generate_class_conditional_noise(y_train, noise_level, fixed_class, fixed_noise)[1]

        for loss in unique_losses:
            for draw_id in tqdm(unique_draw_ids):
                # Filter once and reuse
                sub_df = df[(df["loss"] == loss) & (df["noise"] == noise_level) & (df["draw_id"] == draw_id)].iloc[0]

                ambiguity = np.clip(sub_df.ambiguity_train / 100, 0, 1)
                probs = sub_df.train_probs
                test_probs = sub_df.test_probs
                
                unanticipated_retrain = sub_df.unanticipated_retrain

                # Compute noisy labels
                u_vec = get_u(y_train, T=T, seed=draw_id, noise_type=noise_type)
                yn_train = flip_labels(y_train, u_vec)  

                # Predictions & uncertainty
                if probs.ndim == 2:
                    preds, confidence = np.argmax(probs, axis=1), np.max(probs, axis=1)
                    test_preds, confidence_retrain = np.argmax(test_probs, axis=1), np.max(test_probs, axis=1)
                else:
                    preds = (probs > 0.5).astype(int)
                    confidence = np.where(probs > 0.5, probs, 1 - probs)
                    
                    test_preds = (test_probs > 0.5).astype(int)
                    confidence_retrain = np.where(unanticipated_retrain > 0.5, unanticipated_retrain, 1 - unanticipated_retrain)

                uncertainty = 1 - confidence

                for abstain_percentage in tqdm(abstain_percentages):
                    for method, criteria in [("ambiguity", ambiguity), ("confidence", uncertainty)]:
                    
                        # Compute abstention mask
                        abstain_count = int(abstain_percentage * len(preds))
                        abstain = abstain_order(criteria, abstain_count)
                        non_abstain = (1 - abstain).astype(bool)

                        coverage = np.mean(non_abstain)*100

                        try:
                            # Compute abstain metrics
                            abstain_metrics = compute_abstain_metrics(abstain_percentage, preds, criteria, y_vec=y_train, train=True, yn_vec=yn_train)

                            # Store results
                            results.extend([
                                {"metric": metric, "value": abstain_metrics[metric], "coverage": coverage, "threshold": abstain_percentage,
                                 "noise": noise_level, "loss": loss, "draw_id": draw_id, "method": method}
                                for metric in metric_lis
                            ])
                            
                        except:
                            break
                        
                        print(len(test_preds), len(unanticipated_retrain), len(y_test))
                    # Compute abstain metrics
                        abstain_metrics = compute_abstain_metrics(abstain_percentage, test_preds, unanticipated_retrain, y_vec=y_test, train=False)

                        # Store results
                        results.extend([
                            {"metric": "risk", "value": abstain_metrics["risk"], "coverage": coverage, "threshold": abstain_percentage,
                             "noise": noise_level, "loss": loss, "draw_id": draw_id, "method": method}
                        ])


                        
    return pd.DataFrame(results)





def metrics_abstain_subgroup(dataset, model_type="LR", noise_type="class_conditional", misspecify="correct", fixed_class=0, fixed_noise=0.0):
    splits = []
    metrics = []
    values = []
    coverages = []
    thresholds = []
    noise_levels = []
    losses = []
    groups = []
    draw_ids = []
    methods = []
    

    # Load dataset splits
    X_train, X_test, y_train, y_test, group_train, group_test = load_dataset_splits(dataset, group="age")

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Load abstain data
    data = load_abstain(dataset, model_type, noise_type, misspecify=misspecify)
    df = pd.DataFrame(data)

    for noise_level in df.noise.unique():
        if noise_type == "class_independent":
            _, T = generate_class_independent_noise(y_train, noise_level)
        else:
            _, T = generate_class_conditional_noise(y_train, noise_level, fixed_class, fixed_noise)

        for loss in df.loss.unique():
            for draw_id in df.draw_id.unique():
                
                sub_df = df[(df["loss"] == loss) & (df["noise"] == noise_level) & (df["draw_id"] == draw_id)]
                
                for train in [True, False]:
                    metric_lis = ['risk','regret', 'fpr', 'fnr'] if train else ['risk']
                    
                    if train:
                        #ambiguity = np.clip(sub_df.ambiguity_train.values[0] / 100, 0, 1)
                        new = np.clip(sub_df.ambiguity_train.values[0] / 100, 0, 1)
                        probs = sub_df.train_probs.values[0]
                    
                        u_vec = get_u(y_train, T=T, seed=draw_id, noise_type=noise_type)
                        y_vec = y_train
                        yn_train = flip_labels(y_train, u_vec)  # XOR
                    else:
                        #ambiguity = np.clip(sub_df.ambiguity_test.values[0] / 100, 0, 1)
                        probs = sub_df.test_probs.values[0]
                        
                        y_vec = y_test

                    if probs.ndim == 2:
                        preds = np.argmax(probs, axis=1)
                        confidence = np.max(probs, axis=1)
                    else:
                        preds = (probs > 0.5).astype(int)
                        confidence = np.where(probs > 0.5, probs, 1 - probs)
                        
                    uncertainty = 1 - confidence

                    for method in ["ambiguity", "new", "confidence"]:
                        if method == "ambiguity":
                            criteria = ambiguity
                        elif method == "confidence":
                            criteria = uncertainty
                        elif method == "new":
                            criteria = new
                        
                        for abstain_percentage in np.linspace(0, 0.99, 100):
                            
                            # Compute group-level metrics
                            for group in ["age", "hiv"] if (dataset == "saps" or dataset == "saps_imbalanced") else ["age", "sex"]:
                                _, _, _, _, group_train, group_test = load_dataset_splits(dataset, group)
    
                                for group_val in np.unique(group_train):
                                    group_vec = group_train if train else group_test
                                    
                                    group_stats = compute_abstain_group_stats(abstain_percentage, group_val, group_vec, preds, criteria, y_vec, train=train, yn_vec=yn_train)
                                    
                                    for metric in metric_lis:
                                        if not train:
                                            metrics.append(metric+"_test")
                                        else:
                                            metrics.append(metric)
                                       
                                        values.append(group_stats[metric])
                                        
                                        coverages.append(group_stats['coverage'])
                                        thresholds.append(abstain_percentage)
                                        noise_levels.append(noise_level)
                                        losses.append(loss)
                                        groups.append(f"{group}_{group_val}")
                                        draw_ids.append(draw_id)
                                        methods.append(method)

                            for c in [0, 1]:
                                group_stats = compute_abstain_group_stats(abstain_percentage, c, y_vec, preds, criteria, y_vec, train=train, yn_vec=yn_train)
                                    
                                for metric in metric_lis:
                                
                                    if not train:
                                        metrics.append(metric+"_test")
                                    else:
                                        metrics.append(metric)

                                    values.append(group_stats[metric])
                                    coverages.append(group_stats['coverage'])
                                    thresholds.append(abstain_percentage)
                                    noise_levels.append(noise_level)
                                    losses.append(loss)
                                    groups.append(f"class_{c}")
                                    draw_ids.append(draw_id)
                                    methods.append(method)

    # Create a DataFrame from the arrays
    data = pd.DataFrame({
        'metric': metrics,
        'value': values,
        'coverage': coverages,
        'threshold': thresholds,
        'noise': noise_levels,
        'loss': losses,
        'group': groups,
        'draw_id': draw_ids,
        'method': methods
    })
    return data

       
       

def plot_metrics(data, loss_type="BCE", noise_level=0.2, group = False):

    data["abstention"] = 100-data["coverage"]
    
    # Define your custom color palette for each method
    method_colors = {
        "new": "#ce3d26",
        "unanticipated": "#8896FB",   # Purple
        "confidence": "#808080"  # Gray
    }
    
    # Define your custom color palette for each method
    loss_colors = {
        "BCE": "#8896FB",   
        "forward": "#fc8803",  
        "backward": "#4ed476"}

    # Set the font style to sans-serif
    plt.rcParams["font.family"] = "sans-serif"

    #metrics = ["regret", "fpr", "fnr", "clean_risk", "clean_risk_test"] #data.metric.unique()
    metrics = ["regret",  "clean_risk_test"] #data.metric.unique()

    # Plot total level metrics

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))  # Create a new figure with multiple columns

    sub_data = data[(data["loss"] == loss_type) & (data["noise"] == noise_level)]

    for metric, ax in zip(metrics, axes):
        for method in sub_data['method'].unique():

            method_data = sub_data[(sub_data['method'] == method)& (sub_data['metric'] == metric)]
            color = method_colors.get(method, "#808080")
            sns.lineplot(data=method_data, x="coverage", y="value", ax=ax, linewidth=1, color=color, linestyle='--', label=None)
            ax.scatter(method_data["coverage"].values, method_data["value"].values, color=color, s=5, alpha=0.5)

        ax.set_xlabel("Abstention Rate", fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.set_title(f"{metric} (Noise Level: {noise_level})")

        ax.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))  # Create a new figure with multiple columns

    sub_data = data[(data["method"] == "new") & (data["noise"] == noise_level)]

    for metric, ax in zip(metrics, axes):
        for loss_type in sub_data['loss'].unique():

            method_data = sub_data[(sub_data['loss'] == loss_type)& (sub_data['metric'] == metric)]
            color = loss_colors.get(loss_type, "#808080")
            sns.lineplot(data=method_data, x="abstention", y="value", ax=ax, linewidth=1, color=color, linestyle='--', label=None)
            ax.scatter(method_data["abstention"].values, method_data["value"].values, color=color, s=5, alpha=0.5)

        ax.set_xlabel("Abstention Rate", fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.set_title(f"{metric} (Noise Level: {noise_level})")

        ax.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()
    
    if group:
        # Plot group level metrics
        group_data = data[(data["loss"] == loss_type) & (data["noise"] == noise_level) &(data["method"] == "ambiguity")]

        for group in group_data["group"].str.split("_").str[0].unique():

            fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))  # Create a new figure with multiple columns

            sub_data = group_data[group_data["group"].str.contains(f"{group}_")]

            for metric, ax in zip(metrics, axes):
                for hue in sub_data.group.unique():
                    
                    method_data = sub_data[(sub_data['group'] == hue)& (sub_data['metric'] == metric)]
                    sns.lineplot(data=method_data, x="abstention", y="value", ax=ax, linewidth=1, label=hue)
                    ax.scatter(method_data["abstention"].values, method_data["value"].values, s=5, alpha=0.5)

                ax.set_xlabel("Abstention Rate", fontsize=14)
                ax.set_ylabel(metric, fontsize=14)
                ax.set_title(f"{metric} (Noise Level: {noise_level}, Group: {group})")

                ax.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5)
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.tick_params(axis='both', which='minor', labelsize=12)
                ax.legend(fontsize=12, title_fontsize=14)  # Add legend here

            plt.tight_layout(rect=[0, 0.1, 1, 1])
            plt.show()      
 

    data["abstention"] = 100-data["coverage"]
    
    # Define your custom color palette for each method
    method_colors = {
        "new": "#8896FB",   # Purple
        "confidence": "#808080"  # Gray
    }
    
    # Define your custom color palette for each method
    loss_colors = {
        "BCE": "#8896FB",   
        "forward": "#fc8803",  
        "backward": "#4ed476"}

    # Set the font style to sans-serif
    plt.rcParams["font.family"] = "sans-serif"

    metrics = ["regret", "fpr", "fnr", "clean_risk", "clean_risk_test"] #data.metric.unique()

    # Plot total level metrics

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))  # Create a new figure with multiple columns

    sub_data = data[(data["loss"] == loss_type) & (data["noise"] == noise_level)]

    for metric, ax in zip(metrics, axes):
        for method in sub_data['method'].unique():

            method_data = sub_data[(sub_data['method'] == method)& (sub_data['metric'] == metric)]
            color = method_colors.get(method, "#808080")
            sns.lineplot(data=method_data, x="abstention", y="value", ax=ax, linewidth=1, color=color, linestyle='--', label=None)
            ax.scatter(method_data["abstention"].values, method_data["value"].values, color=color, s=5, alpha=0.5)

        ax.set_xlabel("Abstention Rate", fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.set_title(f"{metric} (Noise Level: {noise_level})")

        ax.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))  # Create a new figure with multiple columns

    sub_data = data[(data["method"] == "ambiguity") & (data["noise"] == noise_level)]

    for metric, ax in zip(metrics, axes):
        for loss_type in sub_data['loss'].unique():

            method_data = sub_data[(sub_data['loss'] == loss_type)& (sub_data['metric'] == metric)]
            color = loss_colors.get(loss_type, "#808080")
            sns.lineplot(data=method_data, x="abstention", y="value", ax=ax, linewidth=1, color=color, linestyle='--', label=None)
            ax.scatter(method_data["abstention"].values, method_data["value"].values, color=color, s=5, alpha=0.5)

        ax.set_xlabel("Abstention Rate", fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.set_title(f"{metric} (Noise Level: {noise_level})")

        ax.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()
    
    if group:
        # Plot group level metrics
        group_data = data[(data["loss"] == loss_type) & (data["noise"] == noise_level) &(data["method"] == "ambiguity")]

        for group in group_data["group"].str.split("_").str[0].unique():

            fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))  # Create a new figure with multiple columns

            sub_data = group_data[group_data["group"].str.contains(f"{group}_")]

            for metric, ax in zip(metrics, axes):
                for hue in sub_data.group.unique():
                    
                    method_data = sub_data[(sub_data['group'] == hue)& (sub_data['metric'] == metric)]
                    sns.lineplot(data=method_data, x="abstention", y="value", ax=ax, linewidth=1, label=hue)
                    ax.scatter(method_data["abstention"].values, method_data["value"].values, s=5, alpha=0.5)

                ax.set_xlabel("Abstention Rate", fontsize=14)
                ax.set_ylabel(metric, fontsize=14)
                ax.set_title(f"{metric} (Noise Level: {noise_level}, Group: {group})")

                ax.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5)
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.tick_params(axis='both', which='minor', labelsize=12)
                ax.legend(fontsize=12, title_fontsize=14)  # Add legend here

            plt.tight_layout(rect=[0, 0.1, 1, 1])
            plt.show()