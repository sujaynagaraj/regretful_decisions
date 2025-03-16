import numpy as np
from src.noise import *


def get_uncertainty(m, max_iter, preds, yn_train, p_y_x_dict,group_train = None, group_test = None, noise_type="class_independent", model_type="LR", T=None, epsilon=0.25, misspecify=False):
    
    typical_count = 0
    y_vec = yn_train
    all_plausible_labels = []
    
    for seed in (range(1, max_iter+1)):
        u_vec = infer_u(y_vec, group=group_train, noise_type=noise_type, p_y_x_dict=p_y_x_dict, T=T, seed=seed)
        typical_flag, _ = is_typical(u_vec, p_y_x_dict, group=group_train, T=T, y_vec=y_vec, noise_type=noise_type, uncertainty_type="backward", epsilon=epsilon)
        
        if misspecify or noise_type == "group":
            typical_flag = True
            
        if not typical_flag:
            continue
            
        flipped_labels = flip_labels(y_vec, u_vec)
        all_plausible_labels.append(flipped_labels)

        typical_count += 1

        if typical_count == m:
            break
    
    all_plausible_labels = np.array(all_plausible_labels)  # Shape: (k, n)
    
    # Calculate Actual Mistake as a vector of mean values for each instance
    actual_mistakes = np.mean(preds != all_plausible_labels, axis=0)  # Shape: (n,)
    

    # Calculate Unanticipated Mistake as a vector of mean values for each instance
    # Expand preds and yn_train to match dimensions for comparison
    preds_expanded = np.expand_dims(preds, axis=0)  # Shape: (1, n)
    yn_train_expanded = np.expand_dims(yn_train, axis=0)  # Shape: (1, n)

    # Case 1: pred == yn_train but pred != all_plausible_labels
    case_1 = (preds_expanded == yn_train_expanded) & (preds_expanded != all_plausible_labels)
    
    # Case 2: pred != yn_train but pred == all_plausible_labels
    case_2 = (preds_expanded != yn_train_expanded) & (preds_expanded == all_plausible_labels)
    
    # Calculate mean unanticipated mistakes for each instance
    unanticipated_mistakes = np.mean(case_1 | case_2, axis=0)  # Shape: (n,)

    return actual_mistakes, unanticipated_mistakes  
    
def calculate_unanticipated(preds, flipped_labels, yn):
    error_clean = (preds != flipped_labels)
    correct_noisy = (preds == yn)

    error_noisy = preds != yn
    correct_clean = preds == flipped_labels

    unanticipated_mistake = ((error_clean) & (correct_noisy))+ ((correct_clean) & (error_noisy))

    return unanticipated_mistake


def calculate_robustness_rate(predicted_probabilities, epsilon=0.01):
    # Calculate the variance of the predicted probabilities across models for each sample
    variance = np.var(predicted_probabilities, axis=0)
    
    # Determine if the variance is below the threshold (epsilon^2)
    #is_robust = variance[:, 1] < epsilon ** 2
    is_robust = variance < epsilon ** 2
    
    # Calculate the robustness rate
    robustness_rate = np.mean(is_robust)
    return robustness_rate

def calculate_ambiguity(preds, labels):
    """
    Calculate ambiguity for each instance based on model disagreement.

    Args:
        train_preds (numpy.ndarray): Array of shape (m, k) where m is the number of models and k is the number of instances.
        plausible_labels (numpy.ndarray): Array of shape (m, k) representing plausible labels for each model-instance pair.

    Returns:
        numpy.ndarray: A (k,) array representing ambiguity scores for each instance.
    """
    # Compute disagreement: 1 if prediction disagrees with plausible label, 0 otherwise
    disagreement = (preds != labels).astype(int)
    
    # Compute ambiguity as the fraction of models that disagree per instance
    ambiguity = np.mean(disagreement, axis=0)

    return ambiguity


def estimate_disagreement(predicted_probabilities):
    # Number of models
    m = predicted_probabilities.shape[0]

    predictions = (predicted_probabilities > 0.5).astype(int)
    
    # Calculate the sample mean of predictions for each example
    p_hat = np.mean(predictions, axis=0)
    
    # Calculate the unbiased estimator for disagreement for each example
    mu_hat = 4* (m / (m - 1)) * p_hat * (1 - p_hat)
    
    return mu_hat*100


def disagreement_percentage(disagreement_rates, threshold):
    """
    Calculates the percentage of examples with a disagreement rate greater than threshold.

    :param disagreement_rates: Array or list of disagreement rates.
    :return: Percentage of examples with disagreement rate > threshold
    """
    num_examples = len(disagreement_rates)
    num_high_disagreement = sum(rate >= threshold for rate in disagreement_rates)

    percentage = (num_high_disagreement / num_examples)
    return percentage


def calculate_error_rate(predictions, labels):
    """
    Calculate the error rate per example in predictions across models.
    
    :param predictions: A 2D numpy array where each row contains predictions from a different model.
    :param labels: A 1D numpy array containing the true labels for the examples.
    :return: The error rate per example and the overall error rate.
    """
     # Ensure predictions is a numpy array
    predictions = np.asarray(predictions)

    # Ensure labels is a numpy array
    labels = np.asarray(labels)

    # Count the number of models
    num_models = predictions.shape[0]

    # Calculate the number of models where prediction != label for each example
    incorrect_predictions = predictions != labels.reshape(1, -1)

    # Calculate the error rate per example
    error_rate_per_example = np.mean(incorrect_predictions, axis=0)


    return error_rate_per_example * 100



def iou(array1, array2):
    """
    Compute the Intersection over Union (IoU) based on common indices in two arrays.
    
    Parameters:
    - array1 (np.ndarray): First array of indices.
    - array2 (np.ndarray): Second array of indices.
    
    Returns:
    - float: The IoU of the two arrays.
    """
    # Convert arrays to sets
    set1 = set(array1)
    set2 = set(array2)

    # Calculate intersection and union
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # Compute the IoU
    if len(union) == 0:
        return 0  # To handle the case where there is no union
    else:
        iou = len(intersection) / len(union)
        return iou

def dice(array1, array2):
    """
    Compute the Dice Coefficient of two lists.
    
    Parameters:
    - list1 (list): First list of items.
    - list2 (list): Second list of items.
    
    Returns:
    - float: The Dice Coefficient of the two lists.
    """
    set1 = set(array1)
    set2 = set(array2)
    
    intersection = len(set1.intersection(set2))
    dice = (2 * intersection) / (len(set1) + len(set2))
    return dice

def indices_above_threshold(data, threshold):
    """
    Return the indices of elements in a list or numpy array that are above a given threshold.
    
    Parameters:
    - data (list or np.ndarray): The list or numpy array to search.
    - threshold (float or int): The threshold above which indices should be returned.
    
    Returns:
    - np.ndarray: An array of indices where the corresponding elements are above the threshold.
    """
    # Convert data to a numpy array if it's not already one
    data_array = np.array(data)
    # Find indices where the condition is true
    indices = np.where(data_array >= threshold)[0]

    return indices

def compute_overlap(array1, array2, threshold, overlap_metric="dice"):
    """
    Compute the overlap metric (IoU or Dice) between two arrays after filtering based on a threshold.
    
    Parameters:
    - array1 (list or np.ndarray): First input array.
    - array2 (list or np.ndarray): Second input array.
    - threshold (float): Threshold to filter elements by.
    - metric (str): Metric to compute ('iou' or 'dice').
    
    Returns:
    - float: Computed metric (IoU or Dice) based on the sets of indices above the threshold.
    """
    # Get indices above threshold for both arrays

    indices1 = set(indices_above_threshold(array1, threshold))
    indices2 = set(indices_above_threshold(array2, threshold))
    
    # Calculate the metric based on the selected type
    if overlap_metric.lower() == 'iou':
        return iou(indices1, indices2)
    elif overlap_metric.lower() == 'dice':
        return dice(indices1, indices2)
    else:
        raise ValueError("Unsupported metric specified. Use 'iou' or 'dice'.")


def regret_FPR_FNR(err_true, err_anticipated):
    # Ensure the input arrays are numpy arrays
    err_true = np.array(err_true)
    err_anticipated = np.array(err_anticipated)

    # Calculate False Positives (FP) and False Negatives (FN)
    false_positives = np.where((err_anticipated == 1) & (err_true == 0))[0]
    false_negatives = np.where((err_anticipated == 0) & (err_true == 1))[0]

    # Calculate the rates with checks for division by zero
    #fp_rate = len(false_positives) / (len(false_positives) + len(false_negatives)) if (len(false_positives) + len(false_negatives))  > 0 else 0.0
    #fn_rate = len(false_negatives) / (len(false_positives) + len(false_negatives))  if (len(false_positives) + len(false_negatives))  > 0 else 0.0

    fp_rate = len(false_positives) / len(err_true)
    fn_rate = len(false_negatives) / len(err_true)
    
    return fp_rate, false_positives, fn_rate, false_negatives


import numpy as np

def compute_majority_vote(probs_test):
    """
    Compute uncertainty based on majority vote disagreement.

    Args:
        probs_test (numpy.ndarray): Shape (m, k) where m = number of models, k = number of instances.

    Returns:
        numpy.ndarray: Uncertainty scores for each instance (shape: (k,))
    """
    m, k = probs_test.shape  # Number of models, Number of instances

    # Convert probabilities to binary predictions (1 if >0.5, else 0)
    binary_preds = (probs_test > 0.5).astype(int)  # Shape: (m, k)

    # Count votes for class 1
    vote_count = np.sum(binary_preds, axis=0)  # Shape: (k,)

    # Compute absolute distance from perfect uncertainty (m/2)
    distance_from_half = np.abs(vote_count - (m / 2))

    # Normalize confidence: 1 when all agree, 0 when split evenly
    confidence = (2 * distance_from_half) / m

    # Compute uncertainty: 1 - confidence
    uncertainty_test = 1 - confidence

    return uncertainty_test


def compute_loo(probs_test):
    """
    Compute ambiguity by randomly selecting one model's probabilities for all instances 
    and measuring disagreement across other models.

    Args:
        probs_test (numpy.ndarray): Shape (m, k), where:
            - m = number of models
            - k = number of instances

    Returns:
        numpy.ndarray: Ambiguity scores for each instance (shape: (k,))
    """
    m, k = probs_test.shape  # Number of models, Number of instances

    # Convert probabilities to binary predictions (1 if >0.5, else 0)
    binary_preds = (probs_test > 0.5).astype(int)  # Shape: (m, k)

    # Randomly select ONE model (row) for all instances
    selected_model_idx = np.random.randint(0, m)  # Choose one model index
    selected_preds = binary_preds[selected_model_idx]  # Shape: (k,)

    # Compute disagreement: Fraction of models that disagree with selected model
    disagreement = (binary_preds != selected_preds)  # Boolean matrix (m, k)

    # Compute ambiguity: Fraction of models that disagree per instance
    ambiguity_scores = np.mean(disagreement, axis=0)  # Shape: (k,)

    return ambiguity_scores



def conformal_prediction_plausible(plausible_probs, train_probs, plausible_labels, alpha=0.1):
    """
    Compute conformal prediction sets per model and per instance, along with ambiguity scores.

    Args:
        plausible_probs (numpy.ndarray): Probabilities of plausible models for test set (m, k).
        train_probs (numpy.ndarray): Probabilities of plausible models for calibration set (m, n).
        plausible_labels (numpy.ndarray): Labels assigned by each model for the calibration set (m, n).
        alpha (float): Confidence level (e.g., 0.1 for 90% confidence).

    Returns:
        numpy.ndarray: Confidence scores for test instances (shape: (k,)).
        numpy.ndarray: Prediction sets per model and per test instance (shape: (m, k)).
        numpy.ndarray: Ambiguity scores (fraction of models where prediction set contains both `0` and `1`).
    """
    m, n = train_probs.shape  # Number of models, calibration instances
    k = plausible_probs.shape[1]  # Number of test instances

    # 1: Compute conformal scores for each model based on plausible labels
    cal_scores = 1 - np.where(plausible_labels == 1, train_probs, 1 - train_probs)  # Shape (m, n)

    # 2: Compute quantile threshold q_hat per model
    q_level = np.ceil((n + 1) * (1 - alpha)) / n  # Adjusted quantile level
    qhat = np.quantile(cal_scores, q_level, axis=1, method="higher")  # Shape: (m,)

    # 3: Compute test set probabilities per model
    test_probs = plausible_probs  # Shape (m, k)

    # 4: Compute prediction sets per model and per test instance
    prediction_sets = np.empty((m, k), dtype=object)  # Shape (m, k), store sets

    for model_idx in range(m):
        for instance_idx in range(k):
            prediction = set()
            if test_probs[model_idx, instance_idx] >= (1 - qhat[model_idx]):
                prediction.add(1)  # Include class 1
            if (1 - test_probs[model_idx, instance_idx]) >= (1 - qhat[model_idx]):
                prediction.add(0)  # Include class 0
            prediction_sets[model_idx, instance_idx] = prediction

    # 5: Compute ambiguity scores (fraction of models where prediction set contains {0,1})
    ambiguity_scores = np.mean(
        [[{0,1} == prediction_sets[m, i] for m in range(m)] for i in range(k)], axis=1
    )  # Shape (k,)

    # 6: Compute confidence scores as the mean probability per test instance
    confidence_scores = np.mean(test_probs, axis=0)  # Shape (k,)

    return confidence_scores, prediction_sets, ambiguity_scores

from scipy.stats import entropy

def compute_entropy_score(probs_test):
    """
    Compute uncertainty using binary entropy.

    Args:
        probs_test (numpy.ndarray): Shape (m, k), where each value is P(y=1|X).

    Returns:
        numpy.ndarray: Entropy scores for each instance (shape: (k,))
    """
    mean_probs = np.mean(probs_test, axis=0)  # Average probability across models
    entropy_scores = - (mean_probs * np.log(mean_probs + 1e-10) + (1 - mean_probs) * np.log(1 - mean_probs + 1e-10))
    return entropy_scores