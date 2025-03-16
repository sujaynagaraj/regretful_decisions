import torch
import torch.nn as nn

import numpy as np

import timeit


def backward_loss(outputs, labels, T, device):
    #start_time = timeit.default_timer()

    softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss(reduction="none").to(device)
    
    # Prepare outputs
    outputs = outputs.to(device).to(dtype=torch.float32)
    labels = labels.to(device)

    # Use the provided single noise transition matrix
    T_inv = torch.linalg.inv(T)

    all_label_loss = []
    for c in range(outputs.shape[-1]):
        class_labels = torch.full(labels.shape, c, device=device).long()
        loss = criterion(outputs, class_labels)
        all_label_loss.append(loss)

    all_label_loss = torch.stack(all_label_loss, dim=-1).unsqueeze(-1)

    # Compute backward corrected loss
    backward_loss_values = torch.matmul(T_inv.float(), all_label_loss.float())
    backward_loss = backward_loss_values[range(labels.size(0)), labels.long(), 0].mean()

    # code you want to evaluate
    #elapsed = timeit.default_timer() - start_time
    #print("backward ", elapsed)

    return backward_loss


def forward_loss(outputs, labels, T, device, noise_type = "class_independent"):
    start_time = timeit.default_timer()

    softmax = nn.Softmax(dim=1)
    criterion = nn.NLLLoss(reduction="mean").to(device)
    
    # Prepare outputs and labels
    outputs = outputs.to(device).to(dtype=torch.float32)
    labels = labels.to(device)
    clean_posterior = softmax(outputs)

   
    T = T.to(device).to(dtype=torch.float32)
    

    T_T = torch.transpose(T, 0, 1)


    # Adjust the outputs based on the noise transition matrix
    noisy_posterior = torch.matmul(T_T, clean_posterior.unsqueeze(-1)).squeeze()
    
    # Calculate loss
    loss = criterion(noisy_posterior.log(), labels)
    
    return loss





def instance_backward_01loss(y_true, y_pred, T, threshold=0):
    """
    Calculate the noise-corrected 0-1 loss for a set of predictions.

    :param y_true: The observed (noisy) labels.
    :param y_pred: The predicted labels (0 or 1).
    :param T: The noise transition matrix.
    :param threshold: Threshold for converting the corrected loss to binary.
    :return: A tuple containing the noise-corrected 0-1 loss across the batch and the instance-level loss.
    """
    # Convert y_true and y_pred to numpy arrays if they are not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Convert y_pred from 0 and 1 to -1 and 1
    y_pred_converted = 2 * y_pred - 1

    # Calculate the inverse noise transition matrix
    T_inv = np.linalg.inv(T)

    # Initialize the instance-level loss array
    instance_losses = np.zeros(len(y_true))

    # Compute the 0-1 loss for each prediction
    for i in range(len(y_true)):
        for true_label in range(T.shape[0]):
            # Convert the true_label to -1 and 1 for comparison
            true_label_converted = 2 * true_label - 1
            # Compute the uncorrected 0-1 loss for the current assumed true label
            uncorrected_loss = 1 if y_pred_converted[i] != true_label_converted else 0
            # Weight the uncorrected loss by the probability of the true label given the observed label
            instance_losses[i] += T_inv[true_label, y_true[i]] * uncorrected_loss

    # Apply threshold to instance losses to convert to binary
    instance_losses_binary = (instance_losses > threshold).astype(int)
    # Calculate the batch loss as the average of the instance-level binary losses
    batch_loss_binary = np.mean(instance_losses_binary)

    return batch_loss_binary, instance_losses_binary


def instance_01loss(y_true, y_pred):
    """
    Calculate the mean 0-1 loss for a set of predictions and the instance-level loss.
    :param y_true: The true labels.
    :param y_pred: The predicted labels.
    :return: A tuple containing the mean 0-1 loss and the instance-level loss.
    """
    # Convert y_true and y_pred to numpy arrays if they are not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the instance-level loss
    #instance_losses = np.where(y_true == y_pred, 0, 1)
    instance_losses = np.not_equal(y_true, y_pred)
    
    # Calculate the batch loss as the mean of the instance-level losses
    batch_loss = np.mean(instance_losses)
    
    return batch_loss, instance_losses.astype(int)





def natarajan_unbiased_01_loss(y_true, y_pred, T):
    """
    Calculate the Natarajan Unbiased 0-1 Loss for a set of predictions using individual noise rates for each class.

    :param y_true: The observed (noisy) labels (numpy array of 0s and 1s).
    :param y_pred: The predicted labels (numpy array of 0s and 1s).
    :param rho_pos: The noise rate for the positive class.
    :param rho_neg: The noise rate for the negative class.
    :return: A tuple containing the mean Natarajan Unbiased 0-1 Loss and individual instance losses.
    """
    
    rho_neg = T[0,1]
    rho_pos = T[1,0]
    
    # Ensure y_true and y_pred are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Convert y_true and y_pred from 0 and 1 to -1 and 1
    y_true_converted = 2 * y_true - 1
    y_pred_converted = 2 * y_pred - 1
    
    # Initialize the corrected loss array
    instance_losses = np.zeros(len(y_true))
    
    # Calculate the denominator
    denom = 1 - rho_pos - rho_neg

    # Calculate the corrected loss for each instance
    for i in range(len(y_true)):
        true_label = y_true_converted[i]
        pred_label = y_pred_converted[i]
        
        # Define the 0-1 loss
        loss_true = 1 if pred_label != true_label else 0
        loss_incorrect = 1 if pred_label == true_label else 0

        # Apply the Natarajan correction formula
        corrected_loss = ((1 - rho_neg if true_label == 1 else 1 - rho_pos) * loss_true - 
                          (rho_pos if true_label == 1 else rho_neg) * loss_incorrect) / denom
        instance_losses[i] = corrected_loss

    batch_loss = np.mean(instance_losses)
    
    #instances_losses = np.clip(instances_losses, 0, 1)

    instance_losses_binary = np.clip(instance_losses, 0, 1)
    # Calculate the mean corrected loss
    batch_loss_binary = np.mean(instance_losses_binary)

    return batch_loss, instance_losses, batch_loss_binary, instance_losses_binary.astype(int)


def instance_forward_01loss(y_true, y_probs, T):
    """
    Calculate the noise-corrected 0-1 loss for a set of probabilistic predictions.
    
    :param y_true: The true labels.
    :param y_probs: The predicted probabilities.
    :param T: The noise transition matrix.
    :param threshold: The threshold for converting probabilities to binary predictions.
    :return: A tuple containing the mean 0-1 loss and the instance-level loss.
    """
    # Convert y_probs to numpy array if it is not already
    y_probs = np.array(y_probs)
    T_T = np.transpose(T)

    # Adjust the outputs based on the noise transition matrix using einsum for broadcasting
    noisy_posterior = np.einsum('ij,bj->bi', T_T, y_probs)

    # Convert probabilities to binary predictions
    y_pred = np.argmax(noisy_posterior, axis=1)

    
    # Calculate loss using the modified zero_one_loss function
    batch_loss, instance_losses = instance_01loss(y_true, y_pred)
    
    return batch_loss, instance_losses.astype(int)
