import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, log_loss
from tqdm.notebook import tqdm
import numpy as np

from src.loss_functions import *
from src.noise import *

from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import  LinearSVC
from sklearn.neural_network import MLPClassifier

import timeit


DEFAULT_LR_PARAMS =  {
        'fit_intercept': True,
        # 'intercept_scaling': 1.0,
        'class_weight': None,
        'penalty': None,
        # 'C': 1.0,
        'tol': 1e-4,
        'solver': 'saga',
        'warm_start': True,
        'max_iter': int(1e8),
        'verbose': False,
        'n_jobs': 1
        #'random_state':2024
        }

DEFAULT_SVM_PARAMS = {
        'fit_intercept': True,
        'intercept_scaling': 1.0,
        'class_weight': None,
        'loss': "hinge",
        'penalty': 'l2',
        'C': 1.0,
        'tol': 1e-4,
        'dual': True,
        #'random_state': None,
        'verbose': False
        #'random_state':2024
        }

DEFAULT_NN_PARAMS = {
        'solver': 'lbfgs',
        'alpha': 1e-5,
        'hidden_layer_sizes': (5, 2),
        #'random_state':2024,
        'max_iter': int(1e8)
        }


def train_model_ours(X_train, y_train, X_test, y_test, seed, model_type="LR"):
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Choose the model based on the input
    if model_type == "LR":
        model = LR(**DEFAULT_LR_PARAMS, random_state = seed)
    elif model_type == "SVM":
        model = LinearSVC(**DEFAULT_SVM_PARAMS, random_state = seed)
    elif model_type == "NN":
        model = MLPClassifier(**DEFAULT_NN_PARAMS, random_state = seed)
    else:
        raise ValueError("Unsupported model type. Choose 'LR' or 'SVM'.")

    # Train the model using noisy labels (simulating the impact of label noise)
    model.fit(X_train, y_train)

    # Predictions for training and test sets
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Calculate accuracies
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    if model_type == "SVM":
        train_probs = model.decision_function(X_train)
        test_probs = model.decision_function(X_test)
        
    else:
        train_probs = model.predict_proba(X_train)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
        
    # Calculate log losses
    train_loss = log_loss(y_train, train_probs)
    test_loss = log_loss(y_test, test_probs)
        
    results = (train_acc,
                test_acc,
                train_probs,
                test_probs,
                train_loss,
                test_loss,
                train_preds,
                test_preds
        )
        

    return model, results


# Define the Logistic Regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)

class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 5)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(5, 2)
        self.output = nn.Linear(2, 2)  # Assuming binary classification for simplicity

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output(x)
        return x

def train_model(X_train, y_train, yn_train, X_test, y_test, T,  seed, num_epochs=100, batch_size = 256, correction_type="forward", model_type = "LR"):
    # Check if GPU is available and set the default device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Convert to PyTorch tensors and move them to the device
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    yn_train = torch.tensor(yn_train, dtype=torch.long).to(device)

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    # Create DataLoader for mini-batch SGD
    train_data = TensorDataset(X_train, yn_train, y_train)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    if model_type == "LR":
        # Initialize the model and move it to the device
        model = LogisticRegression(X_train.shape[1]).to(device)
    else:
        # Initialize the model and move it to the device
        model = NeuralNet(X_train.shape[1]).to(device)

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    if correction_type in ['backward', 'forward']:
        #elapsed = timeit.default_timer() - start_time
        #T = torch.tensor(np.array([T_dict[g.item()] for g in group]), dtype=torch.float32).to(device)
        #print("T_dict ", elapsed)

        T = torch.tensor(T).to(device)

    # Train the model
    for epoch in (range(num_epochs)):
        for features, noisy_labels, clean_labels in train_loader:
            

            # Move features and labels to the device
            features, noisy_labels, clean_labels = features.to(device), noisy_labels.to(device), clean_labels.to(device)
            
            # Forward pass
            outputs = model(features)

            if correction_type == 'forward':
                noisy_loss = forward_loss(outputs, noisy_labels, T, device)

                #elapsed = timeit.default_timer() - start_time
                #print("Forward ", elapsed)

            elif correction_type == 'backward':
                noisy_loss = backward_loss(outputs, noisy_labels, T, device)

                #elapsed = timeit.default_timer() - start_time
                #print("Backward ", elapsed)

            else:
                noisy_loss = criterion(outputs, noisy_labels)
                
                #elapsed = timeit.default_timer() - start_time
                #print("BCE ", elapsed)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            noisy_loss.backward()
            optimizer.step()
    
    train_outputs = model(X_train)
    test_outputs = model(X_test)

    #Get final train losses
    if correction_type == 'forward':
        
        #T_train = torch.tensor(np.array([T_dict[g.item()] for g in group_train]), dtype=torch.float32).to(device)
        #T_test = torch.tensor(np.array([T_dict[g.item()] for g in group_test]), dtype=torch.float32).to(device)
        
        noisy_train_loss = forward_loss(train_outputs, yn_train, T, device).item()
        clean_train_loss = forward_loss(train_outputs, y_train, T, device).item()
        
        clean_test_loss = forward_loss(test_outputs, y_test, T, device).item()
        
    elif correction_type == 'backward':
        #T_train = torch.tensor(np.array([T_dict[g.item()] for g in group_train]), dtype=torch.float32).to(device)
        #T_test = torch.tensor(np.array([T_dict[g.item()] for g in group_test]), dtype=torch.float32).to(device)
        
        noisy_train_loss = backward_loss(train_outputs, yn_train, T, device).item()
        clean_train_loss = backward_loss(train_outputs, y_train, T, device).item()
        
        clean_test_loss = backward_loss(test_outputs, y_test, T, device).item()
    else:
        noisy_train_loss = criterion(train_outputs, yn_train).item()

        clean_train_loss = criterion(train_outputs, y_train).item()
        clean_test_loss = criterion(test_outputs, y_test).item()


    # Evaluate the model
    with torch.no_grad():

        _, predicted = torch.max(test_outputs.data, 1)
        # Move the predictions back to the CPU for sklearn accuracy calculation
        clean_test_acc = accuracy_score(y_test.cpu().numpy(), predicted.cpu().numpy())
        test_probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()

        _, predicted = torch.max(train_outputs.data, 1)
        # Move the predictions back to the CPU for sklearn accuracy calculation
        clean_train_acc = accuracy_score(y_train.cpu().numpy(), predicted.cpu().numpy())
        noisy_train_acc = accuracy_score(yn_train.cpu().numpy(), predicted.cpu().numpy())
        train_probs = torch.softmax(train_outputs, dim=1)[:, 1].cpu().numpy()

    results = (noisy_train_loss,
                clean_train_loss, 
                noisy_train_acc,
                clean_train_acc,
                train_probs,
                clean_test_loss, 
                clean_test_acc,
                test_probs
                )
    return model, results


def train_LR_no_test(X_train, y_train,  seed, num_epochs=50, correction_type="forward", noise_transition_matrix=None):
    # Check if GPU is available and set the default device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if correction_type in ['backward', 'forward']:
        # Define noise transition matrix (Example)
        # Convert it to the correct device
        noise_transition_matrix = torch.tensor(noise_transition_matrix, dtype=torch.float32).to(device)
    
    torch.manual_seed(seed)

    # Convert to PyTorch tensors and move them to the device
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    # Create DataLoader for mini-batch SGD
    batch_size = 256
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    # Initialize the model and move it to the device
    model = LogisticRegression(X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr = 0.01)

    # Train the model
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            # Move features and labels to the device
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            if correction_type == 'forward':
                loss = forward_loss(outputs, labels, noise_transition_matrix, device)
            elif correction_type == 'backward':
                loss = backward_loss(outputs, labels, noise_transition_matrix, device)
            else:
                loss = criterion(outputs, labels)
    
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

def get_predictions_LR(model, X_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Convert X_test to a torch tensor and move to the same device as the model
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Ensure the model is in evaluation mode
    model.eval()

    with torch.no_grad():  # Disable gradient calculations
        # Get the raw output from the model
        raw_output = model(X_test_tensor)
        
        # Apply sigmoid function to get probabilities
        probabilities = torch.sigmoid(raw_output)

        # Use argmax to get the class with the highest probability
        predictions = torch.argmax(raw_output, dim=1)

    return predictions



def train_LR_model_variance(X_train, y_train, X_test, y_test, num_models=1000, num_epochs = 1000, correction_type = "None", noise_transition_matrix = None):
    accuracies = []
    predicted_probabilities = []
    for seed in tqdm(range(num_models)):
        probabilities, accuracy, _ = train_LR(X_train, 
                                              y_train, 
                                              X_test, 
                                              y_test, 
                                              seed, 
                                              num_epochs = num_epochs, 
                                              correction_type = correction_type, 
                                              noise_transition_matrix = noise_transition_matrix)
        accuracies.append(accuracy)
        predicted_probabilities.append(probabilities)
    return np.array(predicted_probabilities), np.array(accuracies)

def train_LR_noise_variance(X_train, y_train, X_test, y_test, num_models=1000, num_epochs = 100, correction_type = "None", noise_transition_matrix = None):
    accuracies = []
    predicted_probabilities = []
    for seed in tqdm(range(num_models)):
        
        #VARY THE NP SEED FOR NOISE INJECTION
        np.random.seed(seed)
        
        if correction_type == "CLEAN": #Clean Labels
            probabilities, accuracy, _ = train_LR(X_train, 
                                                  y_train, 
                                                  X_test, 
                                                  y_test, 
                                                  seed=42, #FIX THE TORCH SEED
                                                  num_epochs = num_epochs, 
                                                  correction_type = correction_type, 
                                                  noise_transition_matrix = noise_transition_matrix)
        else: #Noisy Labels
            y_train_noisy = add_label_noise(y_train, noise_transition_matrix)
            probabilities, accuracy, _ = train_LR(X_train, 
                                                  y_train_noisy, 
                                                  X_test, 
                                                  y_test, 
                                                  seed=42, #FIX THE TORCH SEED
                                                  num_epochs = num_epochs, 
                                                  correction_type = correction_type, 
                                                  noise_transition_matrix = noise_transition_matrix)
            
        accuracies.append(accuracy)
        predicted_probabilities.append(probabilities)
    return np.array(predicted_probabilities), np.array(accuracies)


def train_model_ours_regret(X_train, y_train, X_test, y_test, seed, model_type="LR"):
    # Set random seed for reproducibility

    np.random.seed(seed)
    
    # Choose the model based on the input
    if model_type == "LR":
        model = LR(**DEFAULT_LR_PARAMS, random_state = seed)
    elif model_type == "SVM":
        model = LinearSVC(**DEFAULT_SVM_PARAMS, random_state = seed)
    elif model_type == "NN":
        model = MLPClassifier(**DEFAULT_NN_PARAMS, random_state = seed)
    else:
        raise ValueError("Unsupported model type. Choose 'LR' or 'SVM'.")

    # Train the model using noisy labels (simulating the impact of label noise)
    model.fit(X_train, y_train)

    # Predictions for training and test sets
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Calculate accuracies
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    if model_type == "SVM":
        train_probs = model.decision_function(X_train)
        test_probs = model.decision_function(X_test)
        
    else:
        train_probs = model.predict_proba(X_train)
        test_probs = model.predict_proba(X_test)
        
    # Calculate log losses
    train_loss = log_loss(y_train, train_probs)
    test_loss = log_loss(y_test, test_probs)
        

    results = (train_acc,
                test_acc,
                train_probs,
                test_probs,
                train_loss,
                test_loss,
                train_preds,
                test_preds)
        

    return model, results


def train_model_regret_torch(X_train, yn_train, y_train, X_test, y_test, T,  seed, num_epochs=100, batch_size = 256, correction_type="forward", model_type = "LR"):
    # Check if GPU is available and set the default device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Convert to PyTorch tensors and move them to the device
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    yn_train = torch.tensor(yn_train, dtype=torch.long).to(device)

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    # Create DataLoader for mini-batch SGD
    train_data = TensorDataset(X_train, yn_train)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    if model_type == "LR":
        # Initialize the model and move it to the device
        model = LogisticRegression(X_train.shape[1]).to(device)
    else:
        # Initialize the model and move it to the device
        model = NeuralNet(X_train.shape[1]).to(device)

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    if correction_type in ['backward', 'forward']:

        T = torch.tensor(T).to(device)

    # Train the model
    for epoch in (range(num_epochs)):
        for features, noisy_labels in train_loader:


            # Move features and labels to the device
            features, noisy_labels = features.to(device), noisy_labels.to(device)
            
            # Forward pass
            outputs = model(features)

            if correction_type == 'forward':
                noisy_loss = forward_loss(outputs, noisy_labels, T, device)

                #elapsed = timeit.default_timer() - start_time
                #print("Forward ", elapsed)

            elif correction_type == 'backward':
                noisy_loss = backward_loss(outputs, noisy_labels, T, device)

                #elapsed = timeit.default_timer() - start_time
                #print("Backward ", elapsed)

            else:
                noisy_loss = criterion(outputs, noisy_labels)
                
                #elapsed = timeit.default_timer() - start_time
                #print("BCE ", elapsed)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            noisy_loss.backward()
            optimizer.step()
    
    train_outputs = model(X_train)
    test_outputs = model(X_test)

    # Evaluate the model
    with torch.no_grad():
        
        _, train_preds = torch.max(train_outputs.data, 1)
        train_preds = train_preds.cpu().numpy()
        # Move the predictions back to the CPU for sklearn accuracy calculation
        train_probs = torch.softmax(train_outputs, dim=1).cpu().numpy()

        _, test_preds = torch.max(test_outputs.data, 1)
        test_preds = test_preds.cpu().numpy()
        # Move the predictions back to the CPU for sklearn accuracy calculation
        test_probs = torch.softmax(test_outputs, dim=1).cpu().numpy()


    results = (train_preds,
                test_preds,
                train_probs,
                test_probs
                )
    return model, results
