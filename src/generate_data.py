import sys
sys.path.insert(0,'..')


import numpy as np
import pickle as pkl

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, resample
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import os

from src.enhancer import *


def load_metrics(model_type, noise_type, uncertainty_type, metric, group = "age",  dataset = 'cshock_eicu', fixed_class = None, fixed_noise = None, epsilon = 0.1):
    
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    
    #metrics_dir = os.path.join(parent_dir, "results", "raw", "metrics", dataset, model_type, noise_type)
    parent_dir = "/scratch/hdd001/home/snagaraj/"
    metrics_dir = os.path.join(parent_dir, "results", "metrics", dataset, model_type, noise_type)

    # Prepare the data
    
    if uncertainty_type == "forward":
        loss_functions = ["BCE", "forward", "backward"]
    else:
        loss_functions = ["BCE"]

    
    if ("disagreement" in metric) or ("regret" in metric):
        
        _, _, y_train, y_test, group_train, group_test = load_dataset_splits(dataset, group = group)
        
        group_vec = group_test if "test" in metric else group_train
        label_vec = y_test if "test" in metric else y_test
        
        
    rows = []
    
    for loss_function in loss_functions:
        for file_name in os.listdir(metrics_dir):
            if file_name.endswith('.pkl') and uncertainty_type in file_name:
                path = os.path.join(metrics_dir, file_name)

                parts = file_name.split('_')
                noise = float(parts[1]) # Assumes file name format: {uncertainty}_{noise}_{epsilon}_metrics.pkl
                eps = float(parts[2])

                if eps != epsilon:
                    continue
                
                with open(path, 'rb') as file:
                    # The noise level and uncertainty type are inferred from the file name
                    
                    metrics = pkl.load(file)

                for m in metrics.data[loss_function].keys():
                    
                    if m == metric:
                        for i, value in enumerate(metrics.data[loss_function][metric]):
                            if ("disagreement" in metric) or ("regret" in metric):
                                rows.append({
                                    'Metric': m,
                                    'Noise Level (%)': round(noise * 100),  # Assume noise is a fraction
                                    'Rate (%)': value,
                                    'Loss Function': loss_function,
                                    'Index': i,
                                    "Class": label_vec[i],
                                    f"{group}": group_vec[i]})
                
                            else:
                                rows.append({
                                    'Metric': m,
                                    'Noise Level (%)': round(noise * 100),  # Assume noise is a fraction
                                    'Rate (%)': value,
                                    'Loss Function': loss_function,
                                    'Index': i })
                

    # Scan through all files in the directory for the model_type
    
                    
    return pd.DataFrame(rows)



def load_dataset(dataset, include_groups = True):
    
    #dataset = "cshock_eicu"

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    data_path = os.path.join(parent_dir, "data", dataset, dataset+"_data.csv")
    
    df = pd.read_csv(data_path)
    
    if dataset in ["cshock_eicu", "cshock_mimic"]:
        # Labels
        labels = df['hospital_mortality'].values
        
        # Encode 'sex' using LabelEncoder
        label_encoder = LabelEncoder()
        df['age'] = label_encoder.fit_transform(df['age'])
        df['sex'] = label_encoder.fit_transform(df['sex'])

        # Groups
        groups = {
            'age': df['age'].values,
            'sex': df['sex'].values
        }
        
        if include_groups:
            # Features including all except label
            features = pd.get_dummies(df.drop('hospital_mortality', axis=1)).values
        else:
            # Features including all except group and label
            features = pd.get_dummies(df.drop(['hospital_mortality', 'age', 'sex'], axis=1)).values


    elif dataset == "support":

        # Labels
        df['Death_in_5Yr'] =  (df['Death_in_5Yr'] + 1) // 2
        labels = df['Death_in_5Yr'].values.astype(int)
        
        # Encode 'sex' using LabelEncoder
        label_encoder = LabelEncoder()
        

        df['age'] = label_encoder.fit_transform(df['age'])
        df['sex'] = label_encoder.fit_transform(df['sex'])

        # Groups
        groups = {
            'age': df['age'].values,
            'sex': df['sex'].values
        }
        
        if include_groups:
            # Features including all except label
            features = pd.get_dummies(df.drop('Death_in_5Yr', axis=1)).values.astype(int)
        else:
            # Features including all except group and label
            features = pd.get_dummies(df.drop(['Death_in_5Yr', 'age', 'sex'], axis=1)).values.astype(int)
        
    elif dataset == "saps":
        # Labels
        labels = df['DeadAtDischarge'].values

        # Encode 'sex' using LabelEncoder
        label_encoder = LabelEncoder()
        

        df['age'] = label_encoder.fit_transform(df['Age'])
        df['hiv'] = label_encoder.fit_transform(df['HIVWithComplications'])
        
        # Groups
        groups = {
            'age': df["age"].values,
            'hiv': df["hiv"].values
            
        }

        if include_groups:
            # Features including all except label
            features = pd.get_dummies(df.drop('DeadAtDischarge', axis=1)).values.astype(int)
        else:
            # Features including all except group and label
            features = pd.get_dummies(df.drop(['DeadAtDischarge', 'age', 'hiv'], axis=1)).values.astype(int)
        
    elif dataset == "lungcancer":
        # Labels
        labels = df['Malignant'].values.astype(int)
        
        # Encode 'sex' using LabelEncoder
        label_encoder = LabelEncoder()
        df['Age'] = label_encoder.fit_transform(df['Age'])
        df['Gender'] = label_encoder.fit_transform(df['Gender'])

        # Groups
        groups = {
            'age': df['Age'].values,
            'sex': df['Gender'].values
        }
        
        if include_groups:
            # Features including all except label
            features = pd.get_dummies(df.drop('Malignant', axis=1)).values
        else:
            # Features including all except group and label
            features = pd.get_dummies(df.drop(['Malignant', 'Age', 'Gender'], axis=1)).values
            
            
    elif dataset == "enhancer":
        # Labels
        labels = df['Significant'].values

        # Encode 'sex' using LabelEncoder
        label_encoder = LabelEncoder()


        df['chr'] = label_encoder.fit_transform(df['chr'])

        # Groups
        groups = {
            'chr': df["chr"].values

        }

        feature_cols = list(df.columns[-12:])

        

        if include_groups:
            df_features = df[feature_cols + "chr"]
            
        else:
            df_features = df[feature_cols]
        features = pd.get_dummies(df_features).values.astype(int)


    # Create a StandardScaler object
    scaler = StandardScaler()

    # Fit and transform the data
    features = scaler.fit_transform(features)
    
    
            
    return features, labels, groups


def balance_data(features, labels, groups = None):
        # Suppose this function loads your dataset
    #features, labels, groups = load_dataset("cshock_eicu", include_groups=True)
    np.random.seed(2024)
    
    # Find the unique classes and the frequency of each class
    class_counts = np.bincount(labels)
    minority_class = np.argmin(class_counts)
    majority_class = np.argmax(class_counts)

    # Separate the majority and minority classes
    features_minority = features[labels == minority_class]
    features_majority = features[labels == majority_class]
    labels_minority = labels[labels == minority_class]
    labels_majority = labels[labels == majority_class]


    groups_minority = groups[labels == minority_class]
    groups_majority = groups[labels == majority_class]


    # Upsample the minority class
    features_minority_upsampled, labels_minority_upsampled, groups_minority_upsampled = resample(
        features_minority,
        labels_minority,
        groups_minority,
        replace=True,  # Sample with replacement
        n_samples=len(features_majority),  # Match number in majority class
        random_state=2024)  # Reproducible results

    # Combine the majority class with the upsampled minority class
    features_balanced = np.vstack((features_majority, features_minority_upsampled))
    labels_balanced = np.hstack((labels_majority, labels_minority_upsampled))
    
    groups_balanced = np.hstack((groups_majority, groups_minority_upsampled))
    features_balanced, labels_balanced, groups_balanced = shuffle(features_balanced, labels_balanced, groups_balanced, random_state=2024)
    return features_balanced, labels_balanced, groups_balanced
   


def load_MNIST(n_samples, random_state = 42):
    np.random.seed(random_state)
    # Load MNIST dataset
    mnist = fetch_openml(name='mnist_784', version=1)

    # Filter the dataset for digits 1 and 7
    mask = (mnist.target == '1') | (mnist.target == '7')
    X = mnist.data[mask]
    Y = mnist.target[mask]

    # Convert labels to binary: 1 for digit 1, and 0 for digit 7
    Y = (Y == '1').astype(int)

    # Select a random subset of the data
    subset_size = 1000  # for example, 500 samples
    subset_indices = np.random.choice(np.arange(X.shape[0]), size=subset_size, replace=False)
    X = X.iloc[subset_indices].values
    Y = Y.iloc[subset_indices].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, Y

def load_dataset_splits(dataset, group=""):

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))  


    filepath = os.path.join(parent_dir, "data", dataset , f"{dataset}_{group}_processed.pkl")
    with open(filepath, 'rb') as file:

        # Use pickle to write the dictionary to the file
        [X_train, X_test, y_train, y_test, group_train, group_test] = pkl.load(file)

    return X_train, X_test, y_train, y_test, group_train, group_test

def enhancer_train_test_split(features, labels, noisy_labels, effect, test_size=0.2, random_state=None):
    """
    Custom train-test split function that splits data at the group level.

    Args:
        features (numpy.ndarray or pandas.DataFrame): Feature matrix.
        labels (numpy.ndarray or pandas.Series): Labels vector.
        groups (numpy.ndarray or pandas.Series): Group vector (e.g., chromosomes).
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test, group_train, group_test
    """
    if random_state is not None:
        np.random.seed(random_state)

    features, labels, noisy_labels , power,  p_value, effect, groups = load_enhancer()

    # Initialize lists to hold train and test indices
    train_indices = []
    test_indices = []

    # Unique groups (e.g., unique chromosomes)
    unique_groups = np.unique(groups)

    for group in unique_groups:
        # Get indices for the current group
        group_indices = np.where(groups == group)[0]
        
        # Shuffle the indices
        np.random.shuffle(group_indices)
        
        # Determine split point
        split_point = int(len(group_indices) * (1 - test_size))
        
        # Assign indices to train and test
        train_indices.extend(group_indices[:split_point])
        test_indices.extend(group_indices[split_point:])

    # Split the data
    X_train, X_test = features[train_indices], features[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]
    yn_train, yn_test = noisy_labels[train_indices], noisy_labels[test_indices]
    effect_train, effect_test = effect[train_indices], effect[test_indices]
    group_train, group_test = groups[train_indices], groups[test_indices]

    return X_train, X_test, y_train, y_test, yn_train, yn_test, effect_train, effect_test, group_train, group_test



if __name__ == "__main__":

    datasets = ["cshock_eicu", "cshock_mimic"]

    random_state = 2024

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    for dataset in datasets:

        print("Loaded Data!")

        X, y, groups_dict = load_dataset(dataset, include_groups = True)

        
        for group in groups_dict.keys():
            #features, labels, groups = balance_data(X, y, saps = False, groups = groups_dict[group])
            features, labels, groups = X, y, groups_dict[group]
            if dataset == "enhancer":
                X_train, X_test, y_train, y_test, group_train, group_test = enhancer_train_test_split(features, 
                                                                labels, 
                                                                groups,
                                                                test_size=0.2, 
                                                                random_state=2024)
            else:
                
                X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(features, 
                                                                labels, 
                                                                groups,
                                                                test_size=0.2, 
                                                                random_state=2024)
            X_train = X_train
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
            
            filepath = os.path.join(parent_dir, "data", dataset , f"{dataset}_{group}_imbalanced_processed.pkl")

            with open(filepath, 'wb') as file:
                # Use pickle to write the dictionary to the file
                pkl.dump([X_train, X_test, y_train, y_test, group_train, group_test] , file)

            print("Saved Data!")

def metrics_to_df(metrics_dict, noise_level, seed):
    # Initialize an empty list to hold all the data rows
    data = []
    
    # Loop through each loss function in the dictionary
    for loss_func, metrics in metrics_dict.items():
        # Loop through each metric and its list of values
        for metric, values in metrics.items():
            # Loop through each value in the list
            for idx, value in enumerate(values):
                # Append a new row with the loss function, metric, index, and the value
                data.append({'Loss Function': loss_func, 'Noise Level (%)': int(noise_level*100), 'Metric': metric, 'Index': idx, 'Rate (%)': value, 'Seed': seed})
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    
    return df
