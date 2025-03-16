import sys
sys.path.insert(0,'..')

from src.generate_data import *

from src.models import *
from src.loss_functions import *
from src.noise import *
from src.metrics import *
from src.plotting import *
from src.toy_data import *

import sklearn

import pandas as pd

from sklearn.metrics import roc_curve, auc, jaccard_score
from scipy.stats import bernoulli
from operator import xor


def load_dataset_splits(dataset, group=""):

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))  


    filepath = os.path.join(parent_dir, "data", dataset , f"{dataset}_{group}_processed.pkl")
    with open(filepath, 'rb') as file:

        # Use pickle to write the dictionary to the file
        [X_train, X_test, y_train, y_test, group_train, group_test] = pkl.load(file)

    return X_train, X_test, y_train, y_test, group_train, group_test


def run_procedure(m, max_iter, X_train, yn_train, X_test, y_test, p_y_x_dict, 
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
    plausible_labels = []

    y_vec = yn_train

    for seed in tqdm(range(1, min(m, max_iter) + 1)):  # Ensure we don't exceed both `m` and `max_iter`
        
        # Infer noisy label distribution
        u_vec = infer_u(y_vec, group=group_train, noise_type=noise_type, 
                        p_y_x_dict=p_y_x_dict, T=T, seed=seed)

        # Check if noise follows a typical pattern
        if not misspecify and noise_type != "group":
            typical_flag, _ = is_typical(u_vec, p_y_x_dict, group=group_train,  
                                         T=T, y_vec=y_vec, noise_type=noise_type, 
                                         uncertainty_type="backward", epsilon=epsilon)
        else:
            typical_flag = True  # Default to True when misspecified or using "group" noise
        
        # Flip labels based on noise model
        flipped_labels = flip_labels(y_vec, u_vec)

        # Train model with noisy labels
        model, metrics = train_model_ours(X_train, flipped_labels, X_test, y_test, 
                                          seed=2024, model_type=model_type)
        
        # Unpack metrics
        train_acc, test_acc, train_probs, test_probs, train_loss, test_loss, train_preds, test_preds = metrics

        # Store results
        preds_test.append(test_preds)
        preds_train.append(train_preds)
        probs_test.append(test_probs)
        probs_train.append(train_probs)
        plausible_labels.append(flipped_labels)

        # Break when `m` models are trained
        if len(preds_train) >= m:
            break

    return (
        np.array(plausible_labels), 
        np.array(preds_train), 
        np.array(preds_test), 
        np.array(probs_train), 
        np.array(probs_test)
    )

# def run_procedure(m, max_iter, X_train, yn_train, X_test, y_test, p_y_x_dict, group_train = None, group_test = None, noise_type = "class_independent", model_type = "LR", T = None, epsilon = 0.1, misspecify = False):
    
#     typical_count = 0
#     preds_test = []
#     preds_train = []
#     errors_clean_train = []
#     errors_test = []
#     plausible_test_probs = []
    
#     y_vec = yn_train
    
#     for seed in tqdm(range(1, max_iter+1)):
        
#         u_vec = infer_u(y_vec, group = group_train, noise_type = noise_type, p_y_x_dict = p_y_x_dict,  T = T , seed=seed)
        
        
#         if misspecify or noise_type == "group":
#             typical_flag = True
            
#         else:
#             typical_flag, _ = is_typical(u_vec, p_y_x_dict, group = group_train,  T = T, y_vec = y_vec, noise_type = noise_type, uncertainty_type = "backward", epsilon = epsilon)
        
            
#         flipped_labels = flip_labels(y_vec, u_vec)
        
#         model,  (train_acc,
#                 test_acc,
#                 train_probs,
#                 test_probs,
#                 train_loss,
#                 test_loss,
#                 train_preds,
#                 test_preds
#                 ) = train_model_ours(X_train, flipped_labels, X_test, y_test, seed = 2024, model_type=model_type)
        

#         preds_test.append(test_preds)
#         preds_train.append(train_preds)

#         error_clean_train = train_preds != flipped_labels
#         error_test = test_preds != y_test
       
#         #unanticipated_mistake = calculate_unanticipated(train_preds, flipped_labels, y_vec)
        
#         errors_test.append(error_test)
#         errors_clean_train.append(error_clean_train)
#         #unanticipated_mistakes.append(unanticipated_mistake)

#         plausible_test_probs.append(test_probs)

#         typical_count += 1

#         if typical_count == m:
#             break
            
#     predictions_test = np.array(preds_test)
#     ambiguity_test = np.mean(errors_test, axis=0)*100

#     predictions_train = np.array(preds_train)
#     ambiguity_train = np.mean(errors_clean_train, axis=0)*100

#     plausible_prob = np.mean(plausible_test_probs, axis = 0)*100
#     #print(np.array(plausible_test_probs).shape)
#     disagreement_test = estimate_disagreement(np.array(plausible_test_probs))
    
#     return ambiguity_train, ambiguity_test, plausible_prob, disagreement_test
  
class Metrics:
    def __init__(self):
        self.metrics = {}

    def add_metric(self,method, draw_id, metric_name, value):
        if method not in self.metrics:
            self.metrics[method] = {}
        if draw_id not in self.metrics[method]:
            self.metrics[method][draw_id] = {}
        self.metrics[method][draw_id][metric_name] = value

    def get_metric(self, method, draw_id, metric_name):
        return self.metrics.get(method, {}).get(draw_id, {}).get(metric_name, None)

    def get_all_metrics(self, method, draw_id):
        return self.metrics.get(method, {}).get(draw_id, {})

class Vectors:
    def __init__(self):
        self.vectors = {}

    def add_vector(self, method, draw_id, vector_name, value):
        if method not in self.vectors:
            self.vectors[method] = {}
        if draw_id not in self.vectors[method]:
            self.vectors[method][draw_id] = {}
        self.vectors[method][draw_id][vector_name] = value

    def get_vector(self, method, draw_id, vector_name):
        return self.vectors.get(method, {}).get(draw_id, {}).get(vector_name, None)

    def get_all_vectors(self, method, draw_id):
        return self.vectors.get(method, {}).get(draw_id, {})



def run_experiment(dataset, noise_type, model_type, n_models, max_iter, T, training_loss="None", n_draws=5, batch_size=256):
    X_train, X_test, y_train, y_test, group_train, group_test = load_dataset_splits(dataset, group="age")

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    p_y_x_dict = calculate_prior(y_train, noise_type=noise_type, group = group_train)  # Clean prior

    vectors = Vectors()

    for draw_id in range(n_draws):
        u_vec = get_u(y_train, T=T, seed=draw_id, noise_type=noise_type)
        yn_train = flip_labels(y_train, u_vec)

        model, (train_preds, test_preds,
                train_probs, test_probs) = train_model_regret_torch(
            X_train, yn_train, y_train, X_test, y_test, T,
            seed=2024, num_epochs=50, batch_size=batch_size, correction_type=training_loss, model_type=model_type)

        # True Population Error
        pop_err_true_train, instance_err_true_train = instance_01loss(y_train, train_preds)
        pop_err_true_test, instance_err_true_test = instance_01loss(y_test, test_preds)


        (ambiguity_train, 
        ambiguity_test,
        unanticipated_mistake_val) = run_procedure(n_models, 
                                            max_iter, 
                                            X_train, 
                                            yn_train, 
                                            X_test, 
                                            y_test, 
                                            p_y_x_dict, 
                                            group_train = None, 
                                            group_test = None, 
                                            noise_type = noise_type, 
                                            model_type = model_type, 
                                            T = T, 
                                            epsilon = 0.1, 
                                            misspecify = "correct")
        
        vectors.add_vector("metadata", draw_id, "dataset", dataset)
        vectors.add_vector("metadata", draw_id, "noise_type", noise_type)
        vectors.add_vector("metadata", draw_id, "model_type", model_type)
        vectors.add_vector("metadata", draw_id, "n_models", n_models)
        vectors.add_vector("metadata", draw_id, "max_iter", max_iter)
        vectors.add_vector("metadata", draw_id, "training_loss", training_loss)
        vectors.add_vector("metadata", draw_id, "n_draws", n_draws)
        vectors.add_vector("metadata", draw_id, "T", T)
        vectors.add_vector("metadata", draw_id, "y_train", y_train)
        vectors.add_vector("metadata", draw_id, "y_test", y_test)
        vectors.add_vector("metadata", draw_id, "train_preds", train_preds)
        vectors.add_vector("metadata", draw_id, "train_probs", train_probs)
        vectors.add_vector("metadata", draw_id, "test_preds", test_preds)
        vectors.add_vector("metadata", draw_id, "test_probs", test_probs)
        vectors.add_vector("metadata", draw_id, "yn_train", yn_train)
        vectors.add_vector("metadata", draw_id, "instance_err_true_train", instance_err_true_train)
        vectors.add_vector("metadata", draw_id, "instance_err_true_test", instance_err_true_test)
        vectors.add_vector("metadata", draw_id, "train_ambiguity", ambiguity_train)
        vectors.add_vector("metadata", draw_id, "test_ambiguity", ambiguity_test)
        vectors.add_vector("metadata", draw_id, "train_unanticipated", unanticipated_mistake_val)

    return vectors


def compute_group_stats(val, group_train, group_test, instance_err_true_train, instance_err_true_test, instance_err_pred_train, instance_err_pred_test):
    """
    Computes false positive rate (FPR), false negative rate (FNR), and regret rate for a given group value.

    Parameters:
    - val: The group value to compute statistics for.
    - group_train: Group labels for the training set.
    - group_test: Group labels for the test set.
    - instance_err_true_train: True instance errors for the training set.
    - instance_err_true_test: True instance errors for the test set.
    - instance_err_pred_train: Predicted instance errors for the training set.
    - instance_err_pred_test: Predicted instance errors for the test set.

    Returns:
    - Dictionary containing FPR, FNR, and regret rates for the specified group.
    """
    group_train = np.asarray(group_train)
    group_test = np.asarray(group_test)
    instance_err_true_train = np.asarray(instance_err_true_train)
    instance_err_true_test = np.asarray(instance_err_true_test)
    instance_err_pred_train = np.asarray(instance_err_pred_train)
    instance_err_pred_test = np.asarray(instance_err_pred_test)
    
    def calculate_rates(group, instance_err_true, instance_err_pred):
        group_indices = np.where(group == val)[0]
        instance_err_true_group = instance_err_true[group_indices]
        instance_err_pred_group = instance_err_pred[group_indices]
        
        false_positives = np.sum((instance_err_pred_group == 1) & (instance_err_true_group == 0))
        false_negatives = np.sum((instance_err_pred_group == 0) & (instance_err_true_group == 1))
        
        total_in_group = len(group_indices)
        
        fpr = false_positives / total_in_group if total_in_group > 0 else 0.0
        fnr = false_negatives / total_in_group if total_in_group > 0 else 0.0
        
        return fpr, fnr
    
    def calculate_regret_rate(group, instance_err_true, instance_err_pred):
        regret = abs(instance_err_true - instance_err_pred)
        group_indices = np.where(group == val)[0]
        regret_in_group = np.sum(regret[group_indices])
        total_in_group = len(group_indices)
        return regret_in_group / total_in_group if total_in_group > 0 else 0.0
    
    def calculate_risk_rate(group, instance_err_true):
        group_indices = np.where(group == val)[0]
        risk_in_group = np.sum(instance_err_true[group_indices])
        total_in_group = len(group_indices)
        return risk_in_group / total_in_group if total_in_group > 0 else 0.0
    
    # Calculate rates for training and test sets
    fpr_train, fnr_train = calculate_rates(group_train, instance_err_true_train, instance_err_pred_train)
    fpr_test, fnr_test = calculate_rates(group_test, instance_err_true_test, instance_err_pred_test)
    
    # Calculate regret rates
    regret_train = calculate_regret_rate(group_train, instance_err_true_train, instance_err_pred_train)
    regret_test = calculate_regret_rate(group_test, instance_err_true_test, instance_err_pred_test)
    
    risk_train = calculate_risk_rate(group_train, instance_err_true_train)
    risk_test = calculate_risk_rate(group_test, instance_err_true_test)
    
    return {
        'fpr_train': fpr_train,
        'fnr_train': fnr_train,
        'fpr_test': fpr_test,
        'fnr_test': fnr_test,
        'regret_train': regret_train,
        'regret_test': regret_test,
        'risk_train': risk_train,
        'risk_test': risk_test
    }

class MetricsCalculator:
    def __init__(self, vectors):
        """
        Initializes the MetricsCalculator with vectors and a Metrics object.

        Parameters:
        - vectors: A data structure to fetch vectors related to the metrics.
        """
        self.vectors = vectors
        self.metrics = Metrics()

    def calculate_metrics(self, draw_id):
        """
        Calculate and add various metrics to the Metrics object for a given draw_id.

        Parameters:
        - draw_id: The ID of the current draw for which metrics are calculated.
        """
        instance_err_true_train = self.vectors.get_vector("metadata", draw_id, "instance_err_true_train") #True 01 Error Train
        instance_err_true_test = self.vectors.get_vector("metadata", draw_id, "instance_err_true_test") #True 01 Error Test
        
        dataset = self.vectors.get_vector("metadata", draw_id, "dataset")
        noise_type = self.vectors.get_vector("metadata", draw_id, "noise_type")
        model_type = self.vectors.get_vector("metadata", draw_id, "model_type")
        T = self.vectors.get_vector("metadata", draw_id, "T")
        yn_train = self.vectors.get_vector("metadata", draw_id, "yn_train")
        train_preds = self.vectors.get_vector("metadata", draw_id, "train_preds")
        test_preds = self.vectors.get_vector("metadata", draw_id, "test_preds")
        train_probs = self.vectors.get_vector("metadata", draw_id, "train_probs")
        test_probs = self.vectors.get_vector("metadata", draw_id, "test_probs")
        y_train = self.vectors.get_vector("metadata", draw_id, "y_train")
        y_test = self.vectors.get_vector("metadata", draw_id, "y_test")
        ambiguity_train = self.vectors.get_vector("metadata", draw_id, "train_ambiguity")/100
        
        self.metrics.add_metric("metadata", draw_id, "ambiguity_train_vector", ambiguity_train)

        #print(self.vectors.get_all_vectors("metadata", draw_id))
        
        X_train, X_test, y_train, y_test, group_train, group_test = load_dataset_splits(dataset, group = "age")

        n_models = 100
        max_iter = 1000
        p_y_x_dict =  calculate_prior(y_train, 
                                      noise_type = noise_type, 
                                      group=group_train) #Clean prior
        
        posterior = calculate_posterior(yn_train, T, p_y_x_dict[0])
        
        susceptibility_vector_posterior = (posterior > 0).astype(int)
        
        noise_rates = np.where(y_train == 0, T[0, 1], T[1, 0])
        susceptibility_vector_noise =  (noise_rates > 0).astype(int)
        
        self.metrics.add_metric("metadata", draw_id, "susceptibility_vector_posterior", susceptibility_vector_posterior)
        self.metrics.add_metric("metadata", draw_id, "susceptible_posterior", np.sum(susceptibility_vector_posterior)/len(susceptibility_vector_posterior))
        
        self.metrics.add_metric("metadata", draw_id, "susceptibility_vector_noise", susceptibility_vector_noise)
        self.metrics.add_metric("metadata", draw_id, "susceptible_noise", np.sum(susceptibility_vector_noise)/len(susceptibility_vector_noise))
        
        
        epsilon = 0.1
        misspecify_flag = "correct"


        plausible_instance_err_anticipated_train = (ambiguity_train > 0).astype(int)

        
        self.metrics.add_metric("metadata", draw_id, "median_ambiguity", np.median(ambiguity_train))
        self.metrics.add_metric("metadata", draw_id, "mean_ambiguity", np.mean(ambiguity_train))
        
        unresponsive_vector_posterior = ((susceptibility_vector_posterior == 1) & (ambiguity_train == 0))
        unresponsive_vector_noise = ((susceptibility_vector_noise == 1) & (ambiguity_train == 0))
        
        self.metrics.add_metric("metadata", draw_id, "unresponsive_vector_posterior", unresponsive_vector_posterior)
        self.metrics.add_metric("metadata", draw_id, "unresponsive_vector_noise", unresponsive_vector_noise)
        
        self.metrics.add_metric("metadata", draw_id, "unresponsive_posterior", np.sum(unresponsive_vector_posterior)/len(unresponsive_vector_posterior))
        self.metrics.add_metric("metadata", draw_id, "unresponsive_noise", np.sum(unresponsive_vector_noise)/len(unresponsive_vector_noise))
        
        for err_method in ["01", "forward", "backward"]:
            if err_method == "01":
                pop_err_anticipated_train, instance_err_anticipated_train = instance_01loss(yn_train, train_preds)
                pop_err_anticipated_test, instance_err_anticipated_test = instance_01loss(y_test, test_preds)
                
            elif err_method == "forward":
                pop_err_anticipated_train, instance_err_anticipated_train = instance_forward_01loss(yn_train, train_probs, T)
                pop_err_anticipated_test, instance_err_anticipated_test = instance_forward_01loss(y_test, test_probs, T)
                
            elif err_method == "backward":
                _, _, pop_err_anticipated_train, instance_err_anticipated_train = natarajan_unbiased_01_loss(yn_train, train_preds, T)
                _, _, pop_err_anticipated_test, instance_err_anticipated_test = natarajan_unbiased_01_loss(y_test, test_preds, T)
                
                
            empirical_regret_train, regret_instances_train = instance_01loss(instance_err_anticipated_train, instance_err_true_train)

            self.metrics.add_metric(err_method, draw_id, "regret_vector", regret_instances_train)
            self.metrics.add_metric(err_method, draw_id, "clean_risk_test", np.mean(instance_err_true_test)) #average clean 01loss
            self.metrics.add_metric(err_method, draw_id, "noisy_risk_train", np.mean(instance_err_anticipated_train)) #average clean 01loss
            self.metrics.add_metric(err_method, draw_id, "clean_risk_train", np.mean(instance_err_true_train)) #average clean 01loss
            
            self.metrics.add_metric(err_method, draw_id, "delta_pop_err_train", abs(np.mean(instance_err_anticipated_train) - np.mean(instance_err_true_train))) #average clean 01loss
            
            self.metrics.add_metric(err_method, draw_id, "actual_regret_train", empirical_regret_train)
            
            #How well ambiguity flags regretful instances
            self.metrics.add_metric("metadata", draw_id, "coverage_regret_train", np.mean(abs(plausible_instance_err_anticipated_train- regret_instances_train)))
            
            #print(plausible_instance_err_anticipated_train, regret_instances_train)
            plausible_instance_err_anticipated_train = (plausible_instance_err_anticipated_train > 0).astype(int)
            regret_instances_train = (regret_instances_train > 0).astype(int)
            jaccard = jaccard_score(plausible_instance_err_anticipated_train,regret_instances_train)
            self.metrics.add_metric("metadata", draw_id, "jaccard_train", jaccard)
            
            fp_indices_train = ((instance_err_anticipated_train == 1) & (instance_err_true_train == 0))
            fn_indices_train = ((instance_err_anticipated_train == 0) & (instance_err_true_train == 1))
            
            self.metrics.add_metric(err_method, draw_id, "fp_vector", fp_indices_train.astype(int))
            self.metrics.add_metric(err_method, draw_id, "fn_vector", fn_indices_train.astype(int))
            
            indices_regret = np.logical_or(fp_indices_train, fn_indices_train)
            
            #self.metrics.add_metric(err_method, draw_id, "actual_regret_fpr_train",  fp_indices_train.sum()/len(indices_regret) if len(indices_regret) != 0 else 0)
            #self.metrics.add_metric(err_method, draw_id, "actual_regret_fnr_train", fn_indices_train.sum()/len(indices_regret) if len(indices_regret) != 0 else 0)
            
            # Get the indices where instance_err_anticipated_train == 0
            indices_condition = (instance_err_anticipated_train == 0)

            # Calculate the conditional probability P(instance_err_true_train == 1 | instance_err_anticipated_train == 0)
            conditional_prob = fn_indices_train.sum() / indices_condition.sum() if indices_condition.sum() != 0 else 0

            # Add the metric
            self.metrics.add_metric(err_method, draw_id, "actual_regret_fnr_train", conditional_prob)
            
            # Get the indices where instance_err_anticipated_train == 0
            indices_condition = (instance_err_anticipated_train == 1)

            # Calculate the conditional probability P(instance_err_true_train == 1 | instance_err_anticipated_train == 0)
            conditional_prob = fp_indices_train.sum() / indices_condition.sum() if indices_condition.sum() != 0 else 0

            # Add the metric
            self.metrics.add_metric(err_method, draw_id, "actual_regret_fpr_train", conditional_prob)


            non_regret_instances_train = 1-regret_instances_train
        

            regret_indices_train = np.where(regret_instances_train == 1)[0]


            #self.metrics.add_metric(err_method, draw_id, "regret_instances_train", regret_instances_train)
            #self.metrics.add_metric(err_method, draw_id, "ambiguity_train", ambiguity_train)
            #self.metrics.add_metric(err_method, draw_id, "probs_train", train_probs)
            
            groups = ["age", "hiv"] if  (dataset == "saps" or dataset == "saps_imbalanced") else ["age", "sex"]
                    
            for group in groups:
                
                # Load the dataset splits and group labels
                _, _, _, _, group_train, group_test = load_dataset_splits(dataset, group)
                
                for val in np.unique(group_train):
                    group_stats = compute_group_stats(val, group_train, group_test, instance_err_true_train, instance_err_true_test, instance_err_anticipated_train, instance_err_anticipated_test)
                    self.metrics.add_metric(err_method, draw_id, f"fpr_{group}_{val}", group_stats["fpr_train"])

                    self.metrics.add_metric(err_method, draw_id, f"fnr_{group}_{val}", group_stats["fnr_train"])

                    self.metrics.add_metric(err_method, draw_id, f"regret_{group}_{val}", group_stats["regret_train"])
                    self.metrics.add_metric(err_method, draw_id, f"risk_{group}_{val}", group_stats["risk_test"])
            for c in [0,1]:
                
                class_stats = compute_group_stats(c, y_train, y_test, instance_err_true_train, instance_err_true_test, instance_err_anticipated_train, instance_err_anticipated_test)
                self.metrics.add_metric(err_method, draw_id, f"fpr_class_{c}", class_stats["fpr_train"])

                self.metrics.add_metric(err_method, draw_id, f"fnr_class_{c}", class_stats["fnr_train"])
  
                self.metrics.add_metric(err_method, draw_id, f"regret_class_{c}", class_stats["regret_train"])
                self.metrics.add_metric(err_method, draw_id, f"risk_class_{c}", class_stats["risk_test"])
         
    def get_metrics(self):
        """
        Returns the calculated metrics.

        Returns:
        - Metrics object containing all calculated metrics.
        """
        return self.metrics

    def visualize_binary_arrays(self, true_array, anticipated_array, err_method, draw_id):
        """
        Visualize the comparison between true and anticipated error arrays.

        Parameters:
        - true_array: Array of true errors.
        - anticipated_array: Array of anticipated errors.
        - err_method: Error method used.
        - draw_id: ID of the current draw.
        """
        df = pd.DataFrame({
            'True Error': true_array,
            'Anticipated Error': anticipated_array
        })

        plt.figure(figsize=(10, 2))
        sns.heatmap(df.T, cmap="viridis", cbar=False)
        plt.axhline(y=1, color='gray', linestyle='--') 
        plt.title(f'Comparison of True and Anticipated Errors\nMethod: {err_method}, Draw ID: {draw_id}')
        plt.xlabel('Index')
        plt.ylabel('Error Type')
        plt.show()


def load_vectors(dataset, model_type, noise_type, noise_level, training_loss, epsilon=0.1, misspecify="correct", fixed_class = None, fixed_noise = None):
    """
    Load the vectors data from the specified path.

    Parameters:
    - parent_dir (str): The parent directory path.
    - dataset (str): The dataset name.
    - model_type (str): The model type.
    - noise_type (str): The noise type.
    - misspecify (str): The misspecify parameter.
    - noise_level (str): The noise level.
    - epsilon (str): The epsilon value.

    Returns:
    - vectors: The loaded vectors data.
    """
    parent_dir = "/scratch/hdd001/home/snagaraj/"
    files_path = os.path.join(parent_dir, "results", "regret", dataset, model_type, noise_type, misspecify)
    
    if noise_type == "class_independent":
        path = os.path.join(files_path, f"{training_loss}_{noise_level}_{epsilon}_vectors.pkl")
    elif noise_type == "class_conditional":
        path = os.path.join(files_path, f"{training_loss}_{noise_level}_{fixed_class}_{fixed_noise}_{epsilon}_vectors.pkl")

    # Check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file at path {path} does not exist.")
    
    # Load the vectors data from the file
    with open(path, 'rb') as file:
        vectors = pkl.load(file)
    
    return vectors


# class MetricsStorage:
#     def __init__(self, loss_types):
#         self.data = {loss:  {metric: [] for metric in self.metrics()} for loss in loss_types}

#     def metrics(self):
#         return ['regret_train', 'disagreement_train',  'regret_test', 'disagreement_test',  
#                 'noisy_train_loss', 'clean_train_loss', 'clean_test_loss',
#                 'noisy_train_acc', 'clean_train_acc', 'clean_test_acc' , 
#                 'train_loss', 'test_loss', 'train_acc', 'test_acc', 'flip_frequency', 'typical_rate', 'typical_difference',
#                 "preds_train", "preds_test", "train_probs", "test_probs"]

#     def add_metric(self, loss, metric, value):
#         self.data[loss][metric].append(value)

#     def get_metric(self, loss, metric):
#         return self.data[loss][metric]


# def dummy_T_dict(group_train, T):
#     T_dict = {}
#     for key in np.unique(group_train):
#         T_dict[key] = T
#     return T_dict

# def simulate_noise_and_train_model(m, max_iter, X_train, y_train, X_test, y_test, p_y_x_dict, noise_type = "class_independent", uncertainty_type="backward",  model_type = "LR" , fixed_class=0, fixed_noise=0.2, T_true = None, T_est = None, batch_size = 512, base_seed = 2024, epsilon = 0.25):
    
#     if uncertainty_type == "forward":
#         loss_types = ["Ours", "BCE", "backward", "forward"]
#         y_vec = y_train
#     else: # backward
#         loss_types = ["Ours"]

#         #Initial Noise Draw
#         u_vec = get_u(y_train, T = T_true, seed= base_seed, noise_type = noise_type)
#         y_vec = flip_labels(y_train, u_vec) #XOR

#     metrics = MetricsStorage(loss_types)

#     preds_train_dict = {loss: [] for loss in loss_types}
#     preds_test_dict = {loss: [] for loss in loss_types}

#     typical_count = 0

    
#     for seed in tqdm(range(1, max_iter+1)):
#         if uncertainty_type == "forward":
#             # Using a forward model, so get u directly
#             u_vec = get_u(y_vec, T = T_true, seed= seed, noise_type = noise_type)
#         else:
#             u_vec = infer_u(y_vec, noise_type = noise_type, p_y_x_dict = p_y_x_dict,  T = T_est , seed=seed)

#         typical_flag, difference = is_typical(u_vec, p_y_x_dict,  T = T_est, y_vec = y_vec, noise_type = noise_type, uncertainty_type = uncertainty_type, epsilon = epsilon)

#         if not typical_flag: 
#             continue

#         flipped_labels = flip_labels(y_vec, u_vec)

#         if uncertainty_type == "forward":
#             for loss in loss_types:
#                 if loss == "Ours":
#                     model,  (train_acc,
#                         test_acc,
#                         train_probs,
#                         test_probs,
#                         train_loss,
#                         test_loss,
#                         train_preds,
#                         test_preds
#                         ) = train_model_ours(X_train, flipped_labels, X_test, y_test, seed = 2024, model_type=model_type)

#                     preds_train_dict[loss].append(train_preds)
#                     preds_test_dict[loss].append(test_preds)

#                     metrics.add_metric(loss, "noisy_train_loss", train_loss)
#                     metrics.add_metric(loss, "noisy_train_acc", train_acc*100)
#                     metrics.add_metric(loss, "clean_test_loss", test_loss)
#                     metrics.add_metric(loss, "clean_test_acc", test_acc*100)
#                     metrics.add_metric(loss, "typical_difference", difference)
#                     metrics.add_metric(loss, "preds_train", train_preds)
#                     metrics.add_metric(loss, "preds_test", test_preds)
#                     metrics.add_metric(loss, "train_probs", train_probs)
#                     metrics.add_metric(loss, "test_probs", test_probs)

#                 else:
#                     model,  (noisy_train_loss,
#                             clean_train_loss, 
#                             noisy_train_acc,
#                             clean_train_acc,
#                             train_probs,
#                             clean_test_loss, 
#                             clean_test_acc,
#                             test_probs
#                             ) = train_model(X_train, y_train, flipped_labels,  X_test, y_test,  T = T_est, seed=2024, num_epochs=25, batch_size=batch_size, model_type = model_type, correction_type=loss)

#                     preds_train = (train_probs > 0.5).astype(int)
#                     preds_test = (test_probs > 0.5).astype(int)

#                     preds_train_dict[loss].append(preds_train)
#                     preds_test_dict[loss].append(preds_test)

#                     metrics.add_metric(loss, "noisy_train_loss", noisy_train_loss)
#                     metrics.add_metric(loss, "clean_train_loss", clean_train_loss)
#                     metrics.add_metric(loss, "noisy_train_acc", noisy_train_acc*100)
#                     metrics.add_metric(loss, "clean_train_acc", clean_train_acc*100)
#                     metrics.add_metric(loss, "clean_test_loss", clean_test_loss)
#                     metrics.add_metric(loss, "clean_test_acc", clean_test_acc*100)
#                     metrics.add_metric(loss, "flip_frequency", sum(u_vec)/len(u_vec))
#                     metrics.add_metric(loss, "typical_difference", difference)
#                     metrics.add_metric(loss, "preds_train", preds_train)
#                     metrics.add_metric(loss, "preds_test", preds_test)
#                     metrics.add_metric(loss, "train_probs", train_probs)
#                     metrics.add_metric(loss, "test_probs", test_probs)

#         else: #backward_sk
            
#             for loss in loss_types:
#                 model,  (train_acc,
#                         test_acc,
#                         train_probs,
#                         test_probs,
#                         train_loss,
#                         test_loss,
#                         train_preds,
#                         test_preds
#                         ) = train_model_ours(X_train, flipped_labels, X_test, y_test, seed = 2024, model_type=model_type)

#                 preds_train_dict[loss].append(train_preds)
#                 preds_test_dict[loss].append(test_preds)

#                 metrics.add_metric(loss, "train_loss", train_loss)
#                 metrics.add_metric(loss, "train_acc", train_acc*100)
#                 metrics.add_metric(loss, "test_loss", test_loss)
#                 metrics.add_metric(loss, "test_acc", test_acc*100)
#                 metrics.add_metric(loss, "typical_difference", difference)
#                 metrics.add_metric(loss, "preds_train", train_preds)
#                 metrics.add_metric(loss, "preds_test", test_preds)
#                 metrics.add_metric(loss, "train_probs", train_probs)
#                 metrics.add_metric(loss, "test_probs", test_probs)

#         typical_count += 1

#         if typical_count == m:
#             break

#     for loss in loss_types:
#         typical_rate = typical_count / seed
#         print("Typical Rate: ", typical_rate)
#         metrics.add_metric(loss, "typical_rate", typical_rate)

#         predictions_train = np.array(preds_train_dict[loss])

#         predictions_test = np.array(preds_test_dict[loss])

#         try:
#             regret_train = calculate_error_rate(predictions_train, flipped_labels)
#             disagreement_train = estimate_disagreement(predictions_train)

#             regret_test = calculate_error_rate(predictions_test, y_test)
#             disagreement_test = estimate_disagreement(predictions_test)

#         except:
#             print("Error: Could not get Disagreement Metrics")
#             continue

#         for i, item in enumerate(X_train):
#             metrics.add_metric(loss, "regret_train", regret_train[i])
#             metrics.add_metric(loss, "disagreement_train", disagreement_train[i])

#         for i, item in enumerate(X_test):

#             metrics.add_metric(loss, "regret_test", regret_test[i])
#             metrics.add_metric(loss, "disagreement_test", disagreement_test[i])

#     print("DONE")
#     return metrics

# def abstain(rates, threshold):
#     #rates = np.clip(rates, 0, 100)
#     return ((rates > threshold)).astype(int)

  
# def train_model_abstain(X_train, y_train, X_test, y_test, model_type="LR"):
#     # Set random seed for reproducibility

#     seed = 2024
#     np.random.seed(seed)
    
#     # Choose the model based on the input
#     if model_type == "LR":
#         model = LR(**DEFAULT_LR_PARAMS, random_state = seed)
#     elif model_type == "SVM":
#         model = LinearSVC(**DEFAULT_SVM_PARAMS, random_state = seed)
#     elif model_type == "NN":
#         model = MLPClassifier(**DEFAULT_NN_PARAMS, random_state = seed)
#     else:
#         raise ValueError("Unsupported model type. Choose 'LR' or 'SVM'.")

#     # Train the model using noisy labels (simulating the impact of label noise)
#     model.fit(X_train, y_train)

#     # Predictions for training and test sets
#     train_preds = model.predict(X_train)
#     test_preds = model.predict(X_test)

    
 
#     return model, train_preds, test_preds



# def run_procedure_regret(m, max_iter, X_train, yn_train, X_test, y_test,  p_y_x_dict,  noise_type = "class_independent", model_type = "LR", T = None, epsilon = 0.1):
    
#     typical_count = 0
#     preds_train = []
#     preds_test = []
    
#     y_vec = yn_train
    
#     for seed in tqdm(range(1, max_iter+1)):
        
#         u_vec = infer_u(y_vec,  noise_type = noise_type, p_y_x_dict = p_y_x_dict,  T = T , seed=seed)
        
#         typical_flag, _ = is_typical(u_vec, p_y_x_dict,   T = T, y_vec = y_vec, noise_type = noise_type, uncertainty_type = "backward", epsilon = epsilon)

#         if not typical_flag:
#             continue
            
#         flipped_labels = flip_labels(y_vec, u_vec)
        
#         model,  (train_acc,
#                 test_acc,
#                 train_probs,
#                 test_probs,
#                 train_loss,
#                 test_loss,
#                 train_preds,
#                 test_preds)= train_model_ours_regret(X_train, flipped_labels, X_test, y_test, seed = 2024, model_type=model_type)
        
#         preds_train.append(train_preds)
#         preds_test.append(test_preds)

#         typical_count += 1

#         if typical_count == m:
#             break
            
#     predictions_train = np.array(preds_train)
#     disagreement_train = estimate_disagreement(predictions_train)
#     ambiguity_train = calculate_error_rate(predictions_train, y_vec)

#     predictions_test = np.array(preds_test)
#     disagreement_test = estimate_disagreement(predictions_test)
#     ambiguity_test = calculate_error_rate(predictions_test, y_test)

#     return predictions_train, predictions_test, disagreement_train, disagreement_test, ambiguity_train, ambiguity_test

