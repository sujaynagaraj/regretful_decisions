import numpy as np
from scipy.stats import bernoulli

def generate_class_independent_noise(y, flip_p):
    noise_transition_matrix = np.array([[1-flip_p, flip_p], [flip_p, 1-flip_p]])

    return add_label_noise(y, noise_transition_matrix=noise_transition_matrix), noise_transition_matrix

def generate_class_conditional_noise(y, flip_p, fixed_class, fixed_noise):
    if fixed_class == 0:
        noise_transition_matrix = np.array([[1-fixed_noise, fixed_noise], [flip_p, 1-flip_p]])
    else:
        noise_transition_matrix = np.array([[1-flip_p, flip_p], [fixed_noise, 1-fixed_noise]])
    return add_label_noise(y, noise_transition_matrix=noise_transition_matrix), noise_transition_matrix

def generate_group_noise(y, groups, noise_transition_dict):

    return add_label_noise(y, groups, noise_type = "group", noise_transition_dict=noise_transition_dict), noise_transition_dict

def generate_instance_dependent_noise(y, X, flip_p, feature_weights):
    noise_transition_dict = {}
    for instance in np.unique(X, axis=0):
        flip_instance = instance_noise_level(instance, flip_p, feature_weights)
        noise_transition_dict[tuple(instance)] = np.array([[1-flip_instance, flip_instance],
                                                           [flip_instance, 1-flip_instance]])
    return add_label_noise(y, noise_type="feature", X=X, noise_transition_dict=noise_transition_dict), noise_transition_dict


def adjust_transition_matrix(T, adjustment_factor):
    """
    Adjust the transition matrix by a given factor on off-diagonal elements.
    - Positive factor for overestimation (increase noise)
    - Negative factor for underestimation (decrease noise)
    """
    T_adj = T.copy()
    n = T.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j:
                T_adj[i, j] = T[i, j] +( adjustment_factor)
                T_adj[i, i] = 1 - T_adj[i, j]
    return T_adj

def add_label_noise(labels, groups = None, noise_transition_matrix=None, noise_type = "class_independent", X= None, noise_transition_dict=None):
    noisy_labels = np.copy(labels)
    
    np.random.seed(2024)

    if noise_type == "class_independent":
        for i, label in enumerate(labels):
            noisy_labels[i] = np.random.choice([0, 1], p=noise_transition_matrix[label])
    elif noise_type == "class_conditional":
        for i, label in enumerate(labels):
            noisy_labels[i] = np.random.choice([0, 1], p=noise_transition_matrix[label])
    elif noise_type == "group":
        for i, group in enumerate(groups):
            noise_transition_matrix_group = noise_transition_dict[group]
            noisy_labels[i] = np.random.choice([0, 1], p=noise_transition_matrix_group[labels[i]])

    return noisy_labels

def instance_noise_level(instance_features, base_noise_level, feature_weights):
    """
    Calculate noise levels for an instance based on its features and given parameters.
    
    :param instance_features: The features of the instance.
    :param base_noise_level: The base noise level.
    :param feature_weights: Weights to apply to each feature to determine its influence on the noise level.
    :return: The noise level for the instance.
    """
    noise_level = base_noise_level + np.dot(instance_features, feature_weights)
    # Ensure the noise level is within valid probability bounds [0, 0.49]
    noise_level = min(max(noise_level, 0), 0.49)
    return noise_level


def get_u(y,  T, seed, group = None, noise_type = "class_independent"):
    np.random.seed(seed)
    
    if noise_type == "group":
        # Initialize an array to hold the noise rates
        noise_rates = np.zeros_like(y, dtype=float)
        
        # Iterate over each label and group to set the correct noise rate
        for i in range(len(y)):
            # Get the group-specific transition matrix
            T_i = T[group[i]]
            # Use the transition matrix based on the label from y
            if y[i] == 0:
                noise_rates[i] = T_i[0, 1]
            else:
                noise_rates[i] = T_i[1, 0]
    else:
        # Define noise rates based on the label
        noise_rates = np.where(y == 0, T[0, 1], T[1, 0])
        # Sample u for all labels at once
    
    sampled_u = bernoulli.rvs(p=noise_rates)
    
    return sampled_u

def infer_u(yn, seed, p_y_x_dict, group = None, noise_type = "class_independent", T= None):
    np.random.seed(seed)  # Set seed for reproducibility
 
    if noise_type == "group":
        posterior = np.zeros_like(yn, dtype=float)
        
        # Iterate over each label and group to set the correct noise rate
        for i, val in enumerate(yn):
            T_i = T[group[i]]
            posterior[i] = calculate_posterior(val, T_i, p_y_x_dict[group[i]])
        
    else:
        posterior = calculate_posterior(yn, T, p_y_x_dict[0])

    # Generate random samples from a Bernoulli distribution for the entire vector
    sampled_u = bernoulli.rvs(p=posterior)
    
    return sampled_u

    


def calculate_posterior(yn, T, p_y_x):
     # Create arrays of opposite class indices
    opp_class = 1 - yn  # Flips 0 to 1 and 1 to 0

    # Index T for probabilities of unobserved true class and observed noisy class
    p_u_opp = T[opp_class, yn]
    p_u = T[yn, opp_class]

    # Indexing p_y_x for class probabilities
    p_y_given_x_yn = p_y_x[yn]           # P(Y=yn|X)
    p_y_given_x_opp = p_y_x[opp_class]   # P(Y=opp_class|X)

    # Calculate the numerator of Bayes' rule
    numerator = p_u_opp * p_y_given_x_opp

    # Calculate the denominator of Bayes' rule
    denominator = (1 - p_u) * p_y_given_x_yn + p_u_opp * p_y_given_x_opp

    # Element-wise division to find posterior probabilities
    p_u_given_yn_x = numerator / denominator

    return p_u_given_yn_x


def flip_labels(y, u):
    """
    Takes two binary numpy arrays, u and y, and returns a new array noisy_y
    which is the element-wise XOR of u and y.
    """

    # Ensure the inputs are numpy arrays in case they're not already
    u = np.array(u)
    y = np.array(y)

    # Perform element-wise XOR operation
    noisy_y = np.logical_xor(u, y).astype(int)
    
    return noisy_y


def calculate_prior(y, group, noise_type = "class_independent"):
    
    p_y_x_dict = {}
    for g in np.unique(group, axis=0):
        
        if noise_type == "group":

            indices = [idx for idx, elem in enumerate(group) if np.array_equal(elem, g)]
            p1 = np.sum(y[indices])/len(indices)

        else:
            p1 = np.sum(y)/len(y)

        p_y_x = np.array([1-p1,p1])
        p_y_x_dict[g] = p_y_x
    
    return p_y_x_dict
  


def is_typical(u_vec, p_y_x_dict, group = None,  T = None, y_vec = None, epsilon=0.25, noise_type = "class_independent", uncertainty_type = "backward"):
    """
    Checks if the observed flips (u_vec) are typical given the noise model (T) and, optionally,
    the class distributions p_y_x.

    Parameters:
    - u_vec: Array of observed flips.
    - T: Noise transition matrix or dict.
    - y_vec: True labels, required for instance-dependent noise.
    - p_y_x: Class probabilities given features, required for instance-dependent noise.
    - epsilon: Tolerance level for deviation from expected noise.
    - noise_type: Type of noise, affects how typicality is assessed.

    Returns:
    - bool: True if the flips are typical, False otherwise.
    """
    
    if uncertainty_type == "forward" and noise_type=="class_independent":

        noise_rate = T[0,1]
        
        #print(sum(u_vec)/len(u_vec), noise_rate)
    
        if abs(sum(u_vec)/len(u_vec) - noise_rate) <= epsilon*noise_rate:
            return True, abs(sum(u_vec)/len(u_vec) - noise_rate)
        else:
            return False, abs(sum(u_vec)/len(u_vec) - noise_rate)

    elif noise_type == "group":
        bool_flag = True
        for g in np.unique(group):
            p_g = np.sum((group == g))/len(group)
            
            for y in [0,1]:
                
                p_y_x = p_y_x_dict[g]
                
                # Create a boolean mask where all conditions are true
                mask = (u_vec == 1) & (y_vec == y) & (group == g)#(U=u,Y=y)

                # Count the number of True values in the mask
                count = np.sum(mask)
                total_instances = np.sum((y_vec == y) & (group == g))
                
                freq = count/total_instances if total_instances !=0 else 0

                opp_class = 1-y

                 # Compute P(Y_tilde = y | G = g)
                p_y_tilde = (
                    T[g][y, 0] * p_y_x[0] +
                    T[g][y, 1] * p_y_x[1]
                )

                # Compute numerator for P(Y = y | Y_tilde = y, G=g)
                numerator = T[g][y, y] * p_y_x[y]

                # Compute true_freq = P(U=1 | Y_tilde = y, G=g)
                true_freq = 1 - numerator / p_y_tilde

                deviation = abs(freq - true_freq)


                # Check if the deviation exceeds the tolerance
                if deviation > epsilon * true_freq:
                    
                    bool_flag = False
                    break  # Exit inner loop over y
                #print("Here")
            if not bool_flag:
                break  # Exit outer loop over groups


        return bool_flag, deviation
        
    else:
        bool_flag = True

        for y in [0,1]:
            
            p_y_x = p_y_x_dict[0]
            

            # Create a boolean mask where both conditions are true
            mask = (u_vec == 1) & (y_vec == y) #(U=u,Y=y)
            
            # Count the number of True values in the mask
            count = np.sum(mask)

            freq = count/len(u_vec)

            opp_class = 1-y

            p_u_opp = T[opp_class, y]

            p_u = T[y,opp_class]

            # Sum the resulting vector to get P(y_tilde = observed_y_tilde | x = x)
            p_yn = (1-p_u)*p_y_x[y] + p_u_opp*p_y_x[opp_class]
            
            if uncertainty_type == "forward":
                true_freq = p_u*p_y_x[y]
            else:
                true_freq = calculate_posterior(y, T, p_y_x)*p_yn
            
            
            if (abs(freq - true_freq) > epsilon*true_freq): #atypical
                bool_flag = False, abs(freq - true_freq)
                break
                

        return bool_flag, abs(freq - true_freq)