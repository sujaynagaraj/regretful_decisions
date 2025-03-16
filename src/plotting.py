import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns

from src.metrics import *

import pandas as pd

def visualize_pca(X, labels, title):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.scatter(X_pca[labels == 0, 0], X_pca[labels == 0, 1], alpha=0.7, label='Class 0')
    plt.scatter(X_pca[labels == 1, 0], X_pca[labels == 1, 1], alpha=0.7, label='Class 1')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

def plot_boxplot(data_dict, y_range=(0.5, 1), figsize=(10, 5), colors=None, title = "Accuracy", save_path = None):
    """
    Plots a boxplot for the given data.

    :param data_dict: Dictionary with labels as keys and data arrays as values.
    :param y_range: Tuple indicating the range of y-axis.
    :param figsize: Size of the figure.
    :param colors: List of colors for the boxplots.
    """
    labels = list(data_dict.keys())
    data = list(data_dict.values())

    # Initialize the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create boxplot with patch_artist=True
    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    # Set colors for each box
    if colors is None:
        # Default colors if not provided
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightgrey']
    for patch, color in zip(bp['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)

    # Set y-axis limits
    ax.set_ylim(*y_range)

    # Set labels
    ax.set_ylabel(title)
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_robustness_rates(data_dict, epsilon_range=(0.01, 0.5), num_points=50, figsize=(10, 5), colors=None, save_path = None):
    """
    Plots the robustness rates for different sets of predicted probabilities.

    :param data_dict: Dictionary with labels as keys and predicted probabilities arrays as values.
    :param epsilon_range: Tuple indicating the range of epsilon values.
    :param num_points: Number of points to calculate within the epsilon range.
    :param figsize: Size of the figure.
    :param colors: List of colors for the plots.
    """
    epsilon_values = np.linspace(epsilon_range[0], epsilon_range[1], num_points)

    plt.figure(figsize=figsize)

    if colors is None:
        # Default colors if not provided
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightgrey']

    for (label, predicted_probabilities), color in zip(data_dict.items(), colors):
        robustness_rates = [calculate_robustness_rate(predicted_probabilities, epsilon) for epsilon in epsilon_values]
        plt.plot(epsilon_values, robustness_rates, marker='o', label=label, color=color)

    plt.xlabel('Epsilon')
    plt.ylabel('Robustness Rate')
    plt.title('Robustness Rate vs. Epsilon')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_disagreement_rates(data_dict, figsize=(10, 5), colors=None, save_path = None):
    """
    Plots the disagreement rates as boxplots.

    :param data_dict: Dictionary with labels as keys and disagreement rate arrays as values.
    :param figsize: Size of the figure.
    :param colors: List of colors for the boxplots.
    """
    labels = list(data_dict.keys())
    data_disagreement = list(data_dict.values())

    fig, ax = plt.subplots(figsize=figsize)

    # Create boxplot with patch_artist=True
    bp = ax.boxplot(data_disagreement, labels=labels, patch_artist=True)

    # Set colors for each box
    if colors is None:
        # Default colors if not provided
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightgrey']
    for patch, color in zip(bp['boxes'], colors[:len(data_disagreement)]):
        patch.set_facecolor(color)

    # Set labels
    ax.set_ylabel('Disagreement Rate')
    ax.set_title('Model Disagreement Rate')

    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_disagreement_percentage(probabilities_dict, threshold_range=(0, 0.5), num_points=50, figsize=(10, 5), colors=None, save_path = None):
    """
    Plots the percentage of examples with disagreement rates above varying thresholds.

    :param probabilities_dict: Dictionary with labels as keys and arrays of predicted probabilities as values.
    :param threshold_range: Tuple indicating the range of disagreement thresholds.
    :param num_points: Number of threshold points to calculate.
    :param figsize: Size of the figure.
    :param colors: List of colors for the plots.
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_points)

    plt.figure(figsize=figsize)

    if colors is None:
        # Default colors if not provided
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightgrey']

    for (label, probabilities), color in zip(probabilities_dict.items(), colors):
        disagreement_percentages = [disagreement_percentage(estimate_disagreement(probabilities), threshold) for threshold in thresholds]
        plt.plot(thresholds, disagreement_percentages, marker='o', label=label, color=color)

    plt.xlabel('Disagreement Threshold')
    plt.ylabel('% of Samples w. Disagreement > Threshold')
    plt.title('Disagreement % vs. Threshold')
    plt.ylim(0,1)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_metrics(df_accuracy_disagreement, disagreement_df, save_path = None):
    """
    Plots two subplots using seaborn lineplot for Accuracy and Disagreement Rate, and one subplot using matplotlib plot from disagreement_df with Noise Frequency as x-axis and color by Method.

    :param df_accuracy_disagreement: DataFrame containing Accuracy and Disagreement Rate data.
    :param disagreement_df: DataFrame containing data for the third plot.
    """

    # Sort the data by 'Noise Frequency' to ensure the lines are continuous
    df_accuracy_disagreement = df_accuracy_disagreement.sort_values(by='Noise Frequency')
    disagreement_df = disagreement_df.sort_values(by='Noise Frequency')

    # Define color scheme
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightgrey']

    # Create subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 15), sharex=True)

    # First subplot for Accuracy using seaborn lineplot
    sns.lineplot(data=df_accuracy_disagreement, x='Noise Frequency', y='Accuracy', hue='Method', 
                 palette=colors, ax=axes[0], marker='o')
    axes[0].set_title('Accuracy as a Function of Noise Frequency')
    axes[0].set_ylim((0,1))

    # Second subplot using matplotlib plot for the disagreement_df DataFrame
    for method,color in zip(disagreement_df['Method'].unique(), colors):
        subset = disagreement_df[disagreement_df['Method'] == method]
        axes[1].plot(subset['Noise Frequency'], subset['Disagreement Percent'], marker='o', label=method, 
                     color=color)
    axes[1].set_title('Disagreement Rate > 0.5')
    axes[1].set_xlabel('Noise Frequency')
    axes[1].set_ylabel('% Examples w/ Disagreement Rate > 0.5')
    axes[1].set_ylim((0,1))
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Define the function for plotting the decision boundary
def plot_decision_boundary(model, X, y):
    
    np.random.seed(42)
    
    noise_scale = 0.05  # scale of the noise to be added
    X_noisy = X + np.random.normal(scale=noise_scale, size=X.shape)

    # Create a mesh grid on which we will run our model to obtain the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Compute model output for the entire grid
    Z = model[0] + (model[1] * xx) + (model[2] * yy)
    Z = Z > 0  # Decision boundary is where Z flips from negative to positive

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.2)

    # Plot the training points with some noise added
    scatter = plt.scatter(X_noisy[:, 0], X_noisy[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor='k', s=20)
    
    # Add legend, axis labels and title
    plt.legend(handles=scatter.legend_elements()[0], labels=['Class 0', 'Class 1'])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary with Data Points')
    

# Define the function for plotting decision boundaries
def plot_decision_boundaries(models, X, y):
    # Create a grid to plot the decision boundaries
    x_min, x_max = -1, 2
    y_min, y_max = -1, 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # Plot the dataset points
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap=plt.cm.Paired, edgecolors='k')

    # Plot each decision boundary
    for i, model in enumerate(models):
        # Calculate the values on the grid
        Z = model[0] + model[1] * xx + model[2] * yy
        Z = Z >= 0  # Decision boundary is where model output is 0
        plt.contour(xx, yy, Z, levels=[0], cmap=plt.cm.Paired, alpha=0.5, linestyles='solid')

    # Set labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('2D Linear Model Decision Boundaries')

    # Show plot
    plt.show()

def plot_metrics_toy(rates, noise_levels, colors = None, title = None, save_path = None, ax = None):
    """
    Plot the ambiguity rates using Seaborn.
    
    :param ambiguity_rates: A dictionary with ambiguity rates.
    :param noise_levels: A list or array of noise levels that were tested.
    """

    if ax is None:
        ax = plt.gca()

    if colors is None:
        # Default colors if not provided
        colors = {'[0, 0]': 'lightblue', '[0, 1]': 'lightgreen', '[1, 0]': 'lightcoral', '[1, 1]':'lightgrey'}
        
    for (instance, rate) in rates.items():
        ax.plot(noise_levels, rate, marker='o', label=f'{instance}', color=colors[instance], linewidth=5, alpha = 0.8)

    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Rate (%)')
    if title:
        ax.set_title(title)
    ax.set_ylim(0, 100)  # Bound the y range from 0 to 100
    ax.legend()
    ax.grid(True)

    if save_path:
        plt.savefig(save_path)

def plot_losses(losses, noise_levels, colors = None, title = None, save_path = None, ax = None, acc = False):
    """
    Plot the ambiguity rates using Seaborn.
    
    :param ambiguity_rates: A dictionary with ambiguity rates.
    :param noise_levels: A list or array of noise levels that were tested.
    """

    if ax is None:
        ax = plt.gca()

    if colors is None:
        # Default colors if not provided
        colors = {'0-1 Clean': 'lightblue', '0-1 Noisy': 'lightgreen', 'Corrected 0-1 Noisy': 'lightcoral'}
    for (instance, value) in losses.items():
        ax.plot(noise_levels, value, marker='o', label=f'{instance}', color=colors[instance], linewidth=5)

    ax.set_xlabel('Noise Level')
    
    if acc:
        ax.set_ylabel('Accuracy')
    else:
        ax.set_ylabel('Loss')
        
    if title:
        ax.set_title(title)
    #ax.set_ylim(0, 100)  # Bound the y range from 0 to 100
    ax.legend()
    ax.grid(True)

    if save_path:
        plt.savefig(save_path)



def plot_metrics_boxplot(rates, X, title=None, save_path=None, ax=None):
    """
    Plot boxplots of rates for each noise level, with jittered scatter points colored by instance.

    :param rates: Dict with each key being a noise level and each value being a list of rates.
    :param X: np.array of (x1, x2) pairs.
    :param title: Optional title for the plot.
    :param save_path: If provided, save the plot to this path.
    :param ax: Optional matplotlib axes object to plot on.
    """
    # Check if ax is provided, if not create a new figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Specify colors for each instance
    colors = {
        '[0, 0]': 'lightblue', 
        '[0, 1]': 'lightgreen', 
        '[1, 0]': 'lightcoral', 
        '[1, 1]': 'lightgrey'
    }
    
    # Prepare the data for plotting
    data = []
    for noise_level, rate_list in rates.items():
        noise_level_percentage = round(noise_level * 100)  # Convert noise level to integer percentage
        for i, rate in enumerate(rate_list):
            x1, x2 = X[i]
            instance_key = str([x1, x2])
            data.append({
                "Noise Level (%)": f"{noise_level_percentage}%", 
                "Rate (%)": rate, 
                "Instance": instance_key
            })

    df = pd.DataFrame(data)
    
    # Plot the boxplot without coloring the boxes
    sns.boxplot(x="Noise Level (%)", y="Rate (%)", data=df, whis=np.inf, color='lightgray', ax=ax)
    
    # Add jittered scatter plot using seaborn's stripplot
    sns.stripplot(x="Noise Level (%)", y="Rate (%)", hue="Instance", data=df, jitter=True, dodge=False, palette=colors, size=6, linewidth=0.5, edgecolor='gray', ax=ax)
    
    # Customizations
    ax.set_xlabel('Noise Level (%)')
    ax.set_ylabel('Rate (%)')
    ax.set_ylim(-10,110)
    if title:
        ax.set_title(title)
    ax.grid(True)
    ax.legend(title="Instance", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    elif ax is None:  # Only show the plot if ax is not provided
        plt.show()



def plot_metrics_boxplot_real(ambiguity_rates, disagreement_rates, accuracy_rates, X, title=None, save_path=None):
    """
    Plot boxplots of rates for each noise level and loss function, with jittered scatter points.
    Color up to five individual points, if there are more than five unique values for x, 
    then only five chosen indices are colored. Plot for disagreement, ambiguity, and accuracy rates.

    :param disagreement_rates: Dict of dicts, outer keys being loss functions and inner keys being noise levels.
    :param ambiguity_rates: Same structure as disagreement_rates.
    :param accuracy_rates: Same structure as disagreement_rates.
    :param X: np.array of (x1, x2, ...) tuples.
    :param title: Optional title for the plot.
    :param save_path: If provided, save the plot to this path.
    """
    # Calculate number of loss functions to determine subplot columns
    num_loss_functions = len(accuracy_rates)
    color_indices = np.array([0, 10, 100, 200, 500])  # Preset indices to highlight

    # Create a new figure with subplots
    fig, axes = plt.subplots(3, num_loss_functions, figsize=(12 * num_loss_functions, 18))
    if num_loss_functions == 1:
        axes = axes[:, np.newaxis]  # Ensure axes is always 2D array

    # Define a function to prepare and plot data
    def prepare_and_plot(ax, rates, title, highlight=True):
        for loss_function, rate_dict in rates.items():
            data = []
            for noise_level, rate_list in rate_dict.items():
                noise_level_percentage = round(noise_level * 100)  # Convert noise level to percentage
                for i, rate in enumerate(rate_list):
                    color = 'lightgray' if i not in color_indices or not highlight else 'bright'
                    data.append({
                        "Noise Level (%)": f"{noise_level_percentage}%",
                        "Rate (%)": rate,
                        "Index": i,
                        "Color": color,
                        "Loss Function": loss_function
                    })

            df = pd.DataFrame(data)
            sns.boxplot(x="Noise Level (%)", y="Rate (%)", data=df, ax=ax, color="lightblue", whis=np.inf)
            
            # Scatter plot with or without color
            if highlight:
                df_colored = df[df['Color'] != 'lightgray']
                sns.stripplot(x="Noise Level (%)", y="Rate (%)", hue="Index", data=df_colored, jitter=True, dodge=False, palette='bright', size=10, linewidth=0.5, edgecolor='gray', ax=ax)
            else:
                sns.stripplot(x="Noise Level (%)", y="Rate (%)", data=df, jitter=True, color='gray', ax=ax)
            
            # Customizations
            ax.set_xlabel('Noise Level (%)')
            ax.set_ylabel('Rate (%)')
            ax.set_ylim(-10, 110)
            ax.set_title(f'{loss_function} - {title}')
            ax.grid(True)
            if highlight:
                ax.legend(title="Instance", bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.legend([], [], frameon=False)

    # Plot each metric for each loss function
    for idx, metric in enumerate([accuracy_rates, ambiguity_rates, disagreement_rates]):
        metric_name = ["Accuracy", "Regret", "Disagreement"][idx]
        for jdx, (loss_function, _) in enumerate(metric.items()):
            prepare_and_plot(axes[idx, jdx], {loss_function: metric[loss_function]}, metric_name, highlight=idx != 0)

    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
        
        
def plot_boxplots_with_condition(ambiguity_rates, disagreement_rates, X, groups, y_test=None, title=None, save_path=None):
    """
    Plot box plots with scatter for disagreement, ambiguity, and accuracy rates conditioned on y_test values and group parameters.

    :param disagreement_rates: Dict with each key being a noise level and each value being a list of rates.
    :param ambiguity_rates: Dict with each key being a noise level and each value being a list of rates.
    :param accuracy_rates: Dict with each key being a noise level and each value being a list of rates.
    :param X: np.array of (x1, x2, ...) tuples.
    :param groups: np.array corresponding to the group each instance in X belongs to.
    :param y_test: Optional np.array of target variable values to use as hue.
    :param title: Optional title for the plot.
    :param save_path: If provided, save the plot to this path.
    """
    
    num_loss_functions = len(ambiguity_rates)
    
    # Create a new figure with subplots
    fig, axes = plt.subplots(2*num_loss_functions, 2, figsize=(25, 15))
   
    # Function to prepare data and plot
    def prepare_and_plot(ax, rates, title):
        data = []
        for noise_level, rate_list in rates.items():
            noise_level_percentage = round(noise_level * 100)  # Convert noise level to percentage
            for i, rate in enumerate(rate_list):
                if groups[i] in [1,2]:
                    data.append({
                        "Noise Level (%)": f"{noise_level_percentage}%",
                        "Rate (%)": rate,
                        "Group": groups[i],
                        "Y": y_test[i] if y_test is not None else None
                    })

        df = pd.DataFrame(data)

        # Plot conditioned on y_test if available
        if y_test is not None:
            sns.boxplot(x="Noise Level (%)", y="Rate (%)", data=df, ax=ax[0], color="lightblue")
            sns.stripplot(x="Noise Level (%)", y="Rate (%)", hue="Y", data=df, ax=ax[0], jitter=True, dodge=True, palette='bright', marker='o', alpha=0.7)
            ax[0].set_title(f'{title} by Y')
            ax[0].legend(title="Y")
        else:
            ax[0].set_visible(False)  # Hide the axis if y_test is not provided

        # Plot conditioned on groups
        sns.boxplot(x="Noise Level (%)", y="Rate (%)", data=df, ax=ax[1], color="lightblue")
        sns.stripplot(x="Noise Level (%)", y="Rate (%)", hue="Group", data=df, ax=ax[1], jitter=True, dodge=True, palette='bright', marker='o', alpha=0.7)
        ax[1].set_title(f'{title} by Group')
        ax[1].legend(title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')

     # Plot each metric for each loss function
    for idx, metric in enumerate([ambiguity_rates, disagreement_rates]):
        metric_name = ["Regret", "Disagreement"][idx]
        for jdx, (loss_function, _) in enumerate(metric.items()):
            if metric_name == "Regret":
                ind = 0
            else:
                ind = num_loss_functions
            
            prepare_and_plot(axes[ind+jdx], metric[loss_function], metric_name+" " + loss_function)

    # Set overall title
    if title:
        fig.suptitle(title)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect to make space for title if present

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


def plot_metric(metrics_df, metric, highlight = "instance" , highlight_vec = None, ax=None, title = None):

    # Creatabse a DataFrame suitable for Seaborn
    df = metrics_df[metrics_df["Metric"] == metric]
    loss_functions = len(metrics_df["Loss Function"].unique())
    
    # Create the plot
    ax = ax or plt.gca()

    if loss_functions != 1:
        sns.boxplot(x='Noise Level (%)', y='Rate (%)', hue='Loss Function', data=df, ax=ax, palette="Set2")
    else:
        sns.boxplot(x='Noise Level (%)', y='Rate (%)', data=df, ax=ax, palette="Set2")
    # Scatter plot with or without color

    bright_colors = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#DA70D6']
    
    if highlight == "instance":
        sns.stripplot(x="Noise Level (%)", y="Rate (%)", hue="Loss Function", data=df, jitter=True, dodge=True,
                           size=0.5, linewidth=5, ax=ax, alpha = 0.5)
        color_indices = np.array([0, 100, 200, 300, 400])

        if "disagreement" in metric or "regret" in metric:
            for i, item in enumerate(color_indices):
                data_subset = df[df["Index"] == item]
                sns.stripplot(x="Noise Level (%)", y="Rate (%)", hue="Loss Function", data=data_subset, jitter=True, dodge=True,
                              marker='o', facecolors='none', size=5, alpha = 0.8, linewidth=5, edgecolor=bright_colors[i], ax=ax)
    else:
 
        if ("disagreement" in metric) or ("regret" in metric):

            #df[str(highlight)] = highlight_vec
            
            sns.stripplot(x="Noise Level (%)", y="Rate (%)", hue=highlight, data=df, jitter=True,
                           size=5, alpha = 0.6, ax=ax, palette="Set1")
        
    # Add legend and adjust plot details
    handles, labels = ax.get_legend_handles_labels()

    if highlight == "instance":
        l = plt.legend(handles[:(loss_functions)], labels[:(loss_functions)], title='Loss Function')
    else:
        num_groups = len(np.unique(highlight_vec))

        l = plt.legend(handles[:(loss_functions)] + handles[-num_groups:], labels[:(loss_functions)] + labels[-num_groups:], title='Loss Function')
    
    ax.add_artist(l)

    # Set plot title and labels if ax is None
    if ax is None:
        ax.set_xlabel('Noise Level (%)')
        ax.set_ylabel('Rate (%)')
        plt.show()

# Note: Ensure the 'metrics' object and the 'metric' variable are properly defined before calling this function.

def create_boxplot(df, metric_names, ncols=3):
    nrows = (len(metric_names) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), sharey=False)
    #fig.suptitle(f'Boxplots for Training Loss: {training_loss}', fontsize=16)
    axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

    for i, metric_name in enumerate(metric_names):
        sns.boxplot(ax=axs[i], x='noise_level', y='value', hue='training_loss', data=df[df['metric_name'] == metric_name], dodge=True)
        sns.stripplot(ax=axs[i], x='noise_level', y='value', hue='training_loss', data=df[df['metric_name'] == metric_name], 
                      dodge=True, marker='o', s = 3,alpha=1, palette='viridis')
        
    
        #axs[i].set_ylim(0, 1.0)
            
        axs[i].set_title(f'{metric_name}')
        
        # Remove the duplicate legends from the stripplot
        if i == 0:
            handles, labels = axs[i].get_legend_handles_labels()
            fig.legend(handles, labels, title='Training Loss', bbox_to_anchor=(1.05, 0.5), loc='center')
        axs[i].legend_.remove()

    # Remove any empty subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title and legend
    plt.show()
