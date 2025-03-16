import pandas as pd

def acc_prob_to_df(accuracies_dict, probabilities_dict):
    """
    Combines accuracies and predicted probabilities into a single DataFrame.

    :param accuracies_dict: Dictionary with labels as keys and lists of accuracies as values.
    :param probabilities_dict: Dictionary with labels as keys and lists of predicted probabilities as values.
    :return: DataFrame with columns 'Accuracy', 'Predicted Probabilities', and 'Label'.
    """
    data = []

    for label in accuracies_dict:
        for accuracy, probabilities in zip(accuracies_dict[label], probabilities_dict[label]):
            row = {
                "Accuracy": accuracy,
                "Predicted Probabilities": probabilities,
                "Label": label
            }
            data.append(row)

    return pd.DataFrame(data)


def disagreement_to_df(disagreement_dict):
    """
    Converts a disagreement dictionary into a DataFrame.

    :param disagreement_dict: Dictionary with labels as keys and arrays of disagreement rates as values.
    :return: DataFrame with columns 'Disagreement Rate' and 'Label'.
    """
    data = []

    for label, disagreement_rates in disagreement_dict.items():
        for rate in disagreement_rates:
            row = {
                "Disagreement Rate": rate,
                "Label": label
            }
            data.append(row)

    return pd.DataFrame(data)

# Example usage
# disagreement_df = disagreement_dict_to_dataframe(disagreement_dict)
# print(disagreement_df.head())
