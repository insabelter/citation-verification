from tabulate import tabulate
import pandas as pd
from IPython.display import display
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import fisher_exact

def eval_predictions(df, include_relabelled_partially=False, include_not_originally_downloaded=True):
    G_Total = 0
    Sub_Correct_Total = 0
    Unsub_Correct_Total = 0
    Sub_True = 0
    Sub_False = 0
    Unsub_True = 0
    Unsub_False = 0

    invalid_labels = {
        'Unsubstantiated': [],
        'Substantiated': []
    }

    for index, row in df.iterrows():
        if row['Reference Article Downloaded'] == 'Yes':
            if include_not_originally_downloaded or row['Reference Article PDF Available'] == 'Yes':
                if include_relabelled_partially or row['Previously Partially Substantiated'] != 'x':
                    G_Total += 1
                    target_label = row['Label']
                    model_label = row['Model Classification Label']

                    assert target_label in ['Unsubstantiated', 'Substantiated'], f"Row {index} Label is not a valid label: {target_label}"

                    invalid_label = False
                    if model_label not in ['Unsubstantiated', 'Substantiated']:
                        invalid_labels[target_label].append(model_label)
                        print(f"Row {index} Model Classification Label is not a valid label: {model_label}")
                        invalid_label = True
                
                    if target_label == 'Substantiated':
                        Sub_Correct_Total += 1
                        if model_label == 'Substantiated':
                            Sub_True += 1
                        elif model_label == 'Unsubstantiated':
                            Sub_False += 1
                        elif invalid_label:
                            Sub_False += 1
                    elif target_label == 'Unsubstantiated':
                        Unsub_Correct_Total += 1
                        if model_label == 'Substantiated':
                            Unsub_False += 1
                        elif model_label == 'Unsubstantiated':
                            Unsub_True += 1
                        elif invalid_label:
                            Unsub_False += 1

    assert G_Total == Sub_Correct_Total + Unsub_Correct_Total, f"Total G ({G_Total}) does not equal Sub_Correct_Total ({Sub_Correct_Total}) + Unsub_Correct_Total ({Unsub_Correct_Total})"
    assert Sub_True + Sub_False == Sub_Correct_Total, f"Sub_True ({Sub_True}) + Unsub_False ({Unsub_False}) does not equal Sub_Correct_Total ({Sub_Correct_Total})"
    assert Unsub_True + Unsub_False == Unsub_Correct_Total, f"Unsub_True ({Unsub_True}) + Sub_False ({Sub_False}) does not equal Unsub_Correct_Total ({Unsub_Correct_Total})"
    assert G_Total != 0, f"Total G ({G_Total}) is 0, cannot calculate metrics"

    results = {
        'Total': G_Total,
        'Substantiated': {
            'Label Total': Sub_Correct_Total,
            'True Classifications': Sub_True,
            'False Classifications': Sub_False,
            'Accuracy': round(Sub_True / Sub_Correct_Total, 3),
            'Precision': round(Sub_True / (Sub_True + Unsub_False) if (Sub_True + Unsub_False) > 0 else 0.0, 3),
            'Recall': round(Sub_True / Sub_Correct_Total if Sub_Correct_Total > 0 else 0.0, 3),
            'F1 Score': 0,
            'Invalid_Labels': invalid_labels['Substantiated']
        },
        'Unsubstantiated': {
            'Label Total': Unsub_Correct_Total,
            'True Classifications': Unsub_True,
            'False Classifications': Unsub_False,
            'Accuracy': round(Unsub_True / Unsub_Correct_Total, 3),
            'Precision': round(Unsub_True / (Unsub_True + Sub_False) if (Unsub_True + Sub_False) > 0 else 0.0, 3),
            'Recall': round(Unsub_True / Unsub_Correct_Total if Unsub_Correct_Total > 0 else 0.0, 3),
            'F1 Score': 0,
            'Invalid_Labels': invalid_labels['Unsubstantiated']
        },
        'Total Accuracy': round((Sub_True + Unsub_True) / G_Total, 3),
        'Balanced Accuracy': 0
    }
    results['Substantiated']['F1 Score'] = round(2 * ((results['Substantiated']['Precision'] * results['Substantiated']['Recall']) / (results['Substantiated']['Precision'] + results['Substantiated']['Recall'])) if (results['Substantiated']['Precision'] + results['Substantiated']['Recall']) > 0 else 0.0, 3)
    results['Unsubstantiated']['F1 Score'] = round(2 * ((results['Unsubstantiated']['Precision'] * results['Unsubstantiated']['Recall']) / (results['Unsubstantiated']['Precision'] + results['Unsubstantiated']['Recall'])) if (results['Unsubstantiated']['Precision'] + results['Unsubstantiated']['Recall']) > 0 else 0.0, 3)

    results['Balanced Accuracy'] = round((results['Substantiated']['Recall'] + results['Unsubstantiated']['Recall']) / 2, 3)

    return results
    
# def count_preds_for_label(type_predictions_dict, label):
#     """
#     Count the number of predictions for a specific label in a results dictionary.
#     """
#     label_count = 0
#     for _, count in type_predictions_dict.items():
#         label_count += count[label]
#     return label_count

# def print_table_label_accuracies(label_accuracies, string_given=False, two_labels=False):
#     if two_labels:
#         if string_given:
#             table_data = [
#                 [model, 
#                 f"{accuracies['unsubstantiate']}", 
#                 f"{accuracies['fully substantiate']}", 
#                 f"{accuracies['overall']}"]
#                 for model, accuracies in label_accuracies.items()
#             ]
#         else:
#             table_data = [
#                 [model, 
#                 f"{accuracies['unsubstantiate'] * 100:.1f}", 
#                 f"{accuracies['fully substantiate'] * 100:.1f}", 
#                 f"{accuracies['overall'] * 100:.1f}"]
#                 for model, accuracies in label_accuracies.items()
#             ]
#     else:
#         if string_given:
#             table_data = [
#                 [model, 
#                 f"{accuracies['unsubstantiate']}", 
#                 f"{accuracies['partially substantiate']}", 
#                 f"{accuracies['fully substantiate']}", 
#                 f"{accuracies['overall']}"]
#                 for model, accuracies in label_accuracies.items()
#             ]
#         else:
#             table_data = [
#                 [model, 
#                 f"{accuracies['unsubstantiate'] * 100:.1f}", 
#                 f"{accuracies['partially substantiate'] * 100:.1f}", 
#                 f"{accuracies['fully substantiate'] * 100:.1f}", 
#                 f"{accuracies['overall'] * 100:.1f}"]
#                 for model, accuracies in label_accuracies.items()
#             ]

#     # Define headers
#     if two_labels:
#         headers = ['Model', 'Un', 'Fully', 'Overall']
#     else:
#         headers = ['Model', 'Un', 'Partially', 'Fully', 'Overall']

#     # Display the table
#     print(tabulate(table_data, headers=headers, tablefmt='pretty'))

# def calc_preds_per_error_type(df):
#     """
#     Calculates the number total, correct and false class predictions for the unsubstantiated rows per error type in the DataFrame.
#     """
#     error_types = list(df['Error Type'][df['Error Type'].notna()].unique())
#     error_types.sort()

#     preds_per_error_type = {
#         error_type: {
#             "total": 0,
#             "correct_class": 0,
#             "false_class": 0
#         } for error_type in error_types
#     }

#     for _, row in df[df['Label'] == 'unsubstantiate'].iterrows():
#         assert pd.notna(row['Error Type']), f"Error Type is NaN for row: {row}"
#         error_type = row['Error Type']
#         preds_per_error_type[error_type]['total'] += 1
#         if row['Model Classification Label'] == row['Label']:
#             preds_per_error_type[error_type]['correct_class'] += 1
#         else:
#             preds_per_error_type[error_type]['false_class'] += 1

#     return preds_per_error_type

def display_model_results_table(model_results_dict, use_pandas=True):
    """
    Display model evaluation results as a formatted table.
    
    Parameters:
    model_results_dict (dict): Dictionary where keys are model names and values are result dictionaries
                              containing accuracy, precision, recall, specificity, and f1_score
    
    Returns:
    pd.DataFrame: DataFrame with the results formatted as a table
    """
    # Extract the metrics we want to display
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score']
    
    # Create a list to store the table data
    table_data = []
    
    for model_name, results in model_results_dict.items():
        row = [model_name]  # Start with model name
        for metric in metrics:
            if metric in results:
                # Format as decimal with 3 decimal places
                row.append(f"{results[metric]:.3f}")
            else:
                row.append("N/A")
        table_data.append(row)
    
    # Create column headers
    headers = ['Model'] + [metric.capitalize() for metric in metrics]
    
    # Display the table using tabulate
    if not use_pandas:
        print("Model Performance Comparison")
        print("=" * 70)
        print(tabulate(table_data, headers=headers, tablefmt='grid', stralign='center'))
    
    # Also create and return a DataFrame for further analysis
    df_data = []
    for model_name, results in model_results_dict.items():
        row = {'Model': model_name}
        for metric in metrics:
            if metric in results:
                row[metric.capitalize()] = results[metric]
            else:
                row[metric.capitalize()] = None
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.set_index('Model', inplace=True)
    
    if use_pandas:
        display(df)

def get_preds_results(results):
    return {
        "Unsubstantiated": { # negative class
            "preds": results['TN'] + results['FN'],
            "correct_preds": results['TN'],
            "correct_total": results['N (Unsubstantiated)'],
        },
        "Substantiated": { # positive class
            "preds": results['TP'] + results['FP'],
            "correct_preds": results['TP'],
            "correct_total": results['P (Substantiated)'],
        },
    }

def eval_predictions_per_attribute_value(df, attribute, include_relabelled_partially, group_numbers_from=False):
    results = {}
    results['Total'] = eval_predictions(df, include_relabelled_partially=include_relabelled_partially)
    
    # Get unique attribute values and sort them
    attribute_values = df[attribute].unique()
    
    # Try to sort numerically first, fall back to alphabetical sorting
    try:
        # Attempt to convert to numbers and sort numerically
        sorted_values = sorted(attribute_values, key=lambda x: float(x))
        values_are_numeric = True
    except (ValueError, TypeError):
        # If conversion fails, sort alphabetically
        sorted_values = sorted(attribute_values, key=lambda x: str(x))
        values_are_numeric = False
    
    # If group_numbers_from is specified and values are numeric, group values
    if group_numbers_from is not False and values_are_numeric:
        # Convert sorted_values to numbers for comparison
        numeric_values = [float(x) for x in sorted_values]
        
        # Process values below the threshold individually
        for value in sorted_values:
            numeric_value = float(value)
            if numeric_value < group_numbers_from:
                df_value = df[df[attribute] == value]
                results_value = eval_predictions(df_value, include_relabelled_partially=include_relabelled_partially)
                results[value] = results_value
        
        # Group values >= threshold together
        values_to_group = [v for v in sorted_values if float(v) >= group_numbers_from]
        if values_to_group:
            # Create a combined dataframe for all values >= threshold
            df_grouped = df[df[attribute].apply(lambda x: float(x) >= group_numbers_from)]
            results_grouped = eval_predictions(df_grouped, include_relabelled_partially=include_relabelled_partially)
            results[f">= {group_numbers_from}"] = results_grouped
    else:
        # Process all values individually (original behavior)
        for value in sorted_values:
            df_value = df[df[attribute] == value]
            results_value = eval_predictions(df_value, include_relabelled_partially=include_relabelled_partially)
            results[value] = results_value
    
    return results

def display_attribute_results_table(results_dict, attribute, use_pandas=True):
    """
    Display a table of model results by attribute values showing only accuracy.
    
    Parameters:
    results_dict: Dictionary with structure {model_name: {attribute_value: {results...}}}
    use_pandas: If True, use pandas DataFrame display; if False, use tabulate
    
    Returns:
    pandas.DataFrame: The results table
    """
    
    # Get all unique attribute values from the first model
    first_model = list(results_dict.keys())[0]
    attribute_values = list(results_dict[first_model].keys())
    
    # Create data for the table
    table_data = []
    for model_name, model_results in results_dict.items():
        row = [model_name]
        accuracies = []
        for attr_value in attribute_values:
            if attr_value in model_results and 'accuracy' in model_results[attr_value]:
                accuracy = model_results[attr_value]['accuracy']
                row.append(f"{accuracy * 100:.1f}%")
                accuracies.append(accuracy)
            else:
                row.append("N/A")
                accuracies.append(None)
        
        table_data.append(row)
    
    # Create DataFrame
    columns = ['Model', 'Total'] + [f'"{value_name}"' for value_name in attribute_values if value_name != 'Total']
    df = pd.DataFrame(table_data, columns=columns)
    df.set_index('Model', inplace=True)
    
    print(f"Results for attribute '{attribute}':")
    # Display based on preference
    if use_pandas:
        display(df)
    else:
        print(tabulate(df, headers=df.columns, tablefmt='grid', stralign='center'))

def display_attribute_differences_to_total_table(results_dict, attribute, use_pandas=True):
    """
    Display a table showing the difference between each attribute value accuracy and the total accuracy.
    
    Parameters:
    results_dict: Dictionary with structure {model_name: {attribute_value: {results...}}}
    attribute: The attribute name being analyzed
    use_pandas: If True, use pandas DataFrame display; if False, use tabulate
    
    Returns:
    pandas.DataFrame: The results table with differences to total
    """
    
    # Get all unique attribute values from the first model (excluding 'Total')
    first_model = list(results_dict.keys())[0]
    attribute_values = [val for val in results_dict[first_model].keys() if val != 'Total']
    
    # Create data for the table
    table_data = []
    for model_name, model_results in results_dict.items():
        row = [model_name]
        
        # Get total accuracy for this model
        total_accuracy = None
        if 'Total' in model_results and 'accuracy' in model_results['Total']:
            total_accuracy = model_results['Total']['accuracy']
        
        # Calculate differences for each attribute value
        for attr_value in attribute_values:
            if (attr_value in model_results and 
                'accuracy' in model_results[attr_value] and 
                total_accuracy is not None):
                
                attr_accuracy = model_results[attr_value]['accuracy']
                difference = attr_accuracy - total_accuracy
                row.append(f"{difference * 100:+.1f}%")
            else:
                row.append("N/A")
        
        table_data.append(row)
    
    # Create DataFrame
    columns = ['Model'] + [f'"{value_name}"' for value_name in attribute_values]
    df = pd.DataFrame(table_data, columns=columns)
    df.set_index('Model', inplace=True)
    
    print(f"Differences to Total for attribute '{attribute}' (positive = better than total):")
    # Display based on preference
    if use_pandas:
        display(df)
    else:
        print(tabulate(df, headers=df.columns, tablefmt='grid', stralign='center'))

def eval_attribute_subset_vs_rest(df, attribute, attribute_values):
    # Create subset and rest DataFrames
    subset_df = df[df[attribute].isin(attribute_values)]
    rest_df = df[~df[attribute].isin(attribute_values)]
    
    subset_results = eval_predictions(subset_df, include_relabelled_partially=True)
    rest_results = eval_predictions(rest_df, include_relabelled_partially=True)
    
    return {
        'Subset': {
            'Total': subset_results['G (Total)'],
            'Correct': subset_results['TP'] + subset_results['TN'],
        },
        'Rest': {
            'Total': rest_results['G (Total)'],
            'Correct': rest_results['TP'] + rest_results['TN'],
        },
    }

def calc_significance_of_accuracy_difference(attribute_subset_rest_results):
    results = {}

    x1 = attribute_subset_rest_results['Subset']['Correct']
    x2 = attribute_subset_rest_results['Rest']['Correct']
    n1 = attribute_subset_rest_results['Subset']['Total']
    n2 = attribute_subset_rest_results['Rest']['Total']

    z_stat, p_value = proportions_ztest([x1, x2], [n1, n2])
    results['z_test'] = {
        'z_statistic': z_stat,
        'p_value': p_value
    }

    table = [[x1, n1 - x1],
         [x2, n2 - x2]]
    odds_ratio, p_value = fisher_exact(table)
    results['fisher_exact'] = {
        'odds_ratio': odds_ratio,
        'p_value': p_value
    }

    return results

def display_significance_test_results(significance_results, required_p=0.1):
    # Create table data
    table_data = []
    
    for attribute_value, test_results in significance_results.items():
        # Check if this is the expected structure with statistical test results
        if 'z_test' in test_results and 'fisher_exact' in test_results:
            z_test = test_results['z_test']
            fisher_test = test_results['fisher_exact']
            
            row = [
                attribute_value,
                f"{z_test['z_statistic']:.4f}",
                f"{z_test['p_value']:.4f}",
                f"{fisher_test['odds_ratio']:.4f}",
                f"{fisher_test['p_value']:.4f}"
            ]
            table_data.append(row)

    # Create DataFrame
    columns = ['Attribute Value', 'Z-statistic', 'Z-test P-value', 'Odds Ratio', 'Fisher P-value']
    df = pd.DataFrame(table_data, columns=columns)
    df.set_index('Attribute Value', inplace=True)
    
    print("Statistical Significance Tests:")
    print("=" * 50)
    
    # Apply color styling to the DataFrame
    def color_p_values(val):
        """Color p-values based on significance level"""
        try:
            p_val = float(val)
            if p_val < required_p:
                return 'background-color: darkgreen; color: white'
            else:
                return 'background-color: darkred; color: white'
        except (ValueError, TypeError):
            return ''
    
    # Apply styling only to p-value columns
    styled_df = df.style.applymap(color_p_values, subset=['Z-test P-value', 'Fisher P-value'])
    display(styled_df)
    
    return df

def get_attribute_value_groups(df, attribute, group_numbers_from=False):
    attribute_values = df[attribute].unique()
    
    # Try to sort numerically first, fall back to alphabetical sorting
    try:
        # Attempt to convert to numbers and sort numerically
        sorted_values = sorted(attribute_values, key=lambda x: float(x))
        values_are_numeric = True
    except (ValueError, TypeError):
        # If conversion fails, sort alphabetically
        sorted_values = sorted(attribute_values, key=lambda x: str(x))
        values_are_numeric = False
    
    groups = []
    
    # If group_numbers_from is specified and values are numeric, group values
    if group_numbers_from is not False and values_are_numeric:
        # Process values below threshold individually
        for value in sorted_values:
            if float(value) < group_numbers_from:
                groups.append((str(value), [value]))
        
        # Group values >= threshold together
        values_to_group = [value for value in sorted_values if float(value) >= group_numbers_from]
        if values_to_group:
            group_name = f">= {group_numbers_from}"
            groups.append((group_name, values_to_group))
    else:
        # Process all values individually (original behavior)
        for attribute_value in attribute_values:
            groups.append((str(attribute_value), [attribute_value]))
    
    return groups
