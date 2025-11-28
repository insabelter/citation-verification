import pandas as pd
import json
import re

# ------------ DataFrame Loading and Reshaping ------------

def _sort_df(df):
    df = df.sort_values(by=['Citing Article ID', 'Reference Article ID'], ascending=[True, True]).reset_index(drop=True)
    return df

def _remove_json_colons(json_text):
    if json_text and '{' in json_text and '}' in json_text:
        json_text = json_text[json_text.find('{'):json_text.rfind('}') + 1]
    return json_text

def _find_label_within_non_json_text(text):
    if not 'label' in text.lower():
        return None
    if 'unsubstantiated' in text.lower():
        return 'Unsubstantiated'
    elif 'substantiated' in text.lower():
        return 'Substantiated'
    return None

# Add extra columns for the model classification label and explanation by extracting the information from the JSON
# If the JSON is misformed due to leading ```json and trailing ``` then remove them
# Make sure that correct label and model label are both lower case and do not end with d (unsubstaniate instead of unsubstantiated)
def _reshape_model_classification(df):
    for index, row in df.iterrows():
        if row['Reference Article Downloaded'] == 'Yes':
            try:
                row['Model Classification'] = _remove_json_colons(row['Model Classification'])
                model_classification = json.loads(row['Model Classification'])
                df.at[row.name, 'Model Classification Label'] = model_classification['label']
                df.at[row.name, 'Model Classification Explanation'] = model_classification['explanation']
            except (json.JSONDecodeError, KeyError) as e:
                label = _find_label_within_non_json_text(row['Model Classification'])
                if label:
                    print(f"Using extracted label ({label}) from non JSON text!")
                    df.at[row.name, 'Model Classification Label'] = label
                    df.at[row.name, 'Model Classification Explanation'] = row['Model Classification']
                else:
                    print(f"Row {index} Model Classification could not be decoded: {e}")
                    print(row['Model Classification'])
                    df.at[row.name, 'Model Classification Label'] = None
                    df.at[row.name, 'Model Classification Explanation'] = None
        else:
            df.at[row.name, 'Model Classification Label'] = None
            df.at[row.name, 'Model Classification Explanation'] = None
        df.at[row.name, 'Label'] = df.at[row.name, 'Label']
    return df

def _add_claims_to_substantiate_min_max(df):
    def extract_min_max(val):
        if isinstance(val, str):
            match = re.match(r"\[(\d+)(?:-(\d+))?\]", val)
            if match:
                min_val = int(match.group(1))
                max_val = int(match.group(2)) if match.group(2) else min_val
                return min_val, max_val
        return None, None

    min_max = df['Amount Claims to Substantiate'].apply(extract_min_max)
    df['Amount Claims to Substantiate: Minimum'] = min_max.apply(lambda x: x[0])
    df['Amount Claims to Substantiate: Maximum'] = min_max.apply(lambda x: x[1])
    return df

def load_df(chunking, only_text, model, ai_prompt=False):
    path = f"../data/dfs/{'only_text_' if only_text else ''}{chunking}/{model}/{'AI_prompt/' if ai_prompt else ''}ReferenceErrorDetection_data_with_prompt_results.pkl"
    df = pd.read_pickle(path)
    return df

def load_df_for_analysis(chunking, only_text, model, ai_prompt=False):
    df = load_df(chunking, only_text, model, ai_prompt=ai_prompt)
    df = _add_claims_to_substantiate_min_max(df)
    df = _sort_df(df)
    df = _reshape_model_classification(df)
    return df

# ------------ Functions for Results Evaluation ------------

def eval_predictions(df, include_relabelled_partially=True, include_not_originally_downloaded=True):
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

    results = {
        'Total': G_Total,
        'Substantiated': {
            'Label Total': Sub_Correct_Total,
            'True Classifications': Sub_True,
            'False Classifications': Sub_False,
            'Precision': round(Sub_True / (Sub_True + Unsub_False) if (Sub_True + Unsub_False) > 0 else 0.0, 3),
            'Recall': round(Sub_True / Sub_Correct_Total if Sub_Correct_Total > 0 else 0.0, 3),
            'F1-Score': 0,
            'Invalid_Labels': invalid_labels['Substantiated']
        },
        'Unsubstantiated': {
            'Label Total': Unsub_Correct_Total,
            'True Classifications': Unsub_True,
            'False Classifications': Unsub_False,
            'Precision': round(Unsub_True / (Unsub_True + Sub_False) if (Unsub_True + Sub_False) > 0 else 0.0, 3),
            'Recall': round(Unsub_True / Unsub_Correct_Total if Unsub_Correct_Total > 0 else 0.0, 3),
            'F1-Score': 0,
            'Invalid_Labels': invalid_labels['Unsubstantiated']
        },
        'Accuracy': round((Sub_True + Unsub_True) / G_Total if G_Total > 0 else 0.0, 3),
        'Error Rate': round((Sub_False + Unsub_False) / G_Total if G_Total > 0 else 0.0, 3),
        'Balanced Accuracy': 0
    }
    results['Substantiated']['F1-Score'] = round(2 * ((results['Substantiated']['Precision'] * results['Substantiated']['Recall']) / (results['Substantiated']['Precision'] + results['Substantiated']['Recall'])) if (results['Substantiated']['Precision'] + results['Substantiated']['Recall']) > 0 else 0.0, 3)
    results['Unsubstantiated']['F1-Score'] = round(2 * ((results['Unsubstantiated']['Precision'] * results['Unsubstantiated']['Recall']) / (results['Unsubstantiated']['Precision'] + results['Unsubstantiated']['Recall'])) if (results['Unsubstantiated']['Precision'] + results['Unsubstantiated']['Recall']) > 0 else 0.0, 3)

    results['Balanced Accuracy'] = round((results['Substantiated']['Recall'] + results['Unsubstantiated']['Recall']) / 2, 3)

    return results

def eval_predictions_per_attribute_value(df, attribute, include_relabelled_partially, group_numbers_from=False):
    results = {}
    results['Total'] = eval_predictions(df, include_relabelled_partially=include_relabelled_partially)
    
    # Special case for "Claim Contains Number or Formula" attribute
    if attribute == "Claim Contains Number or Formula" and group_numbers_from == "Number/Formula":
        # Group 'No' separately
        df_no = df[df[attribute] == 'No']
        results['No'] = eval_predictions(df_no, include_relabelled_partially=include_relabelled_partially)
        
        # Group 'Number' and 'Formula' together
        df_number_formula = df[df[attribute].isin(['Number', 'Formula'])]
        results['Number/\nFormula'] = eval_predictions(df_number_formula, include_relabelled_partially=include_relabelled_partially)
        
        return results
    
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
        # Process values below the threshold individually
        for value in sorted_values:
            numeric_value = float(value)
            if numeric_value < group_numbers_from:
                df_value = df[df[attribute] == value]
                results_value = eval_predictions(df_value, include_relabelled_partially=include_relabelled_partially)
                results[str(value)] = results_value
        
        # Group values >= threshold together
        values_to_group = [v for v in sorted_values if float(v) >= group_numbers_from]
        if values_to_group:
            # Create a combined dataframe for all values >= threshold using isin
            df_grouped = df[df[attribute].isin(values_to_group)]
            results_grouped = eval_predictions(df_grouped, include_relabelled_partially=include_relabelled_partially)
            results[f"≥ {group_numbers_from}"] = results_grouped
    else:
        # Process all values individually (original behavior)
        for value in sorted_values:
            df_value = df[df[attribute] == value]
            results_value = eval_predictions(df_value, include_relabelled_partially=include_relabelled_partially)
            results[str(value)] = results_value
    
    return results

def eval_per_attribute_value(df, attribute, attribute_values_per_group):
    # attribute_groups: [('1', [1]), ('2', [2]), ('>= 3', [3, 4, 5, 6, 7, 8])]
    results = {}
    for group_name, attribute_values in attribute_values_per_group:
        group_df = df[df[attribute].isin(attribute_values)]
        results[group_name] = eval_predictions(group_df)
    return results

def get_attribute_value_groups(df, attribute, group_numbers_from=False):
    # Special case for "Claim Contains Number or Formula" attribute
    if attribute == "Claim Contains Number or Formula" and group_numbers_from == "Number/Formula":
        return [('No', ['No']), ('Number/\nFormula', ['Number', 'Formula'])]
    
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
            group_name = f"≥ {group_numbers_from}"
            groups.append((group_name, values_to_group))
    else:
        # Process all values individually (original behavior)
        for attribute_value in attribute_values:
            groups.append((str(attribute_value), [attribute_value]))
    
    return groups