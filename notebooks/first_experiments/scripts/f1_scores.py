#!/usr/bin/env python3
"""
Script to calculate precision, recall, F1 score for each label and balanced accuracy
from the results data in all_results.json
"""

import json
import numpy as np
from collections import defaultdict


def calculate_metrics_from_confusion_matrix(type_predictions, labels):
    """
    Calculate precision, recall, F1 score for each label and balanced accuracy
    from confusion matrix data.
    
    Args:
        type_predictions: Dictionary containing confusion matrix data
        labels: List of label names to calculate metrics for
    
    Returns:
        Dictionary containing metrics for each label and overall balanced accuracy
    """
    metrics = {}
    
    # Initialize confusion matrix
    n_labels = len(labels)
    confusion_matrix = np.zeros((n_labels, n_labels))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    # Fill confusion matrix
    for true_label, predictions in type_predictions.items():
        if true_label not in label_to_idx:
            continue
        true_idx = label_to_idx[true_label]
        
        for pred_label, count in predictions.items():
            if pred_label in label_to_idx:
                pred_idx = label_to_idx[pred_label]
                confusion_matrix[true_idx, pred_idx] = count
    
    # Calculate metrics for each label
    for i, label in enumerate(labels):
        # True positives: diagonal element
        tp = confusion_matrix[i, i]
        
        # False positives: sum of column i minus diagonal element
        fp = np.sum(confusion_matrix[:, i]) - tp
        
        # False negatives: sum of row i minus diagonal element
        fn = np.sum(confusion_matrix[i, :]) - tp
        
        # True negatives: total minus tp, fp, fn
        tn = np.sum(confusion_matrix) - tp - fp - fn
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[label] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn)
        }
    
    # Calculate balanced accuracy (average of per-class recalls)
    recalls = [metrics[label]['recall'] for label in labels]
    balanced_accuracy = np.mean(recalls)
    
    metrics['balanced_accuracy'] = round(balanced_accuracy, 4)
    metrics['macro_f1'] = round(np.mean([metrics[label]['f1_score'] for label in labels]), 4)
    
    return metrics


def process_results_data(data):
    """
    Process the entire results data structure and calculate metrics for all configurations.
    
    Args:
        data: The loaded JSON data from all_results.json
    
    Returns:
        Dictionary containing calculated metrics for all configurations
    """
    results = {}
    
    # Define label sets
    three_labels = ['unsubstantiate', 'partially substantiate', 'fully substantiate']
    two_labels = ['unsubstantiate', 'fully substantiate']
    
    for dataset_name, dataset_data in data.items():
        results[dataset_name] = {}
        
        for model_type, model_data in dataset_data.items():
            results[dataset_name][model_type] = {}
            
            for model_name, model_results in model_data.items():
                results[dataset_name][model_type][model_name] = {}
                
                for eval_type, eval_data in model_results.items():
                    if 'type_predictions' not in eval_data:
                        continue
                    
                    # Determine which labels to use
                    if eval_type.startswith('all_labels'):
                        labels = three_labels
                    elif eval_type.startswith('two_labels'):
                        labels = two_labels
                    else:
                        continue
                    
                    # Calculate metrics
                    metrics = calculate_metrics_from_confusion_matrix(
                        eval_data['type_predictions'], 
                        labels
                    )
                    
                    # Add original accuracy for comparison
                    metrics['original_accuracy'] = round(eval_data.get('accuracy', 0.0), 4)
                    metrics['total_samples'] = eval_data.get('total', 0)
                    
                    results[dataset_name][model_type][model_name][eval_type] = metrics
    
    return results


def print_metrics_summary(results):
    """
    Print a formatted summary of the calculated metrics.
    """
    print("=" * 80)
    print("CLASSIFICATION METRICS SUMMARY")
    print("=" * 80)
    
    for dataset_name, dataset_data in results.items():
        print(f"\n{'='*20} {dataset_name.upper()} {'='*20}")
        
        for model_type, model_data in dataset_data.items():
            print(f"\n{'-'*10} {model_type.replace('_', ' ').title()} {'-'*10}")
            
            for model_name, model_results in model_data.items():
                print(f"\nModel: {model_name}")
                
                for eval_type, metrics in model_results.items():
                    print(f"\n  {eval_type.replace('_', ' ').title()}:")
                    print(f"    Original Accuracy: {metrics['original_accuracy']:.4f}")
                    print(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
                    print(f"    Macro F1 Score: {metrics['macro_f1']:.4f}")
                    print(f"    Total Samples: {metrics['total_samples']}")
                    
                    # Print per-label metrics
                    for label in metrics:
                        if isinstance(metrics[label], dict) and 'precision' in metrics[label]:
                            label_metrics = metrics[label]
                            print(f"    {label.title()}:")
                            print(f"      Precision: {label_metrics['precision']:.4f}")
                            print(f"      Recall: {label_metrics['recall']:.4f}")
                            print(f"      F1 Score: {label_metrics['f1_score']:.4f}")


def save_metrics_to_json(results, output_file):
    """
    Save the calculated metrics to a JSON file.
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to: {output_file}")


def main():
    """
    Main function to load data, calculate metrics, and display results.
    """
    # Load the results data
    input_file = '/home/ibelter/master_thesis/citation-verification/data/_first_experiments/all_results.json'
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find file {input_file}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse JSON file: {e}")
        return
    
    # Calculate metrics
    print("Calculating precision, recall, F1 scores, and balanced accuracy...")
    results = process_results_data(data)
    
    # Print summary
    print_metrics_summary(results)
    
    # Save results
    output_file = '/home/ibelter/master_thesis/citation-verification/data/_first_experiments/calculated_metrics.json'
    save_metrics_to_json(results, output_file)
    
    # Print some key findings
    print("\n" + "="*80)
    print("KEY FINDINGS SUMMARY")
    print("="*80)
    
    # Find best performing configurations
    best_configs = []
    for dataset_name, dataset_data in results.items():
        for model_type, model_data in dataset_data.items():
            for model_name, model_results in model_data.items():
                for eval_type, metrics in model_results.items():
                    best_configs.append({
                        'dataset': dataset_name,
                        'model_type': model_type,
                        'model_name': model_name,
                        'eval_type': eval_type,
                        'balanced_accuracy': metrics['balanced_accuracy'],
                        'macro_f1': metrics['macro_f1'],
                        'original_accuracy': metrics['original_accuracy']
                    })
    
    # Sort by balanced accuracy
    best_configs.sort(key=lambda x: x['balanced_accuracy'], reverse=True)
    
    print("\nTop 5 configurations by Balanced Accuracy:")
    for i, config in enumerate(best_configs[:5], 1):
        print(f"{i}. {config['dataset']} - {config['model_type']} - {config['model_name']} - {config['eval_type']}")
        print(f"   Balanced Accuracy: {config['balanced_accuracy']:.4f}, Macro F1: {config['macro_f1']:.4f}, Original Accuracy: {config['original_accuracy']:.4f}")


if __name__ == "__main__":
    main()
