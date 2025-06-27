"""
results.py - Functions for saving and displaying evaluation results.
"""
import os
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, Any

from config import get_dataset_config

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)

def display_evaluation_results(results: Dict[str, Any]):
    """Display evaluation results in a formatted table."""
    if not results or 'overall' not in results:
        print("[ERROR] No results to display")
        return

    dataset_type = results.get('dataset', 'N/A')
    config = get_dataset_config(dataset_type)
    overall = results['overall']

    print(f"\n{'='*70}")
    print(f"🎯 EVALUATION RESULTS - {dataset_type}")
    print(f"{'='*70}")
    print(f"Total Queries: {overall.get('total_queries', 0):,}")
    print(f"Successful Predictions: {overall.get('successful_predictions', 0):,}")
    print(f"-" * 70)
    print(f"Top-1% Accuracy  : {overall.get('r_at_1', 0):.2%}")
    print(f"Top-5% Accuracy  : {overall.get('r_at_5', 0):.2%}")
    print(f"Top-10% Accuracy : {overall.get('r_at_10', 0):.2%}")
    print(f"Top-20% Accuracy : {overall.get('r_at_20', 0):.2%}")  # ✅ Top-20% 追加
    print(f"Mean Rank (MnR)    : {overall.get('mnr', 0):.2f}")
    print(f"Median Rank (MdR)  : {overall.get('mdr', 0):.2f}")
    print(f"Sum of Ranks (Rsum): {overall.get('rsum', 0):,.0f}")
    print(f"MRR                : {overall.get('mrr', 0):.4f}")
    print(f"{'='*70}")

    if 'categories' in results and results['categories']:
        print(f"\n📊 PER-CATEGORY RESULTS:")
        print(f"-" * 110)  # ✅ 幅を拡張
        print(f"{'Cat':<3} {'Name':<20} {'Queries':<8} {'MRR':<8} {'Top1% Acc':<10} {'Top5% Acc':<10} {'Top10% Acc':<11} {'Top20% Acc':<11} {'MnR':<8} {'MdR':<8}")  # ✅ Top20% 追加
        print(f"-" * 110)  # ✅ 幅を拡張
        
        category_names = config.get('category_names', {})
        
        for cat_id in sorted(results['categories'].keys()):
            cat_result = results['categories'][cat_id]
            cat_name = category_names.get(int(cat_id), f"Cat{cat_id}")[:19]
            
            print(f"{cat_id:<3} {cat_name:<20} {cat_result.get('count', 0):<8} "
                  f"{cat_result.get('mrr', 0):<8.4f} "
                  f"{cat_result.get('r_at_1', 0):<10.2%} "
                  f"{cat_result.get('r_at_5', 0):<10.2%} "
                  f"{cat_result.get('r_at_10', 0):<11.2%} "
                  f"{cat_result.get('r_at_20', 0):<11.2%} "  # ✅ Top20% 追加
                  f"{cat_result.get('mnr', 0):<8.1f} "
                  f"{cat_result.get('mdr', 0):<8.1f}")
        
        print(f"-" * 110)  # ✅ 幅を拡張

def save_results_csv(results: Dict[str, Any], config: Dict[str, Any], csv_path: str):
    """Save results as CSV file."""
    if 'categories' not in results:
        return
    
    data = []
    category_names = config.get('category_names', {})
    
    for cat_id_str in sorted(results['categories'].keys()):
        cat_id = int(cat_id_str)
        cat_result = results['categories'][cat_id_str]
        cat_name = category_names.get(cat_id, f"Category_{cat_id}")
        
        data.append({
            'Category_ID': cat_id,
            'Category_Name': cat_name,
            'Query_Count': cat_result.get('count', 0),
            'Gallery_Size': cat_result.get('gallery_size', 0),
            'MRR': cat_result.get('mrr', 0),
            'Top1_Accuracy': cat_result.get('r_at_1', 0),
            'Top5_Accuracy': cat_result.get('r_at_5', 0),
            'Top10_Accuracy': cat_result.get('r_at_10', 0),
            'Top20_Accuracy': cat_result.get('r_at_20', 0),  # ✅ Top20% 追加
            'MnR': cat_result.get('mnr', 0),
            'MdR': cat_result.get('mdr', 0),
            'Rsum': cat_result.get('rsum', 0)
        })
    
    overall = results['overall']
    data.append({
        'Category_ID': 'Overall',
        'Category_Name': 'All Categories',
        'Query_Count': overall.get('total_queries', 0),
        'Gallery_Size': sum(results['categories'][cat].get('gallery_size', 0) for cat in results['categories']),
        'MRR': overall.get('mrr', 0),
        'Top1_Accuracy': overall.get('r_at_1', 0),
        'Top5_Accuracy': overall.get('r_at_5', 0),
        'Top10_Accuracy': overall.get('r_at_10', 0),
        'Top20_Accuracy': overall.get('r_at_20', 0),  # ✅ Top20% 追加
        'MnR': overall.get('mnr', 0),
        'MdR': overall.get('mdr', 0),
        'Rsum': overall.get('rsum', 0)
    })
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] CSV summary saved to: {csv_path}")

def save_text_report(results: Dict[str, Any], report_path: str):
    """Save detailed text report."""
    with open(report_path, 'w', encoding='utf-8') as f:
        dataset_type = results.get('dataset', 'N/A')
        f.write("SET RETRIEVAL EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {dataset_type}\n")
        f.write(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        overall = results.get('overall', {})
        f.write("OVERALL RESULTS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Queries: {overall.get('total_queries', 0):,}\n")
        f.write("METRICS\n")
        f.write("-" * 10 + "\n")
        f.write(f"Top-1% Accuracy  : {overall.get('r_at_1', 0):.2%}\n")
        f.write(f"Top-5% Accuracy  : {overall.get('r_at_5', 0):.2%}\n")
        f.write(f"Top-10% Accuracy : {overall.get('r_at_10', 0):.2%}\n")
        f.write(f"Top-20% Accuracy : {overall.get('r_at_20', 0):.2%}\n")  # ✅ Top20% 追加
        f.write(f"Mean Rank (MnR)    : {overall.get('mnr', 0):.2f}\n")
        f.write(f"Median Rank (MdR)  : {overall.get('mdr', 0):.2f}\n")
        f.write(f"Sum of Ranks (Rsum): {overall.get('rsum', 0):,.0f}\n")
        f.write(f"MRR                : {overall.get('mrr', 0):.4f}\n\n")
    
    print(f"[INFO] Detailed report saved to: {report_path}")

def save_evaluation_results(results: Dict[str, Any], config: Dict[str, Any], output_dir: str):
    """Save all evaluation results to files."""
    print(f"[INFO] 💾 Saving evaluation results...")
    
    json_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyJSONEncoder)
    print(f"[INFO] Results saved to: {json_path}")
    
    csv_path = os.path.join(output_dir, 'results_summary.csv')
    save_results_csv(results, config, csv_path)
    
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    save_text_report(results, report_path)
    
    print(f"[INFO] ✅ All results saved to: {output_dir}")

def compare_results(results_dict: Dict[str, Dict[str, Any]], save_path: str = None):
    """
    Compare multiple experimental results and display side-by-side.
    
    Args:
        results_dict: Dictionary of {experiment_name: results}
        save_path: Optional path to save comparison table
    """
    if not results_dict:
        print("[ERROR] No results to compare")
        return
    
    print(f"\n{'='*100}")
    print(f"🔍 EXPERIMENT COMPARISON")
    print(f"{'='*100}")
    
    # Header
    experiments = list(results_dict.keys())
    header = f"{'Metric':<20}"
    for exp in experiments:
        header += f"{exp:<15}"
    print(header)
    print("-" * 100)
    
    # Metrics to compare
    metrics = [
        ('Top-1% Acc', 'r_at_1', '.2%'),
        ('Top-5% Acc', 'r_at_5', '.2%'),
        ('Top-10% Acc', 'r_at_10', '.2%'),
        ('Top-20% Acc', 'r_at_20', '.2%'),  # ✅ Top20% 追加
        ('MnR', 'mnr', '.2f'),
        ('MdR', 'mdr', '.2f'),
        ('MRR', 'mrr', '.4f')
    ]
    
    comparison_data = []
    
    for metric_name, metric_key, format_str in metrics:
        row = f"{metric_name:<20}"
        row_data = {'Metric': metric_name}
        
        for exp in experiments:
            value = results_dict[exp]['overall'].get(metric_key, 0)
            if format_str == '.2%':
                formatted_value = f"{value:.2%}"
            elif format_str == '.2f':
                formatted_value = f"{value:.2f}"
            elif format_str == '.4f':
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            row += f"{formatted_value:<15}"
            row_data[exp] = value
        
        print(row)
        comparison_data.append(row_data)
    
    print("-" * 100)
    
    # Calculate improvements
    if len(experiments) >= 2:
        baseline = experiments[0]
        print(f"\n📈 IMPROVEMENTS over {baseline}:")
        print("-" * 50)
        
        for metric_name, metric_key, format_str in metrics:
            baseline_val = results_dict[baseline]['overall'].get(metric_key, 0)
            row = f"{metric_name:<20}"
            
            for exp in experiments[1:]:
                current_val = results_dict[exp]['overall'].get(metric_key, 0)
                
                if metric_key in ['mnr', 'mdr']:  # Lower is better
                    improvement = baseline_val - current_val
                    if baseline_val > 0:
                        improvement_pct = (improvement / baseline_val) * 100
                    else:
                        improvement_pct = 0
                else:  # Higher is better
                    improvement = current_val - baseline_val
                    if baseline_val > 0:
                        improvement_pct = (improvement / baseline_val) * 100
                    else:
                        improvement_pct = 0
                
                sign = "+" if improvement > 0 else ""
                row += f"{sign}{improvement_pct:.1f}%"
                row += " " * (15 - len(f"{sign}{improvement_pct:.1f}%"))
            
            print(row)
    
    # Save comparison table if requested
    if save_path:
        df = pd.DataFrame(comparison_data)
        df.to_csv(save_path, index=False)
        print(f"\n[INFO] Comparison table saved to: {save_path}")
    
    print(f"{'='*100}")