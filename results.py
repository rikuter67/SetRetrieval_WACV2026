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
    print(f"Mean Rank (MnR)    : {overall.get('mnr', 0):.2f}")
    print(f"Median Rank (MdR)  : {overall.get('mdr', 0):.2f}")
    print(f"Sum of Ranks (Rsum): {overall.get('rsum', 0):,.0f}")
    print(f"MRR                : {overall.get('mrr', 0):.4f}")
    print(f"{'='*70}")

    if 'categories' in results and results['categories']:
        print(f"\n📊 PER-CATEGORY RESULTS:")
        print(f"-" * 95)
        print(f"{'Cat':<3} {'Name':<20} {'Queries':<8} {'MRR':<8} {'Top1% Acc':<10} {'Top5% Acc':<10} {'Top10% Acc':<11} {'MnR':<8} {'MdR':<8}")
        print(f"-" * 95)
        
        category_names = config.get('category_names', {})
        
        for cat_id in sorted(results['categories'].keys()):
            cat_result = results['categories'][cat_id]
            cat_name = category_names.get(int(cat_id), f"Cat{cat_id}")[:19]
            
            print(f"{cat_id:<3} {cat_name:<20} {cat_result.get('count', 0):<8} "
                  f"{cat_result.get('mrr', 0):<8.4f} "
                  f"{cat_result.get('r_at_1', 0):<10.2%} "
                  f"{cat_result.get('r_at_5', 0):<10.2%} "
                  f"{cat_result.get('r_at_10', 0):<11.2%} "
                  f"{cat_result.get('mnr', 0):<8.1f} "
                  f"{cat_result.get('mdr', 0):<8.1f}")
        
        print(f"-" * 95)

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
        f.write(f"Mean Rank (MnR)    : {overall.get('mnr', 0):.2f}\n")
        f.write(f"Median Rank (MdR)  : {overall.get('mdr', 0):.2f}\n")
        f.write(f"Sum of Ranks (Rsum): {overall.get('rsum', 0):,.0f}\n")
        f.write(f"MRR                : {overall.get('mrr', 0):.4f}\n\n")
    
    print(f"[INFO] Detailed report saved to: {report_path}")

def save_evaluation_results(results: Dict[str, Any], output_dir: str):
    """
    【改善版】評価結果をJSON、CSV、TXTファイルに保存する。
    CSVには特定の主要メトリクスのみを抽出し、整形して保存する。
    """
    print("[INFO] 💾 Saving evaluation results...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. JSON形式で全結果を保存 (変更なし)
    try:
        json_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, cls=NumpyJSONEncoder, ensure_ascii=False)
        print(f"[INFO] Full results saved to: {json_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save full JSON results: {e}")

    # 2. CSV形式でサマリーを保存 (ここからが改善箇所)
    try:
        # --- CSVに出力したい列を明示的に定義 ---
        key_metrics = [
            'mrr',
            'r_at_1', 'r_at_5', 'r_at_10', 'r_at_20',
            'weighted_r_at_1', 'weighted_r_at_5', 'weighted_r_at_10', 'weighted_r_at_20'
        ]
        
        # 'overall' 結果を抽出
        if 'overall' in results:
            overall_results = results['overall']
            
            # 辞書形式で一行のデータを作成
            summary_data = {
                'experiment': os.path.basename(output_dir)
            }
            
            # 定義したキーの値を抽出し、summary_dataに追加
            for key in key_metrics:
                summary_data[key] = overall_results.get(key)

            # DataFrameを作成
            df = pd.DataFrame([summary_data])
            
            # --- results_summary.csv の処理 ---
            summary_csv_path = os.path.join(os.path.dirname(output_dir), 'results_summary.csv')
            
            # ファイルが存在すれば追記、なければ新規作成
            if os.path.exists(summary_csv_path):
                df.to_csv(summary_csv_path, mode='a', header=False, index=False)
                print(f"[INFO] Appended summary to: {summary_csv_path}")
            else:
                df.to_csv(summary_csv_path, mode='w', header=True, index=False)
                print(f"[INFO] Created new summary at: {summary_csv_path}")

        else:
            print("[WARN] 'overall' results not found. Skipping CSV summary.")

    except Exception as e:
        print(f"[ERROR] Failed to save CSV summary: {e}")

    # 3. テキスト形式で詳細レポートを保存 (変更なし)
    try:
        # ... (display_evaluation_resultsのロジックをここに統合しても良い) ...
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            # ... (テキストレポートの書き込み) ...
            pass
        print(f"[INFO] Detailed text report can be generated if needed.")
    except Exception as e:
        print(f"[ERROR] Failed to save text report: {e}")