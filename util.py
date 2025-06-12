"""
utils.py - Evaluation and Utility Functions for Set Retrieval (WACV 2026)
===========================================================================
Comprehensive evaluation pipeline with corrected metrics, GPU management,
and result visualization for heterogeneous set retrieval.
"""

import os
import json
import pickle
import gzip
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import gc

import json
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, *args, **kwargs):
        return iterator

def evaluate_model(model, test_data, output_dir=None):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained SetRetrievalModel
        test_data: Test dataset
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing evaluation results
    """
    
    print("🔍 Starting evaluation...")
    
    # Initialize metrics
    all_mrr = []
    all_top1 = []
    all_top5 = []
    all_top10 = []
    
    batch_count = 0
    total_queries = 0
    
    # Evaluate on test data
    for batch in test_data:
        try:
            # Get model predictions
            inputs = {
                'query_features': batch['query_features'],
                'query_categories': batch['query_categories']
            }
            predictions = model(inputs, training=False)
            
            # Calculate metrics for this batch
            batch_metrics = calculate_batch_metrics(
                predictions,
                batch['target_features'],
                batch['target_categories']
            )
            
            if batch_metrics['num_queries'] > 0:
                all_mrr.extend(batch_metrics['mrr'])
                all_top1.extend(batch_metrics['top1'])
                all_top5.extend(batch_metrics['top5'])
                all_top10.extend(batch_metrics['top10'])
                total_queries += batch_metrics['num_queries']
            
            batch_count += 1
            
            # Progress update
            if batch_count % 50 == 0:
                print(f"  Processed {batch_count} batches, {total_queries} queries...")
                
        except Exception as e:
            print(f"⚠️ Error in batch {batch_count}: {e}")
            continue
    
    # Calculate overall metrics
    if total_queries > 0:
        results = {
            'overall': {
                'mrr': np.mean(all_mrr),
                'top1': np.mean(all_top1),
                'top5': np.mean(all_top5),
                'top10': np.mean(all_top10),
                'num_queries': total_queries,
                'num_batches': batch_count
            }
        }
        
        print(f"✅ Evaluation completed on {total_queries} queries")
        
        # Save results if output directory is provided
        if output_dir:
            results_path = os.path.join(output_dir, 'evaluation_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"📁 Results saved to: {results_path}")
        
        return results
    else:
        print("❌ No valid queries found for evaluation")
        return {'overall': {'mrr': 0.0, 'top1': 0.0, 'top5': 0.0, 'top10': 0.0}}
        
# =============================================================================
# GPU and Memory Management
# =============================================================================

def setup_gpu_memory():
    """Configure GPU memory growth and optimization"""
    
    print("[INFO] Configuring GPU memory...")
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set visible devices based on CUDA_VISIBLE_DEVICES
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            if cuda_visible:
                print(f"[INFO] Using GPU devices: {cuda_visible}")
            
            print(f"[INFO] ✅ GPU memory configured for {len(gpus)} GPU(s)")
            
            # Print GPU information
            for i, gpu in enumerate(gpus):
                print(f"[INFO] GPU {i}: {gpu.name}")
                
        except RuntimeError as e:
            print(f"[WARN] GPU configuration warning: {e}")
    else:
        print("[INFO] No GPUs found, using CPU")


def clear_memory():
    """Clear memory and run garbage collection"""
    
    gc.collect()
    
    if tf.config.list_physical_devices('GPU'):
        try:
            tf.keras.backend.clear_session()
        except:
            pass


def monitor_memory_usage():
    """Monitor and print memory usage"""
    
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        print(f"[INFO] System memory usage: {memory_percent:.1f}%")
        
        # GPU memory if available
        if tf.config.list_physical_devices('GPU'):
            try:
                from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
                nvmlInit()
                handle = nvmlDeviceGetHandleByIndex(0)
                mem_info = nvmlDeviceGetMemoryInfo(handle)
                gpu_percent = (mem_info.used / mem_info.total) * 100
                print(f"[INFO] GPU memory usage: {gpu_percent:.1f}%")
            except:
                pass
                
    except ImportError:
        pass


# =============================================================================
# Dataset Configuration and Detection
# =============================================================================

DATASET_CONFIGS = {
    'IQON3000': {
        'num_categories': 7,
        'category_range': (1, 7),
        'category_names': {
            1: "トップス系", 2: "アウター系", 3: "ボトムス系",
            4: "ワンピース・ドレス系", 5: "シューズ系",
            6: "バッグ系", 7: "アクセサリー・小物系"
        }
    },
    'DeepFurniture': {
        'num_categories': 11,
        'category_range': (1, 11),
        'category_names': {
            1: "chair", 2: "table", 3: "sofa", 4: "bed", 5: "cabinet",
            6: "lamp", 7: "bookshelf", 8: "desk", 9: "dresser",
            10: "nightstand", 11: "other_furniture"
        }
    }
}


def detect_dataset_type(test_data) -> str:
    """Detect dataset type from test data"""
    
    try:
        # Try to get a sample batch
        sample_batch = next(iter(test_data.take(1)))
        
        # Check category range
        if 'target_categories' in sample_batch:
            categories = sample_batch['target_categories'].numpy()
            max_cat = int(np.max(categories[categories > 0]))
            
            if max_cat <= 7:
                return 'IQON3000'
            else:
                return 'DeepFurniture'
        
    except Exception as e:
        print(f"[WARN] Could not detect dataset type: {e}")
    
    # Default fallback
    return 'IQON3000'


def get_dataset_config(dataset_type: str) -> Dict[str, Any]:
    """Get configuration for dataset type"""
    
    if dataset_type not in DATASET_CONFIGS:
        print(f"[WARN] Unknown dataset type: {dataset_type}, using IQON3000")
        dataset_type = 'IQON3000'
    
    return DATASET_CONFIGS[dataset_type].copy()

class NumpyJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy types.
    Converts np.integer, np.floating, and np.ndarray to native Python types.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)


# =============================================================================
# Data Collection and Gallery Building
# =============================================================================

def collect_test_data(test_data, max_batches: Optional[int] = None) -> Tuple[List, Dict, str]:
    """
    Collect test data and build gallery for evaluation
    
    Args:
        test_data: TensorFlow dataset
        max_batches: Maximum number of batches to process (None for all)
        
    Returns:
        Tuple of (test_items, gallery_by_category, dataset_type)
    """
    
    print("[INFO] 📥 Collecting test data and building gallery...")
    
    # Detect dataset type
    dataset_type = detect_dataset_type(test_data)
    config = get_dataset_config(dataset_type)
    min_cat, max_cat = config['category_range']
    
    print(f"[INFO] Dataset detected: {dataset_type}")
    print(f"[INFO] Categories: {min_cat}-{max_cat}")
    
    # Collect all batches
    all_batches = []
    batch_count = 0

    for batch in tqdm(test_data, desc="Collecting batches"):
        all_batches.append(batch)
        batch_count += 1
        if max_batches and batch_count >= max_batches:
            break
        if batch_count % 50 == 0:
            clear_memory()
    
    print(f"[INFO] Collected {len(all_batches)} batches")
    if not all_batches:
        raise ValueError("No test data collected")
    
    test_items = []
    gallery_by_category = defaultdict(dict)
    
    print("[INFO] Processing batches...")

    
    for batch in tqdm(test_data, desc="Collecting batches"):
        query_features = batch['query_features'].numpy()
        query_categories = batch['query_categories'].numpy()
        target_features = batch['target_features'].numpy()
        target_categories = batch['target_categories'].numpy()
        
        # item_idの取得を安全に行う
        query_ids_val = batch.get('query_item_ids')
        target_ids_val = batch.get('target_item_ids')

        query_ids = query_ids_val.numpy() if query_ids_val is not None else np.arange(len(query_features))
        target_ids = target_ids_val.numpy() if target_ids_val is not None else np.arange(len(target_features))

        batch_size = query_features.shape[0]
        
        # Create test items
        for i in range(batch_size):
            test_items.append({
                'query_features': query_features[i],
                'query_categories': query_categories[i],
                'target_features': target_features[i],
                'target_categories': target_categories[i],
                'query_ids': query_ids[i] if len(query_ids.shape) > 1 else [query_ids[i]],
                'target_ids': target_ids[i] if len(target_ids.shape) > 1 else [target_ids[i]]
            })
            
            # Build gallery from target items
            for j, (feat, cat, item_id) in enumerate(zip(
                target_features[i], target_categories[i], 
                target_ids[i] if len(target_ids.shape) > 1 else [target_ids[i]]
            )):
                # Skip padding items
                if cat == 0 or np.all(feat == 0):
                    continue
                
                # Check category range
                if not (min_cat <= cat <= max_cat):
                    continue
                
                # Normalize feature
                feat_norm = np.linalg.norm(feat)
                if feat_norm > 0:
                    feat = feat / feat_norm
                
                # Add to gallery
                item_id_str = str(int(item_id)) if isinstance(item_id, (int, float)) else str(item_id)
                gallery_by_category[int(cat)][item_id_str] = feat.astype(np.float32)
    
    # Convert gallery to array format for efficient computation
    for cat_id in gallery_by_category:
        items = gallery_by_category[cat_id]
        if items:
            gallery_by_category[cat_id] = {
                'ids': np.array(list(items.keys())),
                'features': np.array(list(items.values()))
            }
    
    # Print gallery statistics
    total_gallery_items = 0
    print(f"[INFO] Gallery statistics:")
    for cat_id in sorted(gallery_by_category.keys()):
        size = len(gallery_by_category[cat_id]['ids'])
        total_gallery_items += size
        cat_name = config['category_names'].get(cat_id, f"Cat{cat_id}")
        print(f"  Category {cat_id} ({cat_name}): {size:,} items")
    
    print(f"[INFO] ✅ Data collection complete:")
    print(f"  Test items: {len(test_items):,}")
    print(f"  Gallery items: {total_gallery_items:,}")
    
    return test_items, dict(gallery_by_category), dataset_type


# =============================================================================
# Model Evaluation with Corrected Metrics
# =============================================================================

def evaluate_model_comprehensive(model, test_items: List, gallery_by_category: Dict, 
                                dataset_type: str) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with corrected Top-K metrics
    
    Args:
        model: Trained model
        test_items: List of test items
        gallery_by_category: Gallery organized by category
        dataset_type: Type of dataset (IQON3000 or DeepFurniture)
        
    Returns:
        Dictionary containing evaluation results
    """
    
    print("[INFO] 🎯 Starting comprehensive evaluation...")
    
    config = get_dataset_config(dataset_type)
    min_cat, max_cat = config['category_range']
    
    # Collect queries by category
    queries_by_category = defaultdict(list)
    
    print("[INFO] Organizing queries by category...")
    for item in tqdm(test_items, desc="Processing test items"):
        target_features = item['target_features']
        target_categories = item['target_categories']
        target_ids = item['target_ids']
        
        # Process each target item as a potential query
        for j, (cat_id, target_id) in enumerate(zip(target_categories, target_ids)):
            # Skip padding items
            if cat_id == 0 or not (min_cat <= cat_id <= max_cat):
                continue
            
            target_id_str = str(int(target_id)) if isinstance(target_id, (int, float)) else str(target_id)
            
            # Check if target exists in gallery
            if (cat_id in gallery_by_category and 
                target_id_str in gallery_by_category[cat_id]['ids']):
                
                queries_by_category[int(cat_id)].append({
                    'query_features': item['query_features'],
                    'query_categories': item['query_categories'],
                    'target_id': target_id_str,
                    'category': int(cat_id)
                })
    
    # Print query statistics
    total_queries = sum(len(queries) for queries in queries_by_category.values())
    print(f"[INFO] Query distribution:")
    for cat_id in sorted(queries_by_category.keys()):
        count = len(queries_by_category[cat_id])
        cat_name = config['category_names'].get(cat_id, f"Cat{cat_id}")
        print(f"  Category {cat_id} ({cat_name}): {count:,} queries")
    
    if total_queries == 0:
        raise ValueError("No valid queries found for evaluation")
    
    # Evaluate each category
    category_results = {}
    all_ranks = []
    successful_predictions = 0
    failed_predictions = 0
    
    print("[INFO] Computing rankings...")
    
    for cat_id, queries in tqdm(queries_by_category.items(), desc="Evaluating categories"):
        if cat_id not in gallery_by_category or len(gallery_by_category[cat_id]['ids']) == 0:
            continue
        
        gallery = gallery_by_category[cat_id]
        gallery_size = len(gallery['ids'])
        category_ranks = []
        
        cat_name = config['category_names'].get(cat_id, f"Cat{cat_id}")
        print(f"[INFO] Processing Category {cat_id} ({cat_name}): {len(queries)} queries, {gallery_size} gallery items")
        
        for query in queries[:1000]:  # Limit to 1000 queries per category for memory
            try:
                # Prepare query input
                query_features = query['query_features']
                query_categories = query['query_categories']
                target_id = query['target_id']
                
                # Normalize query features
                query_features_norm = query_features / (np.linalg.norm(query_features, axis=-1, keepdims=True) + 1e-9)
                
                # Model inference
                query_input = {
                    'query_features': tf.constant(query_features_norm[None, :, :], dtype=tf.float32),
                    'query_categories': tf.constant(query_categories[None, :], dtype=tf.int32)
                }
                
                # Get predictions
                predictions = model(query_input, training=False)
                
                if predictions is None:
                    failed_predictions += 1
                    continue
                
                # Convert to numpy
                if hasattr(predictions, 'numpy'):
                    predictions = predictions.numpy()
                
                # Extract category-specific prediction
                if len(predictions.shape) == 3 and predictions.shape[1] >= cat_id:
                    pred_vector = predictions[0, cat_id - 1, :]  # 0-based indexing
                    
                    # Normalize prediction
                    pred_norm = np.linalg.norm(pred_vector)
                    if pred_norm > 0:
                        pred_vector = pred_vector / pred_norm
                    
                    # Compute similarities with gallery
                    gallery_features = gallery['features']
                    similarities = np.dot(gallery_features, pred_vector)
                    
                    # Get ranking
                    sorted_indices = np.argsort(similarities)[::-1]  # Descending order
                    
                    # Find target position
                    target_indices = np.where(gallery['ids'] == target_id)[0]
                    
                    if len(target_indices) > 0:
                        target_idx = target_indices[0]
                        rank_position = np.where(sorted_indices == target_idx)[0]
                        
                        if len(rank_position) > 0:
                            rank = rank_position[0] + 1  # 1-based rank
                            category_ranks.append(rank)
                            all_ranks.append(rank)
                            successful_predictions += 1
                        else:
                            failed_predictions += 1
                    else:
                        failed_predictions += 1
                else:
                    failed_predictions += 1
                    
            except Exception as e:
                failed_predictions += 1
                continue
        
        # Store category results
        if category_ranks:
            category_ranks = np.array(category_ranks)
            # ▼▼▼▼▼▼▼▼▼▼【ここから修正】▼▼▼▼▼▼▼▼▼▼
            # 'ranks': category_ranks, の行を削除。巨大な生配列はJSONに保存しない。
            category_results[cat_id] = {
                'count': len(category_ranks),
                'gallery_size': gallery_size,
                # 'ranks': category_ranks, # この行を削除またはコメントアウト
                'mrr': float(np.mean(1.0 / category_ranks)),
                'r_at_1': float(np.mean(category_ranks == 1)),
                'r_at_5': float(np.mean(category_ranks <= 5)),
                'r_at_10': float(np.mean(category_ranks <= 10)),
                'r_at_20': float(np.mean(category_ranks <= 20)),
                'mnr': float(np.mean(category_ranks)),
                'mdr': float(np.median(category_ranks)),
                'rsum': float(np.sum(category_ranks)),
                'avg_percentile': float(np.mean(category_ranks / gallery_size * 100))
            }
    
    print(f"[INFO] Evaluation complete:")
    print(f"  Successful predictions: {successful_predictions:,}")
    print(f"  Failed predictions: {failed_predictions:,}")
    
    # Compute overall results
    if not all_ranks:
        raise ValueError("No successful rankings computed")
    
    all_ranks = np.array(all_ranks)
    
    # ランクの合計を計算
    ranks_sum = float(np.sum(all_ranks))

    overall_results = {
        'total_queries': len(all_ranks),
        'successful_predictions': successful_predictions,
        'failed_predictions': failed_predictions,
        'mrr': float(np.mean(1.0 / all_ranks)),
        'r_at_1': float(np.mean(all_ranks == 1)),         # R@1
        'r_at_5': float(np.mean(all_ranks <= 5)),         # R@5
        'r_at_10': float(np.mean(all_ranks <= 10)),       # R@10
        'r_at_20': float(np.mean(all_ranks <= 20)),       # Top-20も念のため残す
        'mnr': float(np.mean(all_ranks)),                 # MnR (Mean Rank)
        'mdr': float(np.median(all_ranks)),               # MdR (Median Rank)
        'rsum': ranks_sum                                 # Rsum (Sum of Ranks)
    }
    
    # Create final results
    results = {
        'dataset': dataset_type,
        'overall': overall_results,
        'categories': category_results,
        'evaluation_info': {
            'total_test_items': len(test_items),
            'total_gallery_items': sum(len(gallery['ids']) for gallery in gallery_by_category.values()),
            'categories_evaluated': len(category_results)
        }
    }
    
    print("[INFO] ✅ Comprehensive evaluation completed")
    return results


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def evaluate_model(model, test_data, output_dir: str) -> Optional[Dict[str, Any]]:
    """
    Main evaluation pipeline
    
    Args:
        model: Trained model to evaluate
        test_data: Test dataset
        output_dir: Directory to save results
        
    Returns:
        Evaluation results dictionary or None if failed
    """
    
    print(f"\n[INFO] 🚀 Starting model evaluation pipeline...")
    print(f"[INFO] Output directory: {output_dir}")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Collect test data and build gallery
        test_items, gallery_by_category, dataset_type = collect_test_data(test_data)
        
        # Step 2: Run comprehensive evaluation
        results = evaluate_model_comprehensive(model, test_items, gallery_by_category, dataset_type)
        
        # Step 3: Save and display results
        save_evaluation_results(results, output_dir)
        display_evaluation_results(results)
        
        # Memory cleanup
        clear_memory()
        
        return results
        
    except Exception as e:
        print(f"[ERROR] ❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error log
        error_path = os.path.join(output_dir, 'evaluation_error.txt')
        with open(error_path, 'w') as f:
            f.write(f"Evaluation Error: {str(e)}\n\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())
        
        return None


# =============================================================================
# Results Display and Saving
# =============================================================================

def display_evaluation_results(results: Dict[str, Any]):
    """Display evaluation results in a formatted table"""
    
    if not results or 'overall' not in results:
        print("[ERROR] No results to display")
        return
    
    overall = results['overall']
    
    print(f"\n{'='*70}")
    print(f"🎯 EVALUATION RESULTS - {results['dataset']}")
    print(f"{'='*70}")
    print(f"Total Queries: {overall.get('total_queries', 0):,}")
    print(f"Successful Predictions: {overall.get('successful_predictions', 0):,}")
    print(f"-" * 70)
    # ▼▼▼▼▼▼▼▼▼▼【ここから修正】▼▼▼▼▼▼▼▼▼▼
    # 表示名をシンプルな形式に変更
    print(f"R@1 : {overall.get('r_at_1', 0):.2%}")
    print(f"R@5 : {overall.get('r_at_5', 0):.2%}")
    print(f"R@10: {overall.get('r_at_10', 0):.2%}")
    print(f"MnR : {overall.get('mnr', 0):.2f}")
    print(f"MdR : {overall.get('mdr', 0):.2f}")
    print(f"Rsum: {overall.get('rsum', 0):,.0f}")
    print(f"MRR : {overall.get('mrr', 0):.4f}")
    # ▲▲▲▲▲▲▲▲▲▲【ここまで修正】▲▲▲▲▲▲▲▲▲▲
    print(f"{'='*70}")
    
    # カテゴリ別結果のヘッダーも修正
    if 'categories' in results and results['categories']:
        print(f"\n📊 PER-CATEGORY RESULTS:")
        print(f"-" * 80)
        # ▼▼▼▼▼▼▼▼▼▼【ここから修正】▼▼▼▼▼▼▼▼▼▼
        print(f"{'Cat':<3} {'Name':<20} {'Queries':<8} {'MRR':<8} {'R@1':<7} {'R@5':<7} {'R@10':<8} {'MnR':<8} {'MdR':<8}")
        # ▲▲▲▲▲▲▲▲▲▲【ここまで修正】▲▲▲▲▲▲▲▲▲▲
        print(f"-" * 80)
        
        config = get_dataset_config(results['dataset'])
        
        for cat_id in sorted(results['categories'].keys()):
            cat_result = results['categories'][cat_id]
            cat_name = config['category_names'].get(cat_id, f"Cat{cat_id}")[:19]
            
            # ▼▼▼▼▼▼▼▼▼▼【ここから修正】▼▼▼▼▼▼▼▼▼▼
            print(f"{cat_id:<3} {cat_name:<20} {cat_result.get('count', 0):<8} "
                  f"{cat_result.get('mrr', 0):<8.4f} "
                  f"{cat_result.get('r_at_1', 0):<7.2%} "
                  f"{cat_result.get('r_at_5', 0):<7.2%} "
                  f"{cat_result.get('r_at_10', 0):<8.2%} "
                  f"{cat_result.get('mnr', 0):<8.1f} "
                  f"{cat_result.get('mdr', 0):<8.1f}")
            # ▲▲▲▲▲▲▲▲▲▲【ここまで修正】▲▲▲▲▲▲▲▲▲▲
        
        print(f"-" * 80)


def save_evaluation_results(results: Dict[str, Any], output_dir: str):
    """Save evaluation results to files"""
    
    print(f"[INFO] 💾 Saving evaluation results...")
    
    # Save JSON results
    json_path = os.path.join(output_dir, 'evaluation_results.json')
    # cls=NumpyJSONEncoder を追加してカスタムエンコーダを指定
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyJSONEncoder)
    print(f"[INFO] Results saved to: {json_path}")
    
    # Save CSV summary
    csv_path = os.path.join(output_dir, 'results_summary.csv')
    save_results_csv(results, csv_path)
    
    # Save detailed text report
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    save_text_report(results, report_path)
    
    print(f"[INFO] ✅ All results saved to: {output_dir}")


def save_results_csv(results: Dict[str, Any], csv_path: str):
    """Save results as CSV file"""
    
    if 'categories' not in results:
        return
    
    data = []
    config = get_dataset_config(results['dataset'])
    
    # Category results
    for cat_id in sorted(results['categories'].keys()):
        cat_result = results['categories'][cat_id]
        cat_name = config['category_names'].get(cat_id, f"Category_{cat_id}")
        
        # ▼▼▼▼▼▼▼▼▼▼【ここから修正】▼▼▼▼▼▼▼▼▼▼
        # CSVのヘッダーと使用するキーを新しい形式に統一
        data.append({
            'Category_ID': cat_id,
            'Category_Name': cat_name,
            'Query_Count': cat_result.get('count', 0),
            'Gallery_Size': cat_result.get('gallery_size', 0),
            'MRR': cat_result.get('mrr', 0),
            'R@1': cat_result.get('r_at_1', 0),
            'R@5': cat_result.get('r_at_5', 0),
            'R@10': cat_result.get('r_at_10', 0),
            'MnR': cat_result.get('mnr', 0),
            'MdR': cat_result.get('mdr', 0),
            'Rsum': cat_result.get('rsum', 0)
        })
        # ▲▲▲▲▲▲▲▲▲▲【ここまで修正】▲▲▲▲▲▲▲▲▲▲
    
    # Overall results
    overall = results['overall']
    # ▼▼▼▼▼▼▼▼▼▼【ここから修正】▼▼▼▼▼▼▼▼▼▼
    # CSVのヘッダーと使用するキーを新しい形式に統一
    data.append({
        'Category_ID': 'Overall',
        'Category_Name': 'All Categories',
        'Query_Count': overall.get('total_queries', 0),
        'Gallery_Size': sum(results['categories'][cat].get('gallery_size', 0) for cat in results['categories']),
        'MRR': overall.get('mrr', 0),
        'R@1': overall.get('r_at_1', 0),
        'R@5': overall.get('r_at_5', 0),
        'R@10': overall.get('r_at_10', 0),
        'MnR': overall.get('mnr', 0),
        'MdR': overall.get('mdr', 0),
        'Rsum': overall.get('rsum', 0)
    })
    # ▲▲▲▲▲▲▲▲▲▲【ここまで修正】▲▲▲▲▲▲▲▲▲▲
    
    # Create and save DataFrame
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] CSV summary saved to: {csv_path}")


def save_text_report(results: Dict[str, Any], report_path: str):
    """Save detailed text report"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("SET RETRIEVAL EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Dataset: {results.get('dataset', 'N/A')}\n")
        f.write(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall results
        overall = results.get('overall', {})
        f.write("OVERALL RESULTS\n")
        f.write("-" * 20 + "\n")
        total_queries = overall.get('total_queries', 0)
        successful_predictions = overall.get('successful_predictions', 0)
        failed_predictions = overall.get('failed_predictions', 0)
        
        f.write(f"Total Queries: {total_queries:,}\n")
        f.write(f"Successful Predictions: {successful_predictions:,}\n")
        if (total_queries + failed_predictions) > 0:
             success_rate = successful_predictions / (total_queries + failed_predictions)
             f.write(f"Success Rate: {success_rate:.1%}\n\n")
        
        f.write("METRICS\n")
        f.write("-" * 10 + "\n")
        # ▼▼▼▼▼▼▼▼▼▼【ここから修正】▼▼▼▼▼▼▼▼▼▼
        # 新しいキー名と表示形式に統一
        f.write(f"R@1 : {overall.get('r_at_1', 0):.2%}\n")
        f.write(f"R@5 : {overall.get('r_at_5', 0):.2%}\n")
        f.write(f"R@10: {overall.get('r_at_10', 0):.2%}\n")
        f.write(f"MnR : {overall.get('mnr', 0):.2f}\n")
        f.write(f"MdR : {overall.get('mdr', 0):.2f}\n")
        f.write(f"Rsum: {overall.get('rsum', 0):,.0f}\n")
        f.write(f"MRR : {overall.get('mrr', 0):.4f}\n\n")
        # ▲▲▲▲▲▲▲▲▲▲【ここまで修正】▲▲▲▲▲▲▲▲▲▲
        
        # Category results
        if 'categories' in results:
            f.write("CATEGORY-WISE RESULTS\n")
            f.write("-" * 25 + "\n")
            
            config = get_dataset_config(results.get('dataset', 'N/A'))
            
            for cat_id in sorted(results['categories'].keys()):
                cat_result = results['categories'][cat_id]
                cat_name = config['category_names'].get(cat_id, f"Category_{cat_id}")
                
                f.write(f"\nCategory {cat_id}: {cat_name}\n")
                f.write(f"  Queries: {cat_result.get('count', 0):,}\n")
                f.write(f"  Gallery Size: {cat_result.get('gallery_size', 0):,}\n")
                # ▼▼▼▼▼▼▼▼▼▼【ここから修正】▼▼▼▼▼▼▼▼▼▼
                f.write(f"  MRR : {cat_result.get('mrr', 0):.4f}\n")
                f.write(f"  R@1 : {cat_result.get('r_at_1', 0):.2%}\n")
                f.write(f"  R@5 : {cat_result.get('r_at_5', 0):.2%}\n")
                f.write(f"  R@10: {cat_result.get('r_at_10', 0):.2%}\n")
                f.write(f"  MnR : {cat_result.get('mnr', 0):.2f}\n")
                f.write(f"  MdR : {cat_result.get('mdr', 0):.2f}\n")
                f.write(f"  Rsum: {cat_result.get('rsum', 0):,.0f}\n")
                # ▲▲▲▲▲▲▲▲▲▲【ここまで修正】▲▲▲▲▲▲▲▲▲▲
        
        # Evaluation info
        if 'evaluation_info' in results:
            eval_info = results['evaluation_info']
            f.write(f"\nEVALUATION INFO\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Test Items: {eval_info.get('total_test_items', 0):,}\n")
            f.write(f"Total Gallery Items: {eval_info.get('total_gallery_items', 0):,}\n")
            f.write(f"Categories Evaluated: {eval_info.get('categories_evaluated', 0)}\n")
    
    print(f"[INFO] Detailed report saved to: {report_path}")

# =============================================================================
# Utility Functions
# =============================================================================

def save_results(results: Dict[str, Any], output_dir: str):
    """Main function to save all results"""
    save_evaluation_results(results, output_dir)


def normalize_features(features):
    """L2 normalize features"""
    if isinstance(features, tf.Tensor):
        norm = tf.norm(features, axis=-1, keepdims=True)
        norm = tf.where(norm == 0, 1e-9, norm)
        return features / norm
    else:
        norm = np.linalg.norm(features, axis=-1, keepdims=True)
        norm[norm == 0] = 1e-9
        return features / norm


def compute_similarity_matrix(features1, features2):
    """Compute similarity matrix between two feature sets"""
    # Normalize features
    features1_norm = normalize_features(features1)
    features2_norm = normalize_features(features2)
    
    # Compute cosine similarity
    if isinstance(features1, tf.Tensor):
        similarities = tf.matmul(features1_norm, features2_norm, transpose_b=True)
    else:
        similarities = np.dot(features1_norm, features2_norm.T)
    
    return similarities


def get_top_k_indices(similarities, k: int):
    """Get top-k indices from similarity matrix"""
    if isinstance(similarities, tf.Tensor):
        _, top_k_indices = tf.nn.top_k(similarities, k=k)
        return top_k_indices.numpy()
    else:
        return np.argsort(similarities, axis=-1)[:, -k:][:, ::-1]


# =============================================================================
# Performance Analysis
# =============================================================================

def analyze_performance_by_set_size(results: Dict[str, Any], test_items: List) -> Dict[str, Any]:
    """Analyze performance by set size"""
    
    print("[INFO] Analyzing performance by set size...")
    
    # Group results by set size
    size_groups = defaultdict(list)
    
    for item in test_items:
        query_size = np.sum(item['query_categories'] > 0)
        target_size = np.sum(item['target_categories'] > 0)
        avg_size = (query_size + target_size) / 2
        
        # Group by size ranges
        if avg_size <= 3:
            size_groups['Small (≤3)'].append(item)
        elif avg_size <= 6:
            size_groups['Medium (4-6)'].append(item)
        else:
            size_groups['Large (≥7)'].append(item)
    
    # Compute metrics for each group
    size_analysis = {}
    for size_name, items in size_groups.items():
        size_analysis[size_name] = {
            'count': len(items),
            'avg_query_size': np.mean([np.sum(item['query_categories'] > 0) for item in items]),
            'avg_target_size': np.mean([np.sum(item['target_categories'] > 0) for item in items])
        }
    
    return size_analysis


def create_performance_visualization(results: Dict[str, Any], output_dir: str):
    """Create performance visualization charts"""
    
    try:
        import matplotlib.pyplot as plt
        
        # Create category performance chart
        if 'categories' in results:
            categories = results['categories']
            cat_ids = sorted(categories.keys())
            
            metrics = ['top1', 'top5', 'top10', 'mrr']
            metric_names = ['Top-1', 'Top-5', 'Top-10', 'MRR']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, (metric, name) in enumerate(zip(metrics, metric_names)):
                values = [categories[cat_id][metric] for cat_id in cat_ids]
                axes[i].bar(cat_ids, values)
                axes[i].set_title(f'{name} by Category')
                axes[i].set_xlabel('Category ID')
                axes[i].set_ylabel(name)
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            viz_path = os.path.join(output_dir, 'performance_by_category.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[INFO] Performance visualization saved: {viz_path}")
            
    except ImportError:
        print("[WARN] Matplotlib not available, skipping visualization")


# =============================================================================
# Testing and Validation
# =============================================================================

def validate_evaluation_pipeline(sample_size: int = 100):
    """Validate evaluation pipeline with synthetic data"""
    
    print(f"[INFO] 🧪 Validating evaluation pipeline with {sample_size} samples...")
    
    # Create synthetic model and data
    from models import create_model
    
    model_config = {
        'feature_dim': 512,
        'num_heads': 2,
        'num_layers': 1,
        'num_categories': 7,
        'hidden_dim': 512
    }
    
    model = create_model(model_config)
    
    # Create synthetic test data
    test_items = []
    for i in range(sample_size):
        test_items.append({
            'query_features': np.random.randn(10, 512).astype(np.float32),
            'query_categories': np.random.randint(1, 8, (10,)).astype(np.int32),
            'target_features': np.random.randn(10, 512).astype(np.float32),
            'target_categories': np.random.randint(1, 8, (10,)).astype(np.int32),
            'query_ids': np.arange(10).astype(str),
            'target_ids': np.arange(10, 20).astype(str)
        })
    
    # Create synthetic gallery
    gallery_by_category = {}
    for cat_id in range(1, 8):
        gallery_by_category[cat_id] = {
            'ids': np.array([f"item_{i}" for i in range(100)]),
            'features': np.random.randn(100, 512).astype(np.float32)
        }
    
    # Run evaluation
    try:
        results = evaluate_model_comprehensive(model, test_items, gallery_by_category, 'IQON3000')
        
        if results and 'overall' in results:
            print("[INFO] ✅ Evaluation pipeline validation passed")
            print(f"[INFO] Sample results - MRR: {results['overall']['mrr']:.4f}")
            return True
        else:
            print("[ERROR] ❌ Evaluation pipeline validation failed - no results")
            return False
            
    except Exception as e:
        print(f"[ERROR] ❌ Evaluation pipeline validation failed: {e}")
        return False


if __name__ == "__main__":
    # Test the evaluation pipeline
    print("Set Retrieval Utils - WACV 2026")
    print("=" * 40)
    
    # Setup GPU
    setup_gpu_memory()
    
    # Validate pipeline
    validate_evaluation_pipeline()
    
    print("Utils module ready for use!")