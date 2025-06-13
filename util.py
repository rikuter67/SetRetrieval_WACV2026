# # """
# # utils.py - Evaluation and Utility Functions for Set Retrieval (WACV 2026)
# # ===========================================================================
# # Comprehensive evaluation pipeline with corrected metrics, GPU management,
# # and result visualization for heterogeneous set retrieval.
# # """

# # import os
# # import json
# # import pickle
# # import gzip
# # import time
# # from typing import Dict, List, Tuple, Optional, Any
# # from collections import defaultdict
# # import gc

# # import json
# # import numpy as np
# # import pandas as pd
# # import tensorflow as tf

# # try:
# #     from tqdm import tqdm
# # except ImportError:
# #     def tqdm(iterator, *args, **kwargs):
# #         return iterator

     
# # # =============================================================================
# # # GPU and Memory Management
# # # =============================================================================

# # def setup_gpu_memory():
# #     """Configure GPU memory growth and optimization"""
    
# #     print("[INFO] Configuring GPU memory...")
    
# #     gpus = tf.config.experimental.list_physical_devices('GPU')
    
# #     if gpus:
# #         try:
# #             # Enable memory growth for all GPUs
# #             for gpu in gpus:
# #                 tf.config.experimental.set_memory_growth(gpu, True)
            
# #             # Set visible devices based on CUDA_VISIBLE_DEVICES
# #             cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
# #             if cuda_visible:
# #                 print(f"[INFO] Using GPU devices: {cuda_visible}")
            
# #             print(f"[INFO] ✅ GPU memory configured for {len(gpus)} GPU(s)")
            
# #             # Print GPU information
# #             for i, gpu in enumerate(gpus):
# #                 print(f"[INFO] GPU {i}: {gpu.name}")
                
# #         except RuntimeError as e:
# #             print(f"[WARN] GPU configuration warning: {e}")
# #     else:
# #         print("[INFO] No GPUs found, using CPU")


# # def clear_memory():
# #     """Clear memory and run garbage collection"""
    
# #     gc.collect()
    
# #     if tf.config.list_physical_devices('GPU'):
# #         try:
# #             tf.keras.backend.clear_session()
# #         except:
# #             pass


# # def monitor_memory_usage():
# #     """Monitor and print memory usage"""
    
# #     try:
# #         import psutil
# #         memory_percent = psutil.virtual_memory().percent
# #         print(f"[INFO] System memory usage: {memory_percent:.1f}%")
        
# #         # GPU memory if available
# #         if tf.config.list_physical_devices('GPU'):
# #             try:
# #                 from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
# #                 nvmlInit()
# #                 handle = nvmlDeviceGetHandleByIndex(0)
# #                 mem_info = nvmlDeviceGetMemoryInfo(handle)
# #                 gpu_percent = (mem_info.used / mem_info.total) * 100
# #                 print(f"[INFO] GPU memory usage: {gpu_percent:.1f}%")
# #             except:
# #                 pass
                
# #     except ImportError:
# #         pass




# # def get_dataset_config(dataset_type: str) -> Dict[str, Any]:
# #     """Get configuration for dataset type"""
    
# #     if dataset_type not in DATASET_CONFIGS:
# #         print(f"[WARN] Unknown dataset type: {dataset_type}, using IQON3000")
# #         dataset_type = 'IQON3000'
    
# #     return DATASET_CONFIGS[dataset_type].copy()

# # class NumpyJSONEncoder(json.JSONEncoder):
# #     """
# #     Custom JSON encoder for NumPy types.
# #     Converts np.integer, np.floating, and np.ndarray to native Python types.
# #     """
# #     def default(self, obj):
# #         if isinstance(obj, np.integer):
# #             return int(obj)
# #         elif isinstance(obj, np.floating):
# #             return float(obj)
# #         elif isinstance(obj, np.ndarray):
# #             return obj.tolist()
# #         return super(NumpyJSONEncoder, self).default(obj)


# # # =============================================================================
# # # Data Collection and Gallery Building
# # # =============================================================================

# # def collect_test_data(test_data, max_batches: Optional[int] = None) -> Tuple[List, Dict, str]:
# #     """
# #     Collect test data and build gallery for evaluation
    
# #     Args:
# #         test_data: TensorFlow dataset
# #         max_batches: Maximum number of batches to process (None for all)
        
# #     Returns:
# #         Tuple of (test_items, gallery_by_category, dataset_type)
# #     """
    
# #     print("[INFO] 📥 Collecting test data and building gallery...")
    
# #     # Detect dataset type
# #     dataset_type = detect_dataset_type(test_data)
# #     config = get_dataset_config(dataset_type)
# #     min_cat, max_cat = config['category_range']
    
# #     print(f"[INFO] Dataset detected: {dataset_type}")
# #     print(f"[INFO] Categories: {min_cat}-{max_cat}")
    
# #     # Collect all batches
# #     all_batches = []
# #     batch_count = 0

# #     for batch in tqdm(test_data, desc="Collecting batches"):
# #         all_batches.append(batch)
# #         batch_count += 1
# #         if max_batches and batch_count >= max_batches:
# #             break
# #         if batch_count % 50 == 0:
# #             clear_memory()
    
# #     print(f"[INFO] Collected {len(all_batches)} batches")
# #     if not all_batches:
# #         raise ValueError("No test data collected")
    
# #     test_items = []
# #     gallery_by_category = defaultdict(dict)
    
# #     print("[INFO] Processing batches...")

    
# #     for batch in tqdm(test_data, desc="Collecting batches"):
# #         query_features = batch['query_features'].numpy()
# #         query_categories = batch['query_categories'].numpy()
# #         target_features = batch['target_features'].numpy()
# #         target_categories = batch['target_categories'].numpy()
        
# #         # item_idの取得を安全に行う
# #         query_ids_val = batch.get('query_item_ids')
# #         target_ids_val = batch.get('target_item_ids')

# #         query_ids = query_ids_val.numpy() if query_ids_val is not None else np.arange(len(query_features))
# #         target_ids = target_ids_val.numpy() if target_ids_val is not None else np.arange(len(target_features))

# #         batch_size = query_features.shape[0]
        
# #         # Create test items
# #         for i in range(batch_size):
# #             test_items.append({
# #                 'query_features': query_features[i],
# #                 'query_categories': query_categories[i],
# #                 'target_features': target_features[i],
# #                 'target_categories': target_categories[i],
# #                 'query_ids': query_ids[i] if len(query_ids.shape) > 1 else [query_ids[i]],
# #                 'target_ids': target_ids[i] if len(target_ids.shape) > 1 else [target_ids[i]]
# #             })
            
# #             # Build gallery from target items
# #             for j, (feat, cat, item_id) in enumerate(zip(
# #                 target_features[i], target_categories[i], 
# #                 target_ids[i] if len(target_ids.shape) > 1 else [target_ids[i]]
# #             )):
# #                 # Skip padding items
# #                 if cat == 0 or np.all(feat == 0):
# #                     continue
                
# #                 # Check category range
# #                 if not (min_cat <= cat <= max_cat):
# #                     continue
                
# #                 # Normalize feature
# #                 feat_norm = np.linalg.norm(feat)
# #                 if feat_norm > 0:
# #                     feat = feat / feat_norm
                
# #                 # Add to gallery
# #                 item_id_str = str(int(item_id)) if isinstance(item_id, (int, float)) else str(item_id)
# #                 gallery_by_category[int(cat)][item_id_str] = feat.astype(np.float32)
    
# #     # Convert gallery to array format for efficient computation
# #     for cat_id in gallery_by_category:
# #         items = gallery_by_category[cat_id]
# #         if items:
# #             gallery_by_category[cat_id] = {
# #                 'ids': np.array(list(items.keys())),
# #                 'features': np.array(list(items.values()))
# #             }
    
# #     # Print gallery statistics
# #     total_gallery_items = 0
# #     print(f"[INFO] Gallery statistics:")
# #     for cat_id in sorted(gallery_by_category.keys()):
# #         size = len(gallery_by_category[cat_id]['ids'])
# #         total_gallery_items += size
# #         cat_name = config['category_names'].get(cat_id, f"Cat{cat_id}")
# #         print(f"  Category {cat_id} ({cat_name}): {size:,} items")
    
# #     print(f"[INFO] ✅ Data collection complete:")
# #     print(f"  Test items: {len(test_items):,}")
# #     print(f"  Gallery items: {total_gallery_items:,}")
    
# #     return test_items, dict(gallery_by_category), dataset_type


# # # =============================================================================
# # # Model Evaluation with Corrected Metrics
# # # =============================================================================

# # def evaluate_model_comprehensive(model, test_items: List, gallery_by_category: Dict, 
# #                                 dataset_type: str) -> Dict[str, Any]:
# #     """
# #     Comprehensive model evaluation with corrected Top-K metrics
    
# #     Args:
# #         model: Trained model
# #         test_items: List of test items
# #         gallery_by_category: Gallery organized by category
# #         dataset_type: Type of dataset (IQON3000 or DeepFurniture)
        
# #     Returns:
# #         Dictionary containing evaluation results
# #     """
    
# #     print("[INFO] 🎯 Starting comprehensive evaluation...")
    
# #     config = get_dataset_config(dataset_type)
# #     min_cat, max_cat = config['category_range']
    
# #     # Collect queries by category
# #     queries_by_category = defaultdict(list)
    
# #     print("[INFO] Organizing queries by category...")
# #     for item in tqdm(test_items, desc="Processing test items"):
# #         target_features = item['target_features']
# #         target_categories = item['target_categories']
# #         target_ids = item['target_ids']
        
# #         # Process each target item as a potential query
# #         for j, (cat_id, target_id) in enumerate(zip(target_categories, target_ids)):
# #             # Skip padding items
# #             if cat_id == 0 or not (min_cat <= cat_id <= max_cat):
# #                 continue
            
# #             target_id_str = str(int(target_id)) if isinstance(target_id, (int, float)) else str(target_id)
            
# #             # Check if target exists in gallery
# #             if (cat_id in gallery_by_category and 
# #                 target_id_str in gallery_by_category[cat_id]['ids']):
                
# #                 queries_by_category[int(cat_id)].append({
# #                     'query_features': item['query_features'],
# #                     'query_categories': item['query_categories'],
# #                     'target_id': target_id_str,
# #                     'category': int(cat_id)
# #                 })
    
# #     # Print query statistics
# #     total_queries = sum(len(queries) for queries in queries_by_category.values())
# #     print(f"[INFO] Query distribution:")
# #     for cat_id in sorted(queries_by_category.keys()):
# #         count = len(queries_by_category[cat_id])
# #         cat_name = config['category_names'].get(cat_id, f"Cat{cat_id}")
# #         print(f"  Category {cat_id} ({cat_name}): {count:,} queries")
    
# #     if total_queries == 0:
# #         raise ValueError("No valid queries found for evaluation")
    
# #     # Evaluate each category
# #     category_results = {}
# #     all_ranks = []
# #     successful_predictions = 0
# #     failed_predictions = 0
    
# #     print("[INFO] Computing rankings...")
    
# #     for cat_id, queries in tqdm(queries_by_category.items(), desc="Evaluating categories"):
# #         if cat_id not in gallery_by_category or len(gallery_by_category[cat_id]['ids']) == 0:
# #             continue
        
# #         gallery = gallery_by_category[cat_id]
# #         gallery_size = len(gallery['ids'])
# #         category_ranks = []
        
# #         cat_name = config['category_names'].get(cat_id, f"Cat{cat_id}")
# #         print(f"[INFO] Processing Category {cat_id} ({cat_name}): {len(queries)} queries, {gallery_size} gallery items")
        
# #         for query in queries[:1000]:  # Limit to 1000 queries per category for memory
# #             try:
# #                 # Prepare query input
# #                 query_features = query['query_features']
# #                 query_categories = query['query_categories']
# #                 target_id = query['target_id']
                
# #                 # Normalize query features
# #                 query_features_norm = query_features / (np.linalg.norm(query_features, axis=-1, keepdims=True) + 1e-9)
                
# #                 # Model inference
# #                 query_input = {
# #                     'query_features': tf.constant(query_features_norm[None, :, :], dtype=tf.float32),
# #                     'query_categories': tf.constant(query_categories[None, :], dtype=tf.int32)
# #                 }
                
# #                 # Get predictions
# #                 predictions = model(query_input, training=False)
                
# #                 if predictions is None:
# #                     failed_predictions += 1
# #                     continue
                
# #                 # Convert to numpy
# #                 if hasattr(predictions, 'numpy'):
# #                     predictions = predictions.numpy()
                
# #                 # Extract category-specific prediction
# #                 if len(predictions.shape) == 3 and predictions.shape[1] >= cat_id:
# #                     pred_vector = predictions[0, cat_id - 1, :]  # 0-based indexing
                    
# #                     # Normalize prediction
# #                     pred_norm = np.linalg.norm(pred_vector)
# #                     if pred_norm > 0:
# #                         pred_vector = pred_vector / pred_norm
                    
# #                     # Compute similarities with gallery
# #                     gallery_features = gallery['features']
# #                     similarities = np.dot(gallery_features, pred_vector)
                    
# #                     # Get ranking
# #                     sorted_indices = np.argsort(similarities)[::-1]  # Descending order
                    
# #                     # Find target position
# #                     target_indices = np.where(gallery['ids'] == target_id)[0]
                    
# #                     if len(target_indices) > 0:
# #                         target_idx = target_indices[0]
# #                         rank_position = np.where(sorted_indices == target_idx)[0]
                        
# #                         if len(rank_position) > 0:
# #                             rank = rank_position[0] + 1  # 1-based rank
# #                             category_ranks.append(rank)
# #                             all_ranks.append(rank)
# #                             successful_predictions += 1
# #                         else:
# #                             failed_predictions += 1
# #                     else:
# #                         failed_predictions += 1
# #                 else:
# #                     failed_predictions += 1
                    
# #             except Exception as e:
# #                 failed_predictions += 1
# #                 continue
        
# #         # Store category results
# #         if category_ranks:
# #             category_ranks = np.array(category_ranks)
# #             # ▼▼▼▼▼▼▼▼▼▼【ここから修正】▼▼▼▼▼▼▼▼▼▼
# #             # 'ranks': category_ranks, の行を削除。巨大な生配列はJSONに保存しない。
# #             category_results[cat_id] = {
# #                 'count': len(category_ranks),
# #                 'gallery_size': gallery_size,
# #                 # 'ranks': category_ranks, # この行を削除またはコメントアウト
# #                 'mrr': float(np.mean(1.0 / category_ranks)),
# #                 'r_at_1': float(np.mean(category_ranks == 1)),
# #                 'r_at_5': float(np.mean(category_ranks <= 5)),
# #                 'r_at_10': float(np.mean(category_ranks <= 10)),
# #                 'r_at_20': float(np.mean(category_ranks <= 20)),
# #                 'mnr': float(np.mean(category_ranks)),
# #                 'mdr': float(np.median(category_ranks)),
# #                 'rsum': float(np.sum(category_ranks)),
# #                 'avg_percentile': float(np.mean(category_ranks / gallery_size * 100))
# #             }
    
# #     print(f"[INFO] Evaluation complete:")
# #     print(f"  Successful predictions: {successful_predictions:,}")
# #     print(f"  Failed predictions: {failed_predictions:,}")
    
# #     # Compute overall results
# #     if not all_ranks:
# #         raise ValueError("No successful rankings computed")
    
# #     all_ranks = np.array(all_ranks)
    
# #     # ランクの合計を計算
# #     ranks_sum = float(np.sum(all_ranks))

# #     overall_results = {
# #         'total_queries': len(all_ranks),
# #         'successful_predictions': successful_predictions,
# #         'failed_predictions': failed_predictions,
# #         'mrr': float(np.mean(1.0 / all_ranks)),
# #         'r_at_1': float(np.mean(all_ranks == 1)),         # R@1
# #         'r_at_5': float(np.mean(all_ranks <= 5)),         # R@5
# #         'r_at_10': float(np.mean(all_ranks <= 10)),       # R@10
# #         'r_at_20': float(np.mean(all_ranks <= 20)),       # Top-20も念のため残す
# #         'mnr': float(np.mean(all_ranks)),                 # MnR (Mean Rank)
# #         'mdr': float(np.median(all_ranks)),               # MdR (Median Rank)
# #         'rsum': ranks_sum                                 # Rsum (Sum of Ranks)
# #     }
    
# #     # Create final results
# #     results = {
# #         'dataset': dataset_type,
# #         'overall': overall_results,
# #         'categories': category_results,
# #         'evaluation_info': {
# #             'total_test_items': len(test_items),
# #             'total_gallery_items': sum(len(gallery['ids']) for gallery in gallery_by_category.values()),
# #             'categories_evaluated': len(category_results)
# #         }
# #     }
    
# #     print("[INFO] ✅ Comprehensive evaluation completed")
# #     return results


# # # =============================================================================
# # # Main Evaluation Pipeline
# # # =============================================================================

# # def evaluate_model(model, test_data, output_dir: str) -> Optional[Dict[str, Any]]:
# #     """
# #     Main evaluation pipeline
    
# #     Args:
# #         model: Trained model to evaluate
# #         test_data: Test dataset
# #         output_dir: Directory to save results
        
# #     Returns:
# #         Evaluation results dictionary or None if failed
# #     """
    
# #     print(f"\n[INFO] 🚀 Starting model evaluation pipeline...")
# #     print(f"[INFO] Output directory: {output_dir}")
    
# #     try:
# #         os.makedirs(output_dir, exist_ok=True)
        
# #         # Step 1: Collect test data and build gallery
# #         test_items, gallery_by_category, dataset_type = collect_test_data(test_data)
        
# #         # Step 2: Run comprehensive evaluation
# #         results = evaluate_model_comprehensive(model, test_items, gallery_by_category, dataset_type)
        
# #         # Step 3: Save and display results
# #         save_evaluation_results(results, output_dir)
# #         display_evaluation_results(results)
        
# #         # Memory cleanup
# #         clear_memory()
        
# #         return results
        
# #     except Exception as e:
# #         print(f"[ERROR] ❌ Evaluation failed: {e}")
# #         import traceback
# #         traceback.print_exc()
        
# #         # Save error log
# #         error_path = os.path.join(output_dir, 'evaluation_error.txt')
# #         with open(error_path, 'w') as f:
# #             f.write(f"Evaluation Error: {str(e)}\n\n")
# #             f.write("Traceback:\n")
# #             f.write(traceback.format_exc())
        
# #         return None


# # # =============================================================================
# # # Utility Functions
# # # =============================================================================

# # def normalize_features(features):
# #     """L2 normalize features"""
# #     if isinstance(features, tf.Tensor):
# #         norm = tf.norm(features, axis=-1, keepdims=True)
# #         norm = tf.where(norm == 0, 1e-9, norm)
# #         return features / norm
# #     else:
# #         norm = np.linalg.norm(features, axis=-1, keepdims=True)
# #         norm[norm == 0] = 1e-9
# #         return features / norm


# # def compute_similarity_matrix(features1, features2):
# #     """Compute similarity matrix between two feature sets"""
# #     # Normalize features
# #     features1_norm = normalize_features(features1)
# #     features2_norm = normalize_features(features2)
    
# #     # Compute cosine similarity
# #     if isinstance(features1, tf.Tensor):
# #         similarities = tf.matmul(features1_norm, features2_norm, transpose_b=True)
# #     else:
# #         similarities = np.dot(features1_norm, features2_norm.T)
    
# #     return similarities


# # def get_top_k_indices(similarities, k: int):
# #     """Get top-k indices from similarity matrix"""
# #     if isinstance(similarities, tf.Tensor):
# #         _, top_k_indices = tf.nn.top_k(similarities, k=k)
# #         return top_k_indices.numpy()
# #     else:
# #         return np.argsort(similarities, axis=-1)[:, -k:][:, ::-1]


# # # =============================================================================
# # # Performance Analysis
# # # =============================================================================




# # # =============================================================================
# # # Testing and Validation
# # # =============================================================================

# # # def validate_evaluation_pipeline(sample_size: int = 100):
# # #     """Validate evaluation pipeline with synthetic data"""
    
# # #     print(f"[INFO] 🧪 Validating evaluation pipeline with {sample_size} samples...")
    
# # #     # Create synthetic model and data
# # #     from models import create_model
    
# # #     model_config = {
# # #         'feature_dim': 512,
# # #         'num_heads': 2,
# # #         'num_layers': 1,
# # #         'num_categories': 7,
# # #         'hidden_dim': 512
# # #     }
    
# # #     model = create_model(model_config)
    
# # #     # Create synthetic test data
# # #     test_items = []
# # #     for i in range(sample_size):
# # #         test_items.append({
# # #             'query_features': np.random.randn(10, 512).astype(np.float32),
# # #             'query_categories': np.random.randint(1, 8, (10,)).astype(np.int32),
# # #             'target_features': np.random.randn(10, 512).astype(np.float32),
# # #             'target_categories': np.random.randint(1, 8, (10,)).astype(np.int32),
# # #             'query_ids': np.arange(10).astype(str),
# # #             'target_ids': np.arange(10, 20).astype(str)
# # #         })
    
# # #     # Create synthetic gallery
# # #     gallery_by_category = {}
# # #     for cat_id in range(1, 8):
# # #         gallery_by_category[cat_id] = {
# # #             'ids': np.array([f"item_{i}" for i in range(100)]),
# # #             'features': np.random.randn(100, 512).astype(np.float32)
# # #         }
    
# # #     # Run evaluation
# # #     try:
# # #         results = evaluate_model_comprehensive(model, test_items, gallery_by_category, 'IQON3000')
        
# # #         if results and 'overall' in results:
# # #             print("[INFO] ✅ Evaluation pipeline validation passed")
# # #             print(f"[INFO] Sample results - MRR: {results['overall']['mrr']:.4f}")
# # #             return True
# # #         else:
# # #             print("[ERROR] ❌ Evaluation pipeline validation failed - no results")
# # #             return False
            
# # #     except Exception as e:
# # #         print(f"[ERROR] ❌ Evaluation pipeline validation failed: {e}")
# # #         return False


# # # if __name__ == "__main__":
# # #     # Test the evaluation pipeline
# # #     print("Set Retrieval Utils - WACV 2026")
# # #     print("=" * 40)
    
# # #     # Setup GPU
# # #     setup_gpu_memory()
    
# # #     # Validate pipeline
# # #     validate_evaluation_pipeline()
    
# # #     print("Utils module ready for use!")
# """
# plot.py - Visualization functions for the Set Retrieval project.
# """
# import os
# import numpy as np
# import pandas as pd
# from typing import Dict, Any, List, Optional
# from collections import defaultdict

# from config import get_dataset_config

# try:
#     import matplotlib.pyplot as plt
#     import matplotlib
#     matplotlib.use('Agg')  # Use non-interactive backend for servers
#     HAS_MATPLOTLIB = True
# except ImportError:
#     HAS_MATPLOTLIB = False

# def create_performance_visualization(results: Dict[str, Any], config: Dict[str, Any], output_dir: str):
#     """Create performance visualization charts and save to output directory."""
#     if not HAS_MATPLOTLIB:
#         print("[WARN] Matplotlib not available, skipping performance visualization.")
#         return

#     print("[INFO] 🎨 Creating performance visualizations...")
#     try:
#         if 'categories' not in results or not results['categories']:
#             print("[WARN] No category data to visualize.")
#             return

#         categories = results['categories']
#         dataset_name = results.get('dataset', 'Unknown Dataset')
        
#         cat_ids = sorted([int(k) for k in categories.keys()])
#         category_names = config.get('category_names', {})
#         cat_labels = [category_names.get(cid, f"Cat {cid}") for cid in cat_ids]
        
#         df_data = {
#             'Top1 Acc (%)': [categories[str(cid)].get('r_at_1', 0) * 100 for cid in cat_ids],
#             'Top10 Acc (%)': [categories[str(cid)].get('r_at_10', 0) * 100 for cid in cat_ids],
#             'Mean Rank (MnR)': [categories[str(cid)].get('mnr', 0) for cid in cat_ids],
#             'MRR': [categories[str(cid)].get('mrr', 0) for cid in cat_ids]
#         }
#         df = pd.DataFrame(df_data, index=cat_labels)

#         fig, axes = plt.subplots(2, 2, figsize=(18, 14))
#         fig.suptitle(f'Performance by Category - {dataset_name}', fontsize=20, y=1.0)
        
#         df['Top1 Acc (%)'].plot(kind='bar', ax=axes[0, 0], color='skyblue', rot=45, ha='right')
#         axes[0, 0].set_title('Top-1% Accuracy', fontsize=14)
#         axes[0, 0].set_ylabel('Accuracy (%)')

#         df['Top10 Acc (%)'].plot(kind='bar', ax=axes[0, 1], color='salmon', rot=45, ha='right')
#         axes[0, 1].set_title('Top-10% Accuracy', fontsize=14)
#         axes[0, 1].set_ylabel('Accuracy (%)')

#         df['Mean Rank (MnR)'].plot(kind='bar', ax=axes[1, 0], color='lightgreen', rot=45, ha='right')
#         axes[1, 0].set_title('Mean Rank (Lower is Better)', fontsize=14)
#         axes[1, 0].set_ylabel('Average Rank')

#         df['MRR'].plot(kind='bar', ax=axes[1, 1], color='plum', rot=45, ha='right')
#         axes[1, 1].set_title('Mean Reciprocal Rank (Higher is Better)', fontsize=14)
#         axes[1, 1].set_ylabel('MRR Score')

#         for ax in axes.flatten():
#             ax.grid(axis='y', linestyle='--', alpha=0.7)

#         plt.tight_layout(rect=[0, 0, 1, 0.97])
#         viz_path = os.path.join(output_dir, 'performance_by_category.png')
#         plt.savefig(viz_path, dpi=300, bbox_inches='tight')
#         plt.close(fig)
        
#         print(f"[INFO] ✅ Performance visualization saved to: {viz_path}")

#     except Exception as e:
#         print(f"[ERROR] ❌ Failed to create performance visualization: {e}")


"""
util.py - Core evaluation logic for Set Retrieval.
"""
import os
import gc
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
import tensorflow as tf
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, *args, **kwargs): return iterator

from config import get_dataset_config
from results import save_evaluation_results, display_evaluation_results
from plot import generate_all_visualizations

def setup_gpu_memory():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            print("🚀 GPU setup completed")
        except RuntimeError as e: print(f"GPU setup error: {e}")

def clear_memory():
    gc.collect()
    if tf.config.list_physical_devices('GPU'):
        try: tf.keras.backend.clear_session()
        except: pass

def detect_dataset_type(test_data) -> str:
    try:
        sample_batch = next(iter(test_data.take(1)))
        if 'target_categories' in sample_batch:
            categories = sample_batch['target_categories']
            if hasattr(categories, 'numpy'): categories = categories.numpy()
            max_cat = int(np.max(categories[categories > 0])) if np.any(categories > 0) else 0
            return 'IQON3000' if max_cat <= 7 else 'DeepFurniture'
    except Exception as e:
        print(f"[WARN] Could not detect dataset type: {e}")
    return 'IQON3000'

def collect_test_data(test_data) -> Tuple[List[Dict], Dict, str]:
    """Collects all test data into memory and builds a gallery of unique items."""
    print("[INFO] 📥 Collecting test data and building gallery...")
    dataset_type = detect_dataset_type(test_data)
    config = get_dataset_config(dataset_type)
    min_cat, max_cat = config['category_range']
    
    # ▼▼▼▼▼▼▼▼▼▼【ここからロジックを修正】▼▼▼▼▼▼▼▼▼▼
    all_batches = list(tqdm(test_data.as_numpy_iterator(), desc="Collecting batches"))
    
    test_items = []
    gallery_by_category = defaultdict(dict)
    
    print("[INFO] Processing sets and building gallery...")
    for batch in tqdm(all_batches, desc="Processing batches"):
        batch_size = len(batch['query_features'])
        for i in range(batch_size):
            # 各セットを抽出
            item_set = {
                'query_features': batch['query_features'][i],
                'target_features': batch['target_features'][i],
                'query_categories': batch['query_categories'][i],
                'target_categories': batch['target_categories'][i],
                'query_item_ids': batch['query_item_ids'][i],
                'target_item_ids': batch['target_item_ids'][i]
            }
            test_items.append(item_set)

            # ギャラリーを構築
            for feat, cat, item_id in zip(item_set['target_features'], item_set['target_categories'], item_set['target_item_ids']):
                cat_int = int(cat)
                if min_cat <= cat_int <= max_cat and np.any(feat):
                    gallery_by_category[cat_int][str(item_id)] = feat.astype(np.float32)
    # ▲▲▲▲▲▲▲▲▲▲【ここまで修正】▲▲▲▲▲▲▲▲▲▲

    for cat_id in list(gallery_by_category.keys()):
        items = gallery_by_category[cat_id]
        if items:
            gallery_by_category[cat_id] = {'ids': np.array(list(items.keys())), 'features': np.array(list(items.values()))}
    
    return test_items, dict(gallery_by_category), dataset_type


def evaluate_model_comprehensive(model, test_items, gallery_by_category, dataset_type):
    print("[INFO] 🎯 Starting comprehensive evaluation...")
    config = get_dataset_config(dataset_type)
    min_cat, max_cat = config['category_range']
    queries_by_category = defaultdict(list)

    for item in tqdm(test_items, desc="Organizing queries"):
        for target_cat, target_id in zip(item['target_categories'], item['target_item_ids']):
            target_cat_int = int(target_cat)
            if not (min_cat <= target_cat_int <= max_cat): continue
            target_id_str = str(target_id)
            if target_cat_int in gallery_by_category and target_id_str in gallery_by_category[target_cat_int]['ids']:
                queries_by_category[target_cat_int].append({'query_features': item['query_features'], 'target_id': target_id_str})
    
    category_results, all_ranks = {}, []
    for cat_id, queries in tqdm(queries_by_category.items(), desc="Evaluating categories"):
        if not queries: continue
        gallery = gallery_by_category[cat_id]
        query_features_batch = np.array([q['query_features'] for q in queries])
        predictions_batch = model({'query_features': tf.constant(query_features_batch)}, training=False).numpy()
        pred_vectors = predictions_batch[:, cat_id - 1, :]
        similarities_batch = np.dot(pred_vectors, gallery['features'].T)
        sorted_indices_batch = np.argsort(similarities_batch, axis=1)[:, ::-1]
        
        ranks = []
        for i, query in enumerate(queries):
            sorted_gallery_ids = gallery['ids'][sorted_indices_batch[i]]
            rank_list = np.where(sorted_gallery_ids == query['target_id'])[0]
            if len(rank_list) > 0: ranks.append(rank_list[0] + 1)
        
        if ranks:
            all_ranks.extend(ranks)
            ranks_np = np.array(ranks)
            category_results[str(cat_id)] = {'count': len(ranks_np), 'gallery_size': len(gallery['ids']),
                'mrr': np.mean(1.0 / ranks_np), 'r_at_1': np.mean(ranks_np <= 1),
                'r_at_5': np.mean(ranks_np <= 5), 'r_at_10': np.mean(ranks_np <= 10),
                'mnr': np.mean(ranks_np), 'mdr': np.median(ranks_np),
                'rsum': np.sum(ranks_np)}
    
    overall_results = {}
    if all_ranks:
        all_ranks_np = np.array(all_ranks)
        overall_results = {'total_queries': len(all_ranks_np), 'successful_predictions': len(all_ranks_np),
            'mrr': np.mean(1.0 / all_ranks_np), 'r_at_1': np.mean(all_ranks_np <= 1),
            'r_at_5': np.mean(all_ranks_np <= 5), 'r_at_10': np.mean(all_ranks_np <= 10),
            'mnr': np.mean(all_ranks_np), 'mdr': np.median(all_ranks_np),
            'rsum': np.sum(all_ranks_np)}
    return {'dataset': dataset_type, 'overall': overall_results, 'categories': category_results}

def evaluate_model(model, test_data, output_dir: str, data_dir: str):
    """Main evaluation pipeline."""
    print(f"\n[INFO] 🚀 Starting model evaluation pipeline...")
    try:
        os.makedirs(output_dir, exist_ok=True)
        # data_dir引数をcollect_test_dataには渡さなくなりました
        test_items, gallery, dataset_type = collect_test_data(test_data)
        config = get_dataset_config(dataset_type)
        results = evaluate_model_comprehensive(model, test_items, gallery, dataset_type)
        
        if results:
            # 司令塔として各モジュールの関数を呼び出す
            save_evaluation_results(results, config, output_dir)
            display_evaluation_results(results)
            generate_all_visualizations(model, results, test_items, gallery, config, output_dir, data_dir)
        
        clear_memory()
        return results
    except Exception as e:
        print(f"[ERROR] ❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None