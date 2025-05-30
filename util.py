# util.py - Updated for IQON3000 dataset (11 categories: 1-11)
import os, pickle, gzip
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
import pdb
import tensorflow as tf
from PIL import Image

import time
from collections import defaultdict

# ------------------------------------------------------------------
# 0. Shared utilities
# ------------------------------------------------------------------
def append_dataframe_to_csv(df: pd.DataFrame,
                            output_dir: str,
                            filename: str = "result.csv") -> None:
    """Append DataFrame to CSV (header only the first time)."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)
    mode, header = ("a", False) if os.path.exists(csv_path) else ("w", True)
    df.to_csv(csv_path, mode=mode, header=header, index=False)

# ------------------------------------------------------------------
# 1. image loading - Updated for IQON3000 structure
# ------------------------------------------------------------------
def load_scene_image(scene_id: str, scene_root: str = "data/IQON3000"):
    """
    Load scene image from IQON3000 dataset structure
    """
    path = os.path.join(scene_root, str(scene_id), f"{scene_id}.jpg")
    if not os.path.exists(path):
        # Try alternative path structure
        path = os.path.join(scene_root, str(scene_id), "image.jpg")
    return Image.open(path).convert("RGB") if os.path.exists(path) else None

def load_furniture_image(item_id: str, furn_root: str = "data/IQON3000"):
    """
    Load furniture/item image from IQON3000 dataset structure
    """
    if item_id == "0":    # 0 is padding
        return None
    
    # Try to find the image in the set directories
    for set_dir in os.listdir(furn_root):
        set_path = os.path.join(furn_root, set_dir)
        if os.path.isdir(set_path):
            img_path = os.path.join(set_path, f"{item_id}_m.jpg")
            if os.path.exists(img_path):
                return Image.open(img_path).convert("RGB")
    
    return None

def safe_load_furniture_image(item_id: str, furn_root: str, thumb_size=(150, 150)):
    try:
        img = load_furniture_image(item_id, furn_root)
        return img if img else Image.new("RGB", thumb_size, "gray")
    except Exception:
        return Image.new("RGB", thumb_size, "gray")

# ------------------------------------------------------------------
# 2. feature dictionary and distance search
# ------------------------------------------------------------------
def build_item_feature_dict(model, test_gen) -> Dict[str, np.ndarray]:
    """Build a dictionary of item_id → feature from all test batches."""
    vecs: Dict[str, np.ndarray] = {}
    for ((concat_ft, _, _, _, _, q_ids, t_ids, _), _) in test_gen:
        cf = concat_ft.numpy()
        B2, N, _ = cf.shape
        half = B2 // 2

        # Query
        for b in range(half):
            for vec, iid in zip(cf[b], q_ids.numpy()[b]):
                if iid > 0: vecs[str(int(iid))] = vec
        # Target
        for b in range(half, B2):
            for vec, iid in zip(cf[b], t_ids.numpy()[b-half]):
                if iid > 0: vecs[str(int(iid))] = vec
    print(f"[INFO] built feature‑dict: {len(vecs)} items")
    return vecs

def find_topk_similar_items_by_euclidean(query_vec: np.ndarray,
                                         item_dict: Dict[str, np.ndarray],
                                         k: int = 3,
                                         exclude: Set[str] | None = None
                                         ) -> List[Tuple[str, float]]:
                                         
    exclude = exclude or set()
    dists = [(iid, float(np.linalg.norm(query_vec - v)))
             for iid, v in item_dict.items()
             if iid not in exclude and iid != "0"]
    return sorted(dists, key=lambda x: x[1])[:k]

# ------------------------------------------------------------------
# 3. mini-batch average rank (for in-training metric) - Updated for 11 categories
# ------------------------------------------------------------------
def inbatch_cat_rank(predicted_vectors, ground_truth_vectors,
                     category_labels, step_idx: int):
    """
    Returns a value normalized 0-1 by the average rank divided by the gallery size.
    Updated for IQON3000 dataset with 11 categories (1-11)
    """
    pred, gt, cat = (x.numpy() for x in
                     (predicted_vectors, ground_truth_vectors, category_labels))
    
    # Handle different prediction shapes
    if len(pred.shape) == 3 and pred.shape[1] == 11:
        # pred is (B, 11, D) - category predictions
        B, N, _ = gt.shape
        gallery, lookup = [], []
        for b in range(B):
            for n in range(N):
                if 1 <= cat[b, n] <= 11:
                    gallery.append(gt[b, n])
                    lookup.append((b, n))
        
        if not gallery:
            return tf.constant(-1., tf.float32)

        gallery = np.stack(gallery)
        total = gallery.shape[0]
        ranks = []

        for b in range(B):
            for n in range(N):
                cid = int(cat[b, n])
                if not 1 <= cid <= 11: 
                    continue
                q = pred[b, cid-1]  # 1-based to 0-based index
                sims = gallery @ q
                idx = lookup.index((b, n))
                rank = 1 + np.sum(sims > sims[idx])
                ranks.append(rank)
    else:
        # pred is (B, N, D) - direct predictions
        B, N, _ = gt.shape
        gallery, lookup = [], []
        for b in range(B):
            for n in range(N):
                if 1 <= cat[b, n] <= 11:
                    gallery.append(gt[b, n])
                    lookup.append((b, n))
        
        if not gallery:
            return tf.constant(-1., tf.float32)

        gallery = np.stack(gallery)
        total = gallery.shape[0]
        ranks = []

        for b in range(B):
            for n in range(N):
                cid = int(cat[b, n])
                if not 1 <= cid <= 11: 
                    continue
                q = pred[b, n]
                sims = gallery @ q
                idx = lookup.index((b, n))
                rank = 1 + np.sum(sims > sims[idx])
                ranks.append(rank)

    if not ranks: 
        return tf.constant(-1., tf.float32)
    return tf.constant(np.mean(ranks) / total, tf.float32)

# ------------------------------------------------------------------
# 4. overall test set ranking (XY / YX) - Updated for 11 categories
# ------------------------------------------------------------------
def gather_test_items(test_generator):
    """[(query_feats, query_cats, target_feats, target_cats), ...]"""
    items = []
    for (Xconcat, _, _, catQ, catP, _, _, _), _ in test_generator:
        half = Xconcat.shape[0] // 2
        for i in range(half):
            items.append((Xconcat[i].numpy(), catQ[i].numpy(), Xconcat[i+half].numpy(), catP[i].numpy()))
    return items

def compute_category_whitening_params(embeddings, epsilon=1e-10, eigenvalue_threshold_factor=1e-3, max_eigenvalue_inv=1e3):
    """Compute category-specific whitening parameters"""
    try:
        tensor_data = tf.convert_to_tensor(embeddings, dtype=tf.float32)
        mean_vec = tf.reduce_mean(tensor_data, axis=0)
        centered_data = tensor_data - mean_vec[tf.newaxis, :]
        
        n_samples = tf.cast(tf.shape(centered_data)[0], tf.float32)
        covariance = tf.matmul(centered_data, centered_data, transpose_a=True) / n_samples
        eigenvalues, eigenvectors = tf.linalg.eigh(covariance)
        
        threshold = tf.maximum(epsilon, eigenvalue_threshold_factor * tf.reduce_max(eigenvalues))
        valid_mask = eigenvalues > threshold
        
        if tf.reduce_any(valid_mask):
            indices = tf.where(valid_mask)
            valid_values = tf.gather_nd(eigenvalues, indices)
            col_indices = indices[:, 0]
            valid_vectors = tf.gather(eigenvectors, col_indices, axis=1)
            
            inv_sqrt_values = tf.clip_by_value(
                1.0 / tf.sqrt(valid_values), -max_eigenvalue_inv, max_eigenvalue_inv
            )
            diag_matrix = tf.linalg.diag(inv_sqrt_values)
            whitening_matrix = tf.matmul(
                valid_vectors, tf.matmul(diag_matrix, tf.transpose(valid_vectors))
            )
        else:
            whitening_matrix = tf.eye(tf.shape(covariance)[0], dtype=tf.float32)
        
        return mean_vec.numpy(), whitening_matrix.numpy()
    except Exception as e:
        print(f"  Warning: Category whitening computation failed: {e}")
        return np.mean(embeddings, axis=0), np.eye(embeddings.shape[1])

def apply_whitening_transformation(vectors, mean_vec, whitening_matrix, epsilon=1e-10):
    """Apply whitening transformation and normalize vectors"""
    try:
        centered = vectors - mean_vec
        whitened = np.dot(centered, whitening_matrix.T)
        norms = np.linalg.norm(whitened, axis=1, keepdims=True)
        return whitened / np.maximum(norms, epsilon)
    except Exception as e:
        print(f"  Warning: Whitening application failed: {e}")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, epsilon)

def combine_direction_results(xy_results, yx_results):
    """Combine XY and YX direction results"""
    combined = {}
    
    for key in xy_results.keys():
        if isinstance(xy_results[key], list):
            combined[key] = xy_results[key] + yx_results[key]
        elif isinstance(xy_results[key], dict):
            combined[key] = {}
            for sub_key in xy_results[key].keys():
                if isinstance(xy_results[key][sub_key], list):
                    combined[key][sub_key] = xy_results[key][sub_key] + yx_results[key][sub_key]
    
    return combined

def compute_global_rank(model, test_generator, output_dir="output", 
                       checkpoint_path=None, hard_negative_threshold=0.9,
                       top_k_values=[5, 10, 20], 
                       top_k_percentages=[5, 10, 20],
                       combine_directions=True, enable_visualization=False) -> None:
    """
    Category-specific whitening and detailed Top-K metrics rank evaluation
    Updated for IQON3000 dataset with 11 categories (1-11)
    """
    
    # Numerical stability constants
    EPSILON = 1e-10
    EIGENVALUE_THRESHOLD_FACTOR = 1e-3
    MAX_EIGENVALUE_INV = 1e3
    MIN_WEIGHT = 0.01
    CORRECT_MATCH_THRESHOLD = 1e-8
    PERCENTILES = [0.05, 0.10, 0.20]
    
    print(f"[INFO] Starting category-specific whitening rank evaluation for IQON3000 (threshold = {hard_negative_threshold})")
    
    # Build gallery from test data
    test_items = gather_test_items(test_generator)
    if not test_items:
        print("[WARN] No test items found")
        return
    
    # Extract features and categories for both directions
    galleries = {"XY": [], "YX": []}
    categories = {"XY": [], "YX": []}
    
    for q_feats, q_cats, t_feats, t_cats in test_items:
        for vec, cat in zip(t_feats, t_cats):
            if 1 <= cat <= 11: 
                galleries["XY"].append(vec)
                categories["XY"].append(cat)
        for vec, cat in zip(q_feats, q_cats):
            if 1 <= cat <= 11: 
                galleries["YX"].append(vec)
                categories["YX"].append(cat)
    
    # Convert to numpy arrays and build category maps
    gallery_maps = {"XY": {}, "YX": {}}
    
    for direction in ["XY", "YX"]:
        galleries[direction] = np.array(galleries[direction])
        categories[direction] = np.array(categories[direction])
        
        gallery_maps[direction] = {cat_id: [] for cat_id in range(1, 12)}  # 1-11 categories
        for idx, cat_id in enumerate(categories[direction]):
            gallery_maps[direction][cat_id].append(idx)
    
    # Pre-compute category-specific whitening parameters and galleries
    print("[INFO] Computing category-specific whitening parameters and galleries for 11 categories")
    category_whitening_params = {}
    whitened_galleries = {"XY": {}, "YX": {}}
    
    for cat_id in range(1, 12):  # 1-11 categories
        if not gallery_maps["XY"].get(cat_id):
            continue
            
        print(f"  Processing category {cat_id}")
        
        try:
            vectors_xy = galleries["XY"][gallery_maps["XY"][cat_id]]
            embeddings_xy = np.array([
                model.infer_single_set(vec[None, :])[cat_id - 1]  # 1-based to 0-based
                for vec in vectors_xy
            ])
            
            mean_vec, whitening_matrix = compute_category_whitening_params(
                embeddings_xy, EPSILON, EIGENVALUE_THRESHOLD_FACTOR, MAX_EIGENVALUE_INV
            )
            category_whitening_params[cat_id] = (mean_vec, whitening_matrix)
            
            whitened_galleries["XY"][cat_id] = apply_whitening_transformation(
                embeddings_xy, mean_vec, whitening_matrix, EPSILON
            )
            
            if gallery_maps["YX"].get(cat_id):
                vectors_yx = galleries["YX"][gallery_maps["YX"][cat_id]]
                embeddings_yx = np.array([
                    model.infer_single_set(vec[None, :])[cat_id - 1]  # 1-based to 0-based
                    for vec in vectors_yx
                ])
                
                whitened_galleries["YX"][cat_id] = apply_whitening_transformation(
                    embeddings_yx, mean_vec, whitening_matrix, EPSILON
                )
            
            tf.keras.backend.clear_session()
            
        except Exception as e:
            print(f"  Error processing category {cat_id}: {e}")
    
    # Evaluate search performance
    def evaluate_direction_with_detailed_metrics(direction):
        """Calculate detailed search metrics for specified direction"""
        print(f"[INFO] Evaluating {direction} direction")
        
        metrics = {
            "ranks": [], "true_ranks": [], "weighted_ranks": [],
            "mrrs": [], "hard_neg_rates": [],
            "percentile_hits": {p: [] for p in PERCENTILES},
            "true_percentile_hits": {p: [] for p in PERCENTILES},
            "percentile_weights": {p: [] for p in PERCENTILES},
            "gallery_sizes": [], "categories": []
        }
        
        # Initialize Top-K metrics
        for k in top_k_values:
            metrics[f"top_{k}_hits"] = []
            metrics[f"top_{k}_true_hits"] = []
            metrics[f"top_{k}_weighted_hits"] = []
        
        # Initialize Top-K% metrics
        for k_pct in top_k_percentages:
            metrics[f"top_{k_pct}pct_hits"] = []
            metrics[f"top_{k_pct}pct_true_hits"] = []
            metrics[f"top_{k_pct}pct_weighted_hits"] = []
        
        for item_idx, (q_feats, q_cats, t_feats, t_cats) in enumerate(test_items):
            if direction == "XY":
                input_feats, target_cats, target_feats = q_feats, t_cats, t_feats
            else:  # YX
                input_feats, target_cats, target_feats = t_feats, q_cats, q_feats
            
            predictions = model.infer_single_set(input_feats)  # (11, dim)
            
            for i, cat_id in enumerate(target_cats):
                cat_id = int(cat_id)
                
                if not (1 <= cat_id <= 11) or cat_id not in category_whitening_params:
                    continue
                    
                gallery_indices = gallery_maps[direction].get(cat_id, [])
                if not gallery_indices:
                    continue
                
                try:
                    mean_vec, whitening_matrix = category_whitening_params[cat_id]
                    gallery_whitened = whitened_galleries[direction][cat_id]
                    gallery_size = len(gallery_whitened)
                    
                    query_raw = predictions[cat_id - 1]  # 1-based to 0-based
                    query_whitened = apply_whitening_transformation(
                        query_raw[np.newaxis, :], mean_vec, whitening_matrix, EPSILON
                    )[0]
                    
                    target_emb = model.infer_single_set(target_feats[i][np.newaxis, :])[cat_id - 1]
                    target_whitened = apply_whitening_transformation(
                        target_emb[np.newaxis, :], mean_vec, whitening_matrix, EPSILON
                    )[0]
                    
                    similarities = gallery_whitened @ query_whitened
                    
                    distances = np.sum((gallery_whitened - target_whitened) ** 2, axis=1)
                    correct_idx = np.argmin(distances)
                    
                    if distances[correct_idx] > CORRECT_MATCH_THRESHOLD:
                        continue
                    
                    true_rank = 1 + np.sum(similarities > similarities[correct_idx])
                    
                    accepted_indices = np.where(similarities >= hard_negative_threshold)[0]
                    if correct_idx not in accepted_indices:
                        accepted_indices = np.append(accepted_indices, correct_idx)
                    
                    accepted_sims = similarities[accepted_indices]
                    accepted_ranks = np.array([
                        1 + np.sum(similarities > similarities[idx]) 
                        for idx in accepted_indices
                    ])
                    
                    best_rank = np.min(accepted_ranks)
                    
                    weights = (accepted_sims - hard_negative_threshold) / (1.0 - hard_negative_threshold)
                    weights = np.clip(weights, MIN_WEIGHT, 1.0)
                    weights = weights / np.sum(weights)
                    
                    weighted_rank = np.sum(weights * accepted_ranks)
                    
                    # Store metrics
                    metrics["ranks"].append(best_rank)
                    metrics["true_ranks"].append(true_rank)
                    metrics["weighted_ranks"].append(weighted_rank)
                    metrics["mrrs"].append(1.0 / best_rank)
                    metrics["gallery_sizes"].append(gallery_size)
                    metrics["categories"].append(cat_id)
                    
                    hard_neg_count = len(accepted_indices) - (1 if correct_idx in accepted_indices else 0)
                    hard_neg_rate = hard_neg_count / (gallery_size - 1) if gallery_size > 1 else 0
                    metrics["hard_neg_rates"].append(hard_neg_rate)
                    
                    # Percentile metrics
                    for p in PERCENTILES:
                        threshold = int(p * gallery_size)
                        metrics["percentile_hits"][p].append(best_rank <= threshold)
                        metrics["true_percentile_hits"][p].append(true_rank <= threshold)
                        
                        in_top_p = accepted_ranks <= threshold
                        p_weight = np.sum(weights[in_top_p]) if np.any(in_top_p) else 0.0
                        metrics["percentile_weights"][p].append(p_weight)
                    
                    # Top-K metrics (absolute values)
                    for k in top_k_values:
                        if k <= gallery_size:
                            true_in_top_k = true_rank <= k
                            top_k_similarities = similarities[np.argsort(-similarities)[:k]]
                            acceptable_in_top_k = np.any(top_k_similarities >= hard_negative_threshold)
                            success_traditional = true_in_top_k or acceptable_in_top_k
                            
                            metrics[f"top_{k}_hits"].append(success_traditional)
                            metrics[f"top_{k}_true_hits"].append(true_in_top_k)
                            
                            acceptable_count = np.sum(top_k_similarities >= hard_negative_threshold)
                            weight = min(1.0, acceptable_count / k)
                            metrics[f"top_{k}_weighted_hits"].append(weight)
                        else:
                            metrics[f"top_{k}_hits"].append(True)
                            metrics[f"top_{k}_true_hits"].append(True)
                            metrics[f"top_{k}_weighted_hits"].append(1.0)
                    
                    # Top-K% metrics (relative values)
                    for k_pct in top_k_percentages:
                        k_threshold = max(1, int(gallery_size * k_pct / 100.0))
                        true_in_top_k_pct = true_rank <= k_threshold
                        success_traditional = true_in_top_k_pct
                        
                        sorted_indices = np.argsort(-similarities)[:k_threshold]
                        total_weight = 1.0
                        included_weight = 0.0
                        
                        if correct_idx in sorted_indices:
                            included_weight += 1.0
                        
                        acceptable_indices = np.where(similarities >= hard_negative_threshold)[0]
                        if len(acceptable_indices) > 0:
                            target_similarities = gallery_whitened[acceptable_indices] @ target_whitened
                            acceptable_weights = np.clip(
                                (target_similarities - hard_negative_threshold) / (1.0 - hard_negative_threshold), 
                                0.0, 1.0
                            )
                            total_weight += np.sum(acceptable_weights)
                            
                            included_mask = np.isin(acceptable_indices, sorted_indices)
                            if np.any(included_mask):
                                included_weight += np.sum(acceptable_weights[included_mask])
                        
                        weighted_score = included_weight / total_weight if total_weight > 0 else 0.0
                        
                        metrics[f"top_{k_pct}pct_hits"].append(success_traditional)
                        metrics[f"top_{k_pct}pct_true_hits"].append(true_in_top_k_pct)
                        metrics[f"top_{k_pct}pct_weighted_hits"].append(weighted_score)
                    
                except Exception as e:
                    print(f"  Error evaluating item {item_idx}, category {cat_id}: {e}")
        
        return metrics
    
    # Evaluate both directions
    results = {}
    for direction in ["XY", "YX"]:
        results[direction] = evaluate_direction_with_detailed_metrics(direction)
        tf.keras.backend.clear_session()
    
    # Combine directions if requested
    if combine_directions:
        combined_results = combine_direction_results(results["XY"], results["YX"])
        results["COMBINED"] = combined_results
        print(f"[INFO] Combined XY/YX direction results")
    
    # Calculate summary statistics and create dataframes
    result_rows = []
    
    if combine_directions:
        directions_to_process = ["COMBINED"]
    else:
        directions_to_process = ["XY", "YX"]
    
    # Category names for IQON3000 dataset (11 categories)
    category_names = {
        1: "インナー系",
        2: "ボトムス系", 
        3: "シューズ系",
        4: "バッグ系",
        5: "アクセサリー系",
        6: "帽子",
        7: "Tシャツ・カットソー系",
        8: "シャツ・ブラウス系", 
        9: "ニット・セーター系",
        10: "アウター系（ジャケット+コート）",
        11: "その他（ワンピース・ドレス等）"
    }
    
    for direction in directions_to_process:
        metrics = results[direction]
        
        ranks = np.array(metrics["ranks"])
        true_ranks = np.array(metrics["true_ranks"])
        weighted_ranks = np.array(metrics["weighted_ranks"])
        mrrs = np.array(metrics["mrrs"])
        hard_neg_rates = np.array(metrics["hard_neg_rates"])
        gallery_sizes = np.array(metrics["gallery_sizes"])
        categories = np.array(metrics["categories"])
        
        norm_ranks = ranks / gallery_sizes
        norm_true_ranks = true_ranks / gallery_sizes
        norm_weighted_ranks = weighted_ranks / gallery_sizes
        
        # Calculate category statistics
        for cat_id in range(1, 12):  # 1-11