"""
Unified Dataset Processing Script
===============================
Processes IQON3000 (fashion) and DeepFurniture datasets with CLIP features

Features:
- IQON3000: 17 categories, user-based splitting, category validation
- DeepFurniture: 11 categories, inclusion relationship removal
- Unified L2 normalization and format conversion
- Comprehensive error handling and duplicate removal

Usage:
------
# IQON3000 with user-based splitting (recommended)
python make_datasets.py --dataset iqon3000 --input-dir data/IQON3000 --output-dir datasets/IQON3000 --user-based

# DeepFurniture with inclusion removal
python make_datasets.py   --dataset deepfurniture   --image-dir data/DeepFurniture/furnitures   --output-dir datasets/DeepFurniture  --annotations-json data/DeepFurniture/metadata/annotations.json   --furnitures-jsonl data/DeepFurniture/metadata/furnitures.jsonl
"""

import os
import json
import pickle
import gzip
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

# =============================================================================
# Configuration
# =============================================================================

# IQON3000: 17 categories
IQON3000_CATEGORIES = {
    1: "outerwear", 2: "tops", 3: "dresses", 4: "pants", 5: "skirts",
    6: "shoes", 7: "bags", 8: "hats", 9: "watches", 10: "accessories",
    11: "fashion_goods", 12: "wallets", 13: "legwear", 
    14: "underwear", 15: "beauty", 16: "glasses", 17: "others"
}

# Japanese category mapping for IQON3000
IQON3000_MAPPING = {
    # Outerwear (1)
    "ジャケット": 1, "コート": 1, "アウター": 1, "カーディガン": 1, 
    "ブルゾン": 1, "ダウン": 1, "パーカー": 1, "浴衣": 1, "着物": 1, "ルームウェア": 1,
    # Tops (2)
    "Tシャツ": 2, "カットソー": 2, "シャツ": 2, "ブラウス": 2, 
    "ニット": 2, "セーター": 2, "ベスト": 2, "タンクトップ": 2, 
    "キャミソール": 2, "チュニック": 2, "トップス": 2, "インナー": 2,
    # Dresses (3)
    "ワンピース": 3, "ドレス": 3,
    # Pants (4)
    "パンツ": 4, "ショートパンツ": 4, "ロングパンツ": 4, 
    "ジーンズ": 4, "デニム": 4, "レギンス": 4, "スラックス": 4,
    # Skirts (5)
    "スカート": 5, "ロングスカート": 5,
    # Shoes (6)
    "シューズ": 6, "靴": 6, "スニーカー": 6, "サンダル": 6, 
    "ブーツ": 6, "パンプス": 6, "ルームシューズ": 6, "ローファー": 6,
    # Bags (7)
    "バッグ": 7, "トートバッグ": 7, "ショルダーバッグ": 7, 
    "ハンドバッグ": 7, "クラッチバッグ": 7, "ボストンバッグ": 7, 
    "リュック": 7, "ポーチ": 7,
    # Hats (8)
    "帽子": 8, "ハット": 8, "キャップ": 8, "ニット帽": 8, 
    "ベレー帽": 8, "キャスケット": 8,
    # Others (9-17)
    "時計": 9,
    "アクセサリー": 10, "ジュエリー": 10, "ネックレス": 10, 
    "ブレスレット": 10, "イヤリング": 10, "リング": 10, 
    "ピアス": 10, "ブローチ": 10, "ヘアアクセサリー": 10,
    "ベルト": 11, "スカーフ": 11, "ストール": 11, "マフラー": 11, 
    "手袋": 11, "ファッション雑貨": 11, "ファッション小物": 11, 
    "インテリア": 11, "ステーショナリー": 11,
    "財布": 12, "キーケース・キーホルダー": 12, "小物": 12,
    "ソックス・靴下": 13, "タイツ・ストッキング": 13, "レッグウェア": 13,
    "アンダーウェア": 14, "水着": 14,
    "コスメ": 15, "ネイル": 15, "フレグランス": 15, "ボディケア": 15,
    "サングラス": 16, "メガネ": 16,
    "傘・日傘": 17, "傘": 17, "その他": 17,
}

DEFAULT_IQON3000_CATEGORY = 17

# DeepFurniture: 11 categories
DEEPFURNITURE_CATEGORIES = {
    1: "chair", 2: "table", 3: "sofa", 4: "bed", 5: "cabinet", 
    6: "lamp", 7: "bookshelf", 8: "desk", 9: "dresser", 
    10: "nightstand", 11: "other_furniture"
}

# =============================================================================
# Utility Functions
# =============================================================================

# 特徴量をL2正規化
def normalize_features_l2(features):
    """L2 normalization for feature vectors"""
    if isinstance(features, torch.Tensor):
        norm = features.norm(dim=-1, keepdim=True)
        norm[norm == 0] = 1e-9
        return features / norm
    else:
        norm = np.linalg.norm(features, axis=-1, keepdims=True)
        norm[norm == 0] = 1e-9
        return features / norm

# CLIP画像認識モデルの初期化
def setup_clip_model(device=None):
    """Initialize CLIP model and processor"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    model.eval()
    
    return model, processor, device

# データローダー用のバッチ処理関数
def collate_fn(batch):
    """DataLoader collate function with error handling"""
    inputs_dict = {}
    item_ids_list, cat_ids_list = [], []
    
    for b_input, item_id, cat_id in batch:
        if b_input is None: 
            continue
        for key, val in b_input.items(): 
            inputs_dict.setdefault(key, []).append(val)
        item_ids_list.append(item_id)
        cat_ids_list.append(cat_id)
    
    if not inputs_dict: 
        return {'pixel_values': torch.empty(0, 3, 224, 224)}, [], []
    
    try:
        final_inputs = {}
        for key, val_list in inputs_dict.items():
            if val_list:
                final_inputs[key] = torch.stack(val_list)
            elif key == 'pixel_values': 
                final_inputs[key] = torch.empty(0, 3, 224, 224)
        return final_inputs, item_ids_list, cat_ids_list
    except RuntimeError as e: 
        print(f"Collate error: {e}")
        raise e

# データを指定長さにパディング/切り詰め
def pad_or_truncate(data_list, max_len, pad_value, dtype_override=None):
    """Pad or truncate data to fixed length"""
    if not data_list: 
        if isinstance(pad_value, np.ndarray): 
            return np.array([pad_value.copy() for _ in range(max_len)], 
                          dtype=dtype_override or pad_value.dtype)
        return np.array([pad_value] * max_len, 
                       dtype=dtype_override or type(pad_value))
    
    processed_list = list(data_list[:max_len]) 
    num_padding = max_len - len(processed_list)
    
    if num_padding > 0:
        if processed_list and isinstance(processed_list[0], np.ndarray): 
            processed_list.extend([pad_value.copy() for _ in range(num_padding)])
        elif processed_list: 
            processed_list.extend([pad_value] * num_padding)
        else:
            if isinstance(pad_value, np.ndarray): 
                processed_list = [pad_value.copy() for _ in range(max_len)]
            else: 
                processed_list = [pad_value] * max_len
    
    final_dtype = dtype_override
    if not final_dtype:
        if processed_list and isinstance(processed_list[0], np.ndarray): 
            final_dtype = processed_list[0].dtype
        elif processed_list: 
            final_dtype = type(processed_list[0])
        elif isinstance(pad_value, np.ndarray): 
            final_dtype = pad_value.dtype
        else: 
            final_dtype = type(pad_value)
    
    return np.array(processed_list, dtype=final_dtype)

# =============================================================================
# Data Splitting
# =============================================================================

# ユーザー単位でデータを訓練/検証/テストに分割
def split_data_by_users(set_data, test_size=0.2, val_size=0.1, random_state=42):
    """User-based data splitting to prevent data leakage"""
    print("Executing user-based data splitting...")
    
    # Group data by users
    user_to_data = {}
    for data_item in set_data:
        if isinstance(data_item, tuple) and len(data_item) == 2:
            (user_id, coord_id), items_data = data_item
            user_to_data.setdefault(user_id, []).append(data_item)
    
    # Statistics
    total_users = len(user_to_data)
    total_scenes = sum(len(data_list) for data_list in user_to_data.values())
    
    print(f"Pre-split stats:")
    print(f"  Total users: {total_users}")
    print(f"  Total scenes: {total_scenes}")
    print(f"  Avg scenes per user: {total_scenes/total_users:.1f}")
    
    # Shuffle and split users
    np.random.seed(random_state)
    shuffled_users = np.random.permutation(list(user_to_data.keys()))
    
    # Determine split points by cumulative scene count
    cumulative_scenes = 0
    train_users, val_users, test_users = [], [], []
    
    for user_id in shuffled_users:
        user_scene_count = len(user_to_data[user_id])
        current_ratio = cumulative_scenes / total_scenes
        
        if current_ratio < (1 - test_size - val_size):
            train_users.append(user_id)
        elif current_ratio < (1 - test_size):
            val_users.append(user_id)
        else:
            test_users.append(user_id)
        
        cumulative_scenes += user_scene_count
    
    # Assign data to splits
    train_data = []
    val_data = []
    test_data = []
    
    for user_id in train_users:
        train_data.extend(user_to_data[user_id])
    for user_id in val_users:
        val_data.extend(user_to_data[user_id])
    for user_id in test_users:
        test_data.extend(user_to_data[user_id])
    
    # Verify split results
    print(f"\nUser-based split results:")
    print(f"  Train: {len(train_users)} users, {len(train_data)} scenes")
    print(f"  Val:   {len(val_users)} users, {len(val_data)} scenes")
    print(f"  Test:  {len(test_users)} users, {len(test_data)} scenes")
    
    # Check for overlaps
    user_sets = [set(train_users), set(val_users), set(test_users)]
    overlaps = [
        (user_sets[0] & user_sets[1], "Train-Val"),
        (user_sets[0] & user_sets[2], "Train-Test"), 
        (user_sets[1] & user_sets[2], "Val-Test")
    ]
    
    print(f"\nOverlap check:")
    all_clean = True
    for overlap_set, name in overlaps:
        print(f"  {name} overlap: {len(overlap_set)} users")
        if overlap_set:
            all_clean = False
    
    if all_clean:
        print("  No user overlap! Perfect separation achieved")
    
    return train_data, val_data, test_data

# =============================================================================
# Normalization
# =============================================================================

# 訓練データから正規化用の統計量を計算
def compute_normalization_stats(train_sets):
    """Compute normalization statistics from training data"""
    print("Computing normalization statistics from training data...")
    
    all_train_features = []
    
    for (user_id, coord_id), items_data in train_sets:
        for item_id, category, feature in items_data:
            all_train_features.append(feature)
    
    if not all_train_features:
        print("Warning: No training features found")
        return None
    
    train_features_array = np.stack(all_train_features)
    
    # Compute statistics
    feature_mean = np.mean(train_features_array, axis=0)
    feature_std = np.std(train_features_array, axis=0)
    feature_std[feature_std == 0] = 1e-9
    
    normalization_stats = {
        'method': 'l2_norm',  # Using unified L2 normalization
        'train_mean': feature_mean,
        'train_std': feature_std,
        'train_feature_count': len(all_train_features)
    }
    
    print(f"Normalization stats computed from {len(all_train_features)} training features")
    return normalization_stats

# 計算した統計量で特徴量を正規化
def apply_normalization_with_stats(features, normalization_stats):
    """Apply normalization using computed statistics"""
    if normalization_stats is None:
        return normalize_features_l2(features)
    
    method = normalization_stats.get('method', 'l2_norm')
    
    if method == 'l2_norm':
        return normalize_features_l2(features)
    elif method == 'z_score':
        train_mean = normalization_stats['train_mean']
        train_std = normalization_stats['train_std']
        return (features - train_mean) / train_std
    else:
        raise ValueError(f"Unknown normalization method: {method}")

# =============================================================================
# Category Centers
# =============================================================================

# 各カテゴリの代表ベクトル（中心）を計算
def compute_category_centers(features_dir, category_definitions, category_ids):
    """Compute category centers with unified normalization"""
    print("Computing category centers...")
    
    train_path = os.path.join(features_dir, 'train.pkl')
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found")
        return
    
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    
    # Check data structure
    if len(train_data) >= 9:
        query_features, target_features, _, query_categories, target_categories = train_data[:5]
    else:
        print("Error: Unexpected data format")
        return
    
    print(f"Loaded training data: {len(query_features)} sets")
    
    # Get feature dimension
    if len(query_features) > 0 and query_features[0].shape[-1] > 0:
        embedding_dim = query_features[0].shape[-1]
    else:
        print("Error: No valid features found")
        return
    
    # Collect features by category
    category_features = {cat_id: [] for cat_id in category_ids}
    
    all_features = np.concatenate([query_features, target_features], axis=0)
    all_categories = np.concatenate([query_categories, target_categories], axis=0)
    
    print("Collecting features by category...")
    valid_count = 0
    
    for set_idx in tqdm(range(len(all_features))):
        for item_idx in range(len(all_features[set_idx])):
            feat = all_features[set_idx][item_idx]
            cat = all_categories[set_idx][item_idx]
            
            # Skip padding and out-of-range categories
            if cat == 0 or np.all(feat == 0) or cat not in category_ids:
                continue
            
            # Check and fix normalization
            feat_norm = np.linalg.norm(feat)
            if feat_norm < 0.9 or feat_norm > 1.1:
                feat = normalize_features_l2(feat.reshape(1, -1))[0]
            
            category_features[cat].append(feat)
            valid_count += 1
    
    print(f"Valid features: {valid_count}")
    
    # Compute category centers
    category_centers_dict = {}
    
    for cat_id in category_ids:
        if category_features[cat_id]:
            center_vec = np.mean(np.stack(category_features[cat_id]), axis=0)
            center_vec = normalize_features_l2(center_vec.reshape(1, -1))[0]
            category_centers_dict[cat_id] = center_vec.tolist()
            print(f"Category {cat_id}: {len(category_features[cat_id])} features")
        else:
            print(f"Warning: No features for category {cat_id}. Random initialization.")
            rand_vec = np.random.randn(embedding_dim).astype(np.float32)
            rand_vec = normalize_features_l2(rand_vec.reshape(1, -1))[0]
            category_centers_dict[cat_id] = rand_vec.tolist()
    
    # Save
    output_path = os.path.join(features_dir, 'category_centers.pkl.gz')
    with gzip.open(output_path, 'wb') as f:
        pickle.dump(category_centers_dict, f)
    
    print(f"Category centers saved: {len(category_centers_dict)} -> {output_path}")
    
    # Print summary
    print("Category center summary:")
    for cat_id in sorted(category_centers_dict.keys()):
        cat_name = category_definitions.get(cat_id, f"Category {cat_id}")
        feature_count = len(category_features[cat_id])
        print(f"  {cat_id}: {cat_name} - {feature_count} features")
    
    return category_centers_dict

# =============================================================================
# Format Conversion
# =============================================================================

# アイテムデータの構造を解析
def parse_items_data(items_data, set_idx):
    """Parse item data structure"""
    scene_id = f"scene_{set_idx}"
    items_to_process = None
    
    if isinstance(items_data, tuple) and len(items_data) == 2:
        first_elem, second_elem = items_data
        if isinstance(first_elem, tuple) and len(first_elem) == 2:
            user_id, coord_id = first_elem
            scene_id = f"{user_id}/{coord_id}"
            items_to_process = second_elem
        else:
            items_to_process = items_data
    else:
        items_to_process = items_data
    
    return scene_id, items_to_process

# アイテムをクエリとターゲットに分割 
def split_query_target(items_to_process):
    """Split items into query and target sets"""
    random.shuffle(items_to_process)
    split_idx = len(items_to_process) // 2
    
    if split_idx == 0:
        return None, None, False
    
    query_items = items_to_process[:split_idx]
    target_items = items_to_process[split_idx:]
    
    # Check for overlaps
    query_ids = set(item[0] for item in query_items)
    target_ids = set(item[0] for item in target_items)
    
    had_overlap = bool(query_ids & target_ids)
    
    if had_overlap:
        # Remove overlaps
        target_items = [item for item in target_items if item[0] not in query_ids]
        if not target_items:
            return None, None, True
    
    return query_items, target_items, had_overlap

#  DeepFurniture形式に変換
def convert_to_deepfurniture_format(sets_data, output_file, normalization_stats=None, max_item_num=10):
    """Convert to DeepFurniture format with unified normalization"""
    
    # Initialize lists
    q_feats_list, t_feats_list = [], []
    q_cats_list, t_cats_list = [], []
    q_ids_list, t_ids_list = [], []
    scene_ids_list, set_sizes_list = [], []
    
    skipped_sets = 0
    overlap_warnings = 0
    
    print(f"Converting to DeepFurniture format: {os.path.basename(output_file)}")
    print(f"Normalization: {normalization_stats['method'] if normalization_stats else 'fallback_l2'}")
    
    for set_idx, items_data in enumerate(tqdm(sets_data, desc="Converting")):
        # Parse data structure
        scene_id, items_to_process = parse_items_data(items_data, set_idx)
        
        if len(items_to_process) < 2:
            skipped_sets += 1
            continue
        
        # Split into query and target
        query_items, target_items, had_overlap = split_query_target(items_to_process)
        
        if had_overlap:
            overlap_warnings += 1
        
        if not query_items or not target_items:
            skipped_sets += 1
            continue
        
        # Apply normalization
        q_feats_norm = [apply_normalization_with_stats(feat, normalization_stats) 
                       for _, _, feat in query_items]
        t_feats_norm = [apply_normalization_with_stats(feat, normalization_stats) 
                       for _, _, feat in target_items]
        
        # Apply padding
        feature_dim = q_feats_norm[0].shape[0]
        zero_pad = np.zeros(feature_dim, dtype=np.float32)
        
        q_ids_list.append(pad_or_truncate([item[0] for item in query_items], max_item_num, '0', object))
        q_cats_list.append(pad_or_truncate([item[1] for item in query_items], max_item_num, 0, np.int32))
        q_feats_list.append(pad_or_truncate(q_feats_norm, max_item_num, zero_pad, np.float32))
        
        t_ids_list.append(pad_or_truncate([item[0] for item in target_items], max_item_num, '0', object))
        t_cats_list.append(pad_or_truncate([item[1] for item in target_items], max_item_num, 0, np.int32))
        t_feats_list.append(pad_or_truncate(t_feats_norm, max_item_num, zero_pad, np.float32))
        
        scene_ids_list.append(scene_id)
        set_sizes_list.append(len(items_to_process))
    
    if skipped_sets > 0:
        print(f"Skipped sets: {skipped_sets}")
    if overlap_warnings > 0:
        print(f"Overlap resolutions: {overlap_warnings}")
    
    if not q_feats_list:
        print(f"Warning: No data to save: {output_file}")
        return {}
    
    # Create data tuple with normalization info
    normalization_info = {
        'applied_normalization': normalization_stats,
        'split_name': os.path.basename(output_file).replace('.pkl', ''),
        'feature_count': len(q_feats_list) * max_item_num * 2
    }
    
    df_tuple = (
        np.array(q_feats_list, dtype=np.float32),      # 0: query_features
        np.array(t_feats_list, dtype=np.float32),      # 1: target_features  
        np.array(scene_ids_list, dtype=object),        # 2: scene_ids
        np.array(q_cats_list, dtype=np.int32),         # 3: query_categories
        np.array(t_cats_list, dtype=np.int32),         # 4: target_categories
        np.array(set_sizes_list, dtype=np.int32),      # 5: set_sizes
        np.array(q_ids_list, dtype=object),            # 6: query_item_ids
        np.array(t_ids_list, dtype=object),            # 7: target_item_ids
        normalization_info                             # 8: normalization_info
    )
    
    # Save file
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(df_tuple, f)
        print(f"Saved {len(q_feats_list)} sets to {output_file}")
    except Exception as e:
        print(f"Save error {output_file}: {e}")
        return {}
    
    return {'query_categories': df_tuple[3], 'target_categories': df_tuple[4]}

# =============================================================================
# IQON3000 Dataset
# =============================================================================

# IQON3000データセットの読み込みクラス
class IQON3000Dataset(Dataset):
    """IQON3000 dataset with category validation"""
    
    def __init__(self, iqon_dir, processor):
        self.iqon_dir = iqon_dir
        self.processor = processor
        self.item_info = {}  # item_id -> (user_id, coord_id, category, cat_id, filename)
        self.items = []
        self.validation_stats = {
            'total_processed': 0, 'corrections_made': 0,
            'conflicts_detected': 0, 'name_based_inferences': 0
        }
        self._load_data()
        print(f"IQON3000: Loaded {len(self.items)} unique items")
        self._print_validation_stats()
    
    def _load_data(self):
        """Load IQON3000 data with category validation"""
        print(f"Loading IQON3000 data from: {self.iqon_dir}")
        
        if not os.path.isdir(self.iqon_dir):
            print(f"Error: Directory not found: {self.iqon_dir}")
            return

        # Statistics counters
        stats = {
            'json_decode_errors': 0, 'items_missing_id': 0,
            'items_missing_image': 0, 'duplicate_items_skipped': 0,
            'unmapped_to_default': 0
        }
        
        seen_item_ids = set()  # Duplicate removal
        user_dirs = [d for d in os.listdir(self.iqon_dir) 
                    if os.path.isdir(os.path.join(self.iqon_dir, d))]
        
        print(f"Found {len(user_dirs)} user directories")

        for user_id in tqdm(user_dirs, desc="Processing users"):
            user_path = os.path.join(self.iqon_dir, user_id)
            
            try:
                coord_dirs = [d for d in os.listdir(user_path) 
                            if os.path.isdir(os.path.join(user_path, d))]
            except OSError:
                continue

            for coord_id in coord_dirs:
                coord_path = os.path.join(user_path, coord_id)
                json_file = os.path.join(coord_path, f"{coord_id}.json")

                if not os.path.exists(json_file):
                    continue
                
                # Load JSON
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    stats['json_decode_errors'] += 1
                    continue
                
                # Process items
                for item in data.get('items', []):
                    item_id = item.get('itemId')
                    if not item_id:
                        stats['items_missing_id'] += 1
                        continue
                    
                    item_id_str = str(item_id)
                    
                    # Duplicate check
                    if item_id_str in seen_item_ids:
                        stats['duplicate_items_skipped'] += 1
                        continue
                    seen_item_ids.add(item_id_str)

                    # Find image file
                    image_filename, image_path = self._find_image_file(coord_path, item_id_str, item)
                    if not image_path:
                        stats['items_missing_image'] += 1
                        continue

                    # Category validation and correction
                    corrected_category = self._validate_and_correct_category(item)
                    cat_id, was_fallback = self._map_to_category(corrected_category)
                    
                    if was_fallback:
                        stats['unmapped_to_default'] += 1
                    
                    # Save item info
                    self.item_info[item_id_str] = (user_id, coord_id, corrected_category, cat_id, image_filename)
                    self.items.append(item_id_str)
        
        # Print statistics
        print(f"\nIQON3000 loading complete:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print(f"  Valid items: {len(self.items)}")

    def _find_image_file(self, coord_path, item_id_str, item_detail):
        """Find image file for item"""
        # Standard filename
        standard_filename = f"{item_id_str}_m.jpg"
        standard_path = os.path.join(coord_path, standard_filename)
        
        if os.path.exists(standard_path):
            return standard_filename, standard_path
        
        # Try URL-based filename
        img_url = item_detail.get('imgUrl', '')
        if img_url:
            alt_filename = os.path.basename(img_url)
            if alt_filename.endswith(("_m.jpg", ".jpg", ".png", ".jpeg")):
                alt_path = os.path.join(coord_path, alt_filename)
                if os.path.exists(alt_path):
                    return alt_filename, alt_path
        
        return None, None

    def _validate_and_correct_category(self, item_detail):
        """Validate and correct item category"""
        self.validation_stats['total_processed'] += 1
        
        # Get declared category
        cat_field = item_detail.get('category x color', item_detail.get('categoryName', ''))
        if not cat_field and 'category' in item_detail and isinstance(item_detail['category'], dict):
            cat_field = item_detail['category'].get('name', '')
        
        declared_category = cat_field.split(' × ')[0].strip() if ' × ' in cat_field else cat_field.strip()
        
        # Infer category from item name
        item_name = item_detail.get('itemName', item_detail.get('name', '')).lower()
        inferred_category = self._infer_category_from_name(item_name)
        
        if inferred_category:
            self.validation_stats['name_based_inferences'] += 1
        
        # Consistency check and correction
        if inferred_category and declared_category:
            if self._categories_conflict(declared_category, inferred_category):
                self.validation_stats['conflicts_detected'] += 1
                self.validation_stats['corrections_made'] += 1
                return inferred_category
        
        if not declared_category and inferred_category:
            self.validation_stats['corrections_made'] += 1
            return inferred_category
        
        return inferred_category or declared_category or ''

    def _infer_category_from_name(self, item_name):
        """Infer category from item name"""
        if not item_name:
            return None
            
        # Category keywords (sorted by specificity)
        category_keywords = {
            "サングラス": "サングラス", "sunglasses": "サングラス",
            "メガネ": "メガネ", "眼鏡": "メガネ", "glasses": "メガネ",
            "スニーカー": "シューズ", "ナイキ": "シューズ", "nike": "シューズ",
            "アディダス": "シューズ", "adidas": "シューズ", "コンバース": "シューズ",
            "バックパック": "バッグ", "backpack": "バッグ", "リュック": "バッグ",
            "トートバッグ": "バッグ", "ショルダーバッグ": "バッグ",
            "ニット帽": "帽子", "ベレー帽": "帽子", "キャップ": "帽子",
            "ネックレス": "アクセサリー", "ブレスレット": "アクセサリー",
            # General categories
            "tシャツ": "Tシャツ", "t-shirt": "Tシャツ", "カットソー": "Tシャツ",
            "パーカー": "パーカー", "hoodie": "パーカー", "ジャケット": "ジャケット",
            "コート": "コート", "パンツ": "パンツ", "ジーンズ": "パンツ",
            "スカート": "スカート", "ワンピース": "ワンピース", "ドレス": "ワンピース",
        }
        
        # Check keywords by length (longest first)
        for keyword, category in sorted(category_keywords.items(), key=lambda x: len(x[0]), reverse=True):
            if keyword in item_name:
                return category
        
        return None

    def _categories_conflict(self, declared_cat, inferred_cat):
        """Check if categories conflict"""
        if not declared_cat or not inferred_cat:
            return False
        
        # Category families (no conflict within same family)
        families = [
            ["シューズ", "靴", "スニーカー", "ブーツ", "サンダル", "パンプス"],
            ["バッグ", "リュック", "トートバッグ", "ショルダーバッグ"],
            ["帽子", "キャップ", "ハット", "ニット帽", "ベレー帽"],
            ["アクセサリー", "ピアス", "ネックレス", "ブレスレット"],
            ["Tシャツ", "カットソー", "シャツ", "ブラウス", "ニット"],
            ["サングラス", "メガネ", "眼鏡"]
        ]
        
        for family in families:
            if declared_cat in family and inferred_cat in family:
                return False
        
        # Check if different families
        declared_family = inferred_family = None
        for i, family in enumerate(families):
            if declared_cat in family:
                declared_family = i
            if inferred_cat in family:
                inferred_family = i
        
        return (declared_family is not None and inferred_family is not None and 
                declared_family != inferred_family)

    def _map_to_category(self, category_name):
        """Map category name to ID"""
        if not category_name or category_name.strip() == '':
            return DEFAULT_IQON3000_CATEGORY, True
        
        if category_name in IQON3000_MAPPING:
            return IQON3000_MAPPING[category_name], False
        
        # Partial matching
        for key in sorted(IQON3000_MAPPING.keys(), key=len, reverse=True):
            if key in category_name:
                return IQON3000_MAPPING[key], False
        
        return DEFAULT_IQON3000_CATEGORY, True

    def _print_validation_stats(self):
        """Print validation statistics"""
        stats = self.validation_stats
        print(f"\nCategory validation stats:")
        print(f"  Processed items: {stats['total_processed']}")
        print(f"  Name inferences: {stats['name_based_inferences']}")
        print(f"  Conflicts detected: {stats['conflicts_detected']}")
        print(f"  Corrections applied: {stats['corrections_made']}")
        if stats['total_processed'] > 0:
            print(f"  Correction rate: {stats['corrections_made']/stats['total_processed']*100:.1f}%")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item_id = self.items[idx]
        user_id, coord_id, _, cat_id, filename = self.item_info[item_id]
        img_path = os.path.join(self.iqon_dir, user_id, coord_id, filename)
        
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            return {key: val.squeeze(0) for key, val in inputs.items()}, item_id, cat_id
        except (UnidentifiedImageError, FileNotFoundError, OSError):
            # Dummy image on error
            dummy_image = Image.new("RGB", (224, 224), color="gray")
            dummy_input = self.processor(images=dummy_image, return_tensors="pt")
            return {key: val.squeeze(0) for key, val in dummy_input.items()}, item_id, DEFAULT_IQON3000_CATEGORY

# =============================================================================
# IQON3000 Processing
# =============================================================================

# IQON3000の全処理を統括
def process_iqon3000(input_dir, output_dir, batch_size=32):
    """Process IQON3000 dataset with user-based splitting"""
    print("Processing IQON3000 dataset")
    print("Improvements: User-based splitting + unified normalization")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup CLIP model
    model, processor, device = setup_clip_model()
    
    # Create dataset
    dataset = IQON3000Dataset(input_dir, processor)
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                           collate_fn=collate_fn, num_workers=4, 
                           pin_memory=device.type == 'cuda')
    
    # Extract features
    item_features, item_categories, coordinate_mapping = extract_clip_features(
        dataloader, model, device, dataset
    )
    
    # Build valid coordinate sets
    valid_coordinates = build_valid_coordinates(
        coordinate_mapping, item_categories, dataset.item_info, min_items=4
    )
    
    if not valid_coordinates:
        print("No valid coordinate sets found.")
        return
    
    # Prepare set data
    set_data = prepare_set_data(valid_coordinates, item_features, item_categories)
    
    # User-based splitting
    train_sets, val_sets, test_sets = split_data_by_users(set_data, test_size=0.2, val_size=0.1)
    
    # Compute normalization stats
    norm_stats = compute_normalization_stats(train_sets)
    
    # Convert and save each split
    for sets, split_name in [(train_sets, 'train'), (val_sets, 'validation'), (test_sets, 'test')]:
        if sets:
            convert_to_deepfurniture_format(
                sets, 
                os.path.join(output_dir, f'{split_name}.pkl'),
                norm_stats
            )
    
    # Save metadata
    save_iqon3000_metadata(output_dir, item_categories)
    
    # Compute category centers
    compute_category_centers(output_dir, IQON3000_CATEGORIES, list(range(1, 18)))
    
    print(f"IQON3000 processing complete: {output_dir}")

# CLIP特徴量の抽出
def extract_clip_features(dataloader, model, device, dataset):
    """Extract CLIP features"""
    item_features = {}
    item_categories = {}
    coordinate_mapping = {}
    
    print(f"Extracting CLIP features: {len(dataset)} items")
    
    with torch.no_grad():
        for batch_inputs, batch_item_ids, batch_cat_ids in tqdm(dataloader, desc="Extracting features"):
            if not batch_item_ids or not batch_inputs or 'pixel_values' not in batch_inputs:
                continue
            
            if batch_inputs['pixel_values'].numel() == 0:
                continue
            
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            
            try:
                img_feats = model.get_image_features(**batch_inputs)
                img_feats_np = img_feats.cpu().numpy()  # Raw features
                
                for i, item_id in enumerate(batch_item_ids):
                    if item_id is None:
                        continue
                    
                    item_features[item_id] = img_feats_np[i]
                    item_categories[item_id] = batch_cat_ids[i]
                    
                    # Get coordinate info
                    if item_id in dataset.item_info:
                        _, coord_id, _, _, _ = dataset.item_info[item_id]
                        coordinate_mapping.setdefault(coord_id, []).append(item_id)
                        
            except Exception as e:
                print(f"Feature extraction error: {e}")
                continue
    
    print(f"Feature extraction complete: {len(item_features)} items")
    return item_features, item_categories, coordinate_mapping

# 有効なコーディネートセットの構築
def build_valid_coordinates(coordinate_mapping, item_categories, item_info, min_items=4):
    """Build valid coordinate sets"""
    valid_coordinates = {}
    
    for coord_id, item_ids in coordinate_mapping.items():
        # Filter valid items
        valid_items = [
            item_id for item_id in item_ids 
            if item_id in item_categories and 1 <= item_categories[item_id] <= 17
        ]
        
        if len(valid_items) >= min_items:
            # Get user ID
            if valid_items and valid_items[0] in item_info:
                user_id, _, _, _, _ = item_info[valid_items[0]]
                path_key = (user_id, coord_id)
                valid_coordinates[path_key] = valid_items
    
    print(f"Valid coordinate sets: {len(valid_coordinates)}")
    return valid_coordinates

# セットデータの準備
def prepare_set_data(valid_coordinates, item_features, item_categories):
    """Prepare set data"""
    set_data = []
    
    for (user_id, coord_id), item_ids in valid_coordinates.items():
        items_data = []
        for item_id in item_ids:
            if item_id in item_features and item_id in item_categories:
                items_data.append((
                    item_id,
                    item_categories[item_id],
                    item_features[item_id]
                ))
        
        if len(items_data) >= 2:
            set_data.append(((user_id, coord_id), items_data))
    
    print(f"Prepared set data: {len(set_data)} sets")
    return set_data

# メタデータの保存
def save_iqon3000_metadata(output_dir, item_categories):
    """Save IQON3000 metadata"""
    category_stats = {}
    for cat_id in item_categories.values():
        category_stats[cat_id] = category_stats.get(cat_id, 0) + 1
    
    category_info = {
        'main_categories_def': IQON3000_CATEGORIES,
        'main_category_stats': category_stats
    }
    
    # Save as pickle
    with open(os.path.join(output_dir, 'category_info.pkl'), 'wb') as f:
        pickle.dump(category_info, f)
    
    # Save as JSON
    json_safe = {
        'main_categories_def': category_info['main_categories_def'],
        'main_category_stats': {str(k): v for k, v in category_stats.items()}
    }
    with open(os.path.join(output_dir, 'category_info.json'), 'w', encoding='utf-8') as f:
        json.dump(json_safe, f, ensure_ascii=False, indent=2)

# =============================================================================
# DeepFurniture Processing
# =============================================================================

# DeepFurniture全処理を統括
def process_deepfurniture(image_dir, annotations_json, furnitures_jsonl, 
                                                output_dir, batch_size=32, apply_inclusion_removal=True):
    """Process DeepFurniture with inclusion relationship removal"""
    print("Processing DeepFurniture dataset")
    print(f"Inclusion removal: {'enabled' if apply_inclusion_removal else 'disabled'}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    furniture_to_category = load_jsonl_mapping(Path(furnitures_jsonl), "furniture_id", "category_id")
    
    # Build scenes
    raw_scenes = build_deepfurniture_scenes(annotations_json, furniture_to_category)
    
    # Apply inclusion removal if enabled
    if apply_inclusion_removal:
        remover = InclusionRelationshipRemover(debug_mode=True)
        scenes_to_process = remover.remove_inclusion_relationships(raw_scenes, min_items=4)
        print(f"Inclusion removal: {len(raw_scenes)} -> {len(scenes_to_process)} scenes")
    else:
        scenes_to_process = raw_scenes
    
    # Extract features
    final_scenes = extract_deepfurniture_features(
        scenes_to_process, image_dir, furniture_to_category, batch_size
    )
    
    # Convert to format and save
    convert_deepfurniture_scenes_to_format(final_scenes, output_dir)
    
    # Compute category centers
    compute_category_centers(output_dir, DEEPFURNITURE_CATEGORIES, list(range(1, 12)))
    
    # Save metadata
    with open(os.path.join(output_dir, 'category_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump({'categories': DEEPFURNITURE_CATEGORIES}, f, ensure_ascii=False, indent=2)
    
    print(f"DeepFurniture processing complete: {output_dir}")

# シーンデータの構築
def build_deepfurniture_scenes(annotations_json, furniture_to_category):
    """Build DeepFurniture scenes"""
    print("Building DeepFurniture scenes...")
    
    with open(annotations_json, 'r') as f:
        annotations = json.load(f)
    
    raw_scenes = {}
    category_ids_found = set()
    
    for scene_record in tqdm(annotations, desc="Building scenes"):
        scene_id = scene_record.get("scene", {}).get("sceneTaskID")
        if not scene_id:
            continue
        
        # Collect unique items
        scene_items_unique = {}
        
        for instance in scene_record.get("instances", []):
            furniture_id = str(instance.get("identityID"))
            category_id = instance.get("categoryID")
            
            if (furniture_id and category_id is not None and 
                furniture_id not in scene_items_unique and 
                furniture_id in furniture_to_category):
                scene_items_unique[furniture_id] = category_id
        
        # Check minimum items
        if len(scene_items_unique) >= 4:
            raw_scenes[str(scene_id)] = [(fid, cid) for fid, cid in scene_items_unique.items()]
            category_ids_found.update(scene_items_unique.values())
    
    print(f"Built scenes: {len(raw_scenes)}")
    print(f"Found category IDs: {sorted(list(category_ids_found))}")
    
    return raw_scenes

# DeepFurniture特徴量抽出
def extract_deepfurniture_features(scenes_to_process, image_dir, furniture_to_category, batch_size):
    """Extract DeepFurniture features"""
    print("Extracting DeepFurniture features...")
    
    # Setup CLIP model
    model, processor, device = setup_clip_model()
    
    # Find used furniture IDs
    used_furniture_ids = set()
    for scene_items in scenes_to_process.values():
        for furniture_id, _ in scene_items:
            used_furniture_ids.add(furniture_id)
    
    # Find image files
    image_files = []
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        image_files.extend(Path(image_dir).rglob(ext))
    
    relevant_images = [img for img in image_files if img.stem in used_furniture_ids]
    print(f"Processing images: {len(relevant_images)}")
    
    # Extract features
    all_features = []
    all_furniture_ids = []
    processed_ids = set()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(relevant_images), batch_size), desc="Feature extraction"):
            batch_paths = relevant_images[i:i+batch_size]
            
            # Remove duplicates
            filtered_paths = []
            for path in batch_paths:
                if path.stem not in processed_ids:
                    processed_ids.add(path.stem)
                    filtered_paths.append(path)
            
            if not filtered_paths:
                continue
            
            try:
                pil_images = [Image.open(p).convert("RGB") for p in filtered_paths]
                inputs = processor(images=pil_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                output = model.get_image_features(**inputs)
                output_normalized = normalize_features_l2(output)
                
                all_features.append(output_normalized.cpu().numpy())
                all_furniture_ids.extend([p.stem for p in filtered_paths])
                
            except Exception as e:
                print(f"Batch processing error: {e}")
                continue
    
    if not all_features:
        print("Feature extraction failed")
        return {}
    
    all_features = np.vstack(all_features)
    print(f"Extraction complete: {all_features.shape}")
    
    # Rebuild scenes with features
    final_scenes = {}
    for scene_id, scene_items in tqdm(scenes_to_process.items(), desc="Scene integration"):
        scene_features, scene_categories, scene_furniture_ids = [], [], []
        
        for furniture_id, category_id in scene_items:
            if furniture_id in all_furniture_ids:
                idx = all_furniture_ids.index(furniture_id)
                scene_features.append(all_features[idx])
                scene_categories.append(category_id)
                scene_furniture_ids.append(furniture_id)
        
        if len(scene_features) >= 4:
            final_scenes[scene_id] = {
                "features": np.array(scene_features),
                "category_ids": np.array(scene_categories),
                "item_ids": np.array(scene_furniture_ids, dtype=object)
            }
    
    print(f"Final scenes: {len(final_scenes)}")
    return final_scenes

# JSONL形式のマッピングファイル読み込み
def load_jsonl_mapping(path, key_field, value_field):
    """Load JSONL file mapping"""
    mapping = {}
    unique_values_found = set()
    
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                try:
                    rec = json.loads(line)
                    key = str(rec[key_field])
                    value = int(rec[value_field])
                    mapping[key] = value
                    unique_values_found.add(value)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"Warning: Skipping line {line_num+1} due to error: {e}")
        
        if mapping:
            print(f"Loaded mapping: {len(mapping)} entries")
            print(f"Found {len(unique_values_found)} unique category IDs: {sorted(list(unique_values_found))}")
        else:
            print("Warning: No valid entries loaded")
    except FileNotFoundError:
        print(f"Error: File not found: {path}")
    except Exception as e:
        print(f"Error loading file: {e}")
    
    return mapping

# シーンを標準形式に変換
def convert_deepfurniture_scenes_to_format(scenes, output_dir, max_item_num=10):
    """Convert DeepFurniture scenes to standard format"""
    print("Converting DeepFurniture scenes to format...")
    
    # Initialize data lists
    q_feats, p_feats, q_cats, p_cats = [], [], [], []
    q_ids, p_ids, s_keys = [], [], []
    
    rng = random.Random(42)
    scene_count = 0
    skipped_count = 0
    
    for sid, rec in tqdm(scenes.items(), desc="Converting scenes"):
        feats, cats, ids = rec["features"], rec["category_ids"], rec["item_ids"]
        n = len(feats)
        
        if n < 4 or n > 20:
            skipped_count += 1
            continue
        
        # Shuffle and split
        idx = list(range(n))
        rng.shuffle(idx)
        
        half = n // 2
        q_idx, p_idx = idx[:half], idx[half:]
        
        # Check for overlaps
        q_items = set(ids[q_idx])
        p_items = set(ids[p_idx])
        if q_items & p_items:
            skipped_count += 1
            continue
        
        # Pad data
        def pad_local(arr, target, pad_val=0):
            if len(arr) < target:
                if arr.ndim > 1:
                    pad_shape = (target - len(arr), *arr.shape[1:])
                else:
                    pad_shape = (target - len(arr),)
                pad_arr = np.full(pad_shape, pad_val, dtype=arr.dtype)
                return np.concatenate([arr, pad_arr], axis=0)
            return arr[:target]
        
        # Add data
        q_feats.append(pad_local(feats[q_idx], max_item_num))
        q_ids.append(pad_local(ids[q_idx], max_item_num, pad_val=""))
        q_cats.append(pad_local(cats[q_idx], max_item_num, pad_val=0))
        
        p_feats.append(pad_local(feats[p_idx], max_item_num))
        p_ids.append(pad_local(ids[p_idx], max_item_num, pad_val=""))
        p_cats.append(pad_local(cats[p_idx], max_item_num, pad_val=0))
        
        s_keys.append(sid)
        scene_count += 1
    
    print(f"Processed {scene_count} scenes (skipped {skipped_count})")
    
    if not q_feats:
        print("No valid scenes found")
        return
    
    # Convert to numpy arrays
    Q = np.stack(q_feats)
    P = np.stack(p_feats)
    qcat = np.stack(q_cats)
    pcat = np.stack(p_cats)
    qid = np.stack(q_ids)
    pid = np.stack(p_ids)
    sids = np.array(s_keys)
    
    # Remap categories (padding=0)
    unique = np.unique(np.concatenate([qcat, pcat]))
    unique = unique[unique > 0]
    cat_map = {cid: i + 1 for i, cid in enumerate(sorted(unique))}
    vect = np.vectorize(lambda c: cat_map.get(c, 0))
    qcat = vect(qcat)
    pcat = vect(pcat)
    
    print(f"Found {len(unique)} unique categories, remapped to 1-{len(unique)}")
    
    # Scene-based train/val/test split
    indices = np.arange(len(Q))
    
    # 70/15/15 split
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    # Create split data
    Q_tr, P_tr = Q[train_idx], P[train_idx]
    qcat_tr, pcat_tr = qcat[train_idx], pcat[train_idx]
    qid_tr, pid_tr = qid[train_idx], pid[train_idx]
    sid_tr = sids[train_idx]
    
    Q_val, P_val = Q[val_idx], P[val_idx]
    qcat_val, pcat_val = qcat[val_idx], pcat[val_idx]
    qid_val, pid_val = qid[val_idx], pid[val_idx]
    sid_val = sids[val_idx]
    
    Q_te, P_te = Q[test_idx], P[test_idx]
    qcat_te, pcat_te = qcat[test_idx], pcat[test_idx]
    qid_te, pid_te = qid[test_idx], pid[test_idx]
    sid_te = sids[test_idx]
    
    # Check for query-positive overlaps
    def check_overlap(split_name, q_ids, p_ids):
        """Check for query-positive overlaps"""
        overlap_count = 0
        for i in range(len(q_ids)):
            q_items = set([item for item in q_ids[i] if item and item != "" and item != "0"])
            p_items = set([item for item in p_ids[i] if item and item != "" and item != "0"])
            if q_items & p_items:
                overlap_count += 1
        
        if overlap_count > 0:
            print(f"Warning: {split_name}: {overlap_count}/{len(q_ids)} scenes have query-positive overlap")
        else:
            print(f"Success: {split_name}: No query-positive overlap")
        
        return overlap_count == 0
    
    print("\nQuery-Positive Overlap Check:")
    check_overlap("Train", qid_tr, pid_tr)
    check_overlap("Validation", qid_val, pid_val)
    check_overlap("Test", qid_te, pid_te)
    
    print(f"Features are already L2-normalized")
    print(f"DeepFurniture split: Train {len(Q_tr)}, Validation {len(Q_val)}, Test {len(Q_te)}")
    
    # Save splits
    def save_split(path, objects):
        with open(path, "wb") as f:
            pickle.dump(objects, f)
    
    # Save with original format (set_sizes initialized to zeros)
    save_split(os.path.join(output_dir, "train.pkl"), 
              (Q_tr, P_tr, sid_tr, qcat_tr, pcat_tr, np.zeros(len(Q_tr)), qid_tr, pid_tr))
    save_split(os.path.join(output_dir, "validation.pkl"), 
              (Q_val, P_val, sid_val, qcat_val, pcat_val, np.zeros(len(Q_val)), qid_val, pid_val))
    save_split(os.path.join(output_dir, "test.pkl"), 
              (Q_te, P_te, sid_te, qcat_te, pcat_te, np.zeros(len(Q_te)), qid_te, pid_te))
    
    print("Dataset saved successfully")

# =============================================================================
# Inclusion Relationship Removal
# =============================================================================

# 包含関係（AがBの部分集合）を検出・除去するクラス
class InclusionRelationshipRemover:
    """Remove inclusion relationships from scenes"""
    
    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode
        
    def remove_inclusion_relationships(self, scenes_dict, min_items=4, debug_limit=10):
        """Remove inclusion relationships from scenes
        
        Args:
            scenes_dict: scene_id -> item_list mapping
            min_items: minimum items per scene
            debug_limit: number of examples to show in debug
            
        Returns:
            filtered_scenes: scenes with inclusion relationships removed
        """
        
        if self.debug_mode:
            print(f"Starting inclusion relationship removal")
            print(f"Input scenes: {len(scenes_dict)}")
        
        # Convert scenes to furniture ID sets
        scene_to_furniture_sets = {}
        
        for scene_id, scene_data in scenes_dict.items():
            furniture_ids = self._extract_furniture_ids(scene_data)
            
            # Check minimum items
            if len(furniture_ids) >= min_items:
                scene_to_furniture_sets[scene_id] = set(furniture_ids)
        
        if self.debug_mode:
            print(f"Scenes with min {min_items} items: {len(scene_to_furniture_sets)}")
        
        # Detect inclusion relationships
        inclusion_pairs = self._detect_inclusion_pairs(scene_to_furniture_sets, debug_limit)
        
        if not inclusion_pairs:
            if self.debug_mode:
                print("No inclusion relationships detected")
            return scenes_dict
        
        # Determine removal targets
        scenes_to_remove = self._determine_removal_targets(inclusion_pairs, scene_to_furniture_sets)
        
        # Apply removal
        filtered_scenes = self._apply_removal(scenes_dict, scenes_to_remove)
        
        # Validate removal
        if self.debug_mode:
            self._validate_removal(filtered_scenes, min_items)
        
        return filtered_scenes
    
    def _extract_furniture_ids(self, scene_data):
        """Extract furniture IDs from scene data"""
        
        furniture_ids = []
        
        if isinstance(scene_data, dict):
            # Dictionary format
            if 'item_ids' in scene_data:
                furniture_ids = list(scene_data['item_ids'])
            else:
                # Search other dictionary keys
                for key, value in scene_data.items():
                    if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                        if isinstance(value[0], str):
                            furniture_ids = list(value)
                            break
        
        elif isinstance(scene_data, list):
            # List format
            for item in scene_data:
                if isinstance(item, tuple) and len(item) >= 1:
                    furniture_ids.append(str(item[0]))
                elif isinstance(item, str):
                    furniture_ids.append(item)
                elif isinstance(item, dict) and 'furniture_id' in item:
                    furniture_ids.append(str(item['furniture_id']))
        
        # Remove duplicates and ensure strings
        return list(set([str(fid) for fid in furniture_ids if fid]))
    
    def _detect_inclusion_pairs(self, scene_to_furniture_sets, debug_limit):
        """Detect inclusion relationship pairs"""
        
        inclusion_pairs = []
        scene_ids = list(scene_to_furniture_sets.keys())
        
        if self.debug_mode:
            print(f"Detecting inclusion relationships... ({len(scene_ids)} scenes)")
        
        debug_count = 0
        
        # Check all pairs
        for i, scene_id1 in enumerate(tqdm(scene_ids, desc="Checking inclusion", disable=not self.debug_mode)):
            set1 = scene_to_furniture_sets[scene_id1]
            
            for scene_id2 in scene_ids[i+1:]:
                set2 = scene_to_furniture_sets[scene_id2]
                
                # Check inclusion relationships
                if set1.issubset(set2) and set1 != set2:
                    inclusion_pairs.append((scene_id1, scene_id2, 'subset'))
                    
                    if self.debug_mode and debug_count < debug_limit:
                        print(f"\nInclusion found: {scene_id1} ⊆ {scene_id2}")
                        print(f"  {scene_id1}({len(set1)}): {sorted(list(set1))[:5]}...")
                        print(f"  {scene_id2}({len(set2)}): {sorted(list(set2))[:5]}...")
                        print(f"  Difference({len(set2-set1)}): {sorted(list(set2-set1))[:3]}...")
                        debug_count += 1
                        
                elif set2.issubset(set1) and set1 != set2:
                    inclusion_pairs.append((scene_id2, scene_id1, 'subset'))
                    
                    if self.debug_mode and debug_count < debug_limit:
                        print(f"\nInclusion found: {scene_id2} ⊆ {scene_id1}")
                        print(f"  {scene_id2}({len(set2)}): {sorted(list(set2))[:5]}...")
                        print(f"  {scene_id1}({len(set1)}): {sorted(list(set1))[:5]}...")
                        print(f"  Difference({len(set1-set2)}): {sorted(list(set1-set2))[:3]}...")
                        debug_count += 1
        
        if self.debug_mode:
            print(f"Inclusion detection result: {len(inclusion_pairs)} pairs")
        
        return inclusion_pairs
    
    def _determine_removal_targets(self, inclusion_pairs, scene_to_furniture_sets):
        """Determine which scenes to remove"""
        
        # Strategy: remove smaller scenes (subsets)
        scenes_to_remove = set()
        
        for smaller_scene, larger_scene, relation_type in inclusion_pairs:
            scenes_to_remove.add(smaller_scene)
        
        if self.debug_mode:
            print(f"Removal strategy: remove smaller scenes (subsets)")
            print(f"Scenes to remove: {len(scenes_to_remove)}")
            
            # Show removal examples
            if scenes_to_remove:
                print(f"Removal examples:")
                for i, scene_id in enumerate(list(scenes_to_remove)[:5]):
                    furniture_set = scene_to_furniture_sets[scene_id]
                    print(f"  {scene_id}: {len(furniture_set)} items")
        
        return scenes_to_remove
    
    def _apply_removal(self, scenes_dict, scenes_to_remove):
        """Apply removal"""
        
        original_count = len(scenes_dict)
        
        filtered_scenes = {
            scene_id: scene_data 
            for scene_id, scene_data in scenes_dict.items() 
            if scene_id not in scenes_to_remove
        }
        
        if self.debug_mode:
            removed_count = original_count - len(filtered_scenes)
            print(f"Removal complete")
            print(f"  Original scenes: {original_count}")
            print(f"  Removed scenes: {removed_count}")
            print(f"  Remaining scenes: {len(filtered_scenes)}")
            print(f"  Removal rate: {removed_count/original_count*100:.1f}%")
        
        return filtered_scenes
    
    def _validate_removal(self, filtered_scenes, min_items):
        """Validate removal results"""
        
        print(f"Validating removal...")
        
        # Re-check inclusion relationships
        verification_remover = InclusionRelationshipRemover(debug_mode=False)
        scene_to_furniture_sets = {}
        
        for scene_id, scene_data in filtered_scenes.items():
            furniture_ids = verification_remover._extract_furniture_ids(scene_data)
            if len(furniture_ids) >= min_items:
                scene_to_furniture_sets[scene_id] = set(furniture_ids)
        
        verification_pairs = verification_remover._detect_inclusion_pairs(scene_to_furniture_sets, debug_limit=3)
        
        if len(verification_pairs) == 0:
            print("Inclusion relationships completely removed")
        else:
            print(f"Warning: {len(verification_pairs)} inclusion relationships still remain")
            # Show remaining examples
            for i, (smaller, larger, _) in enumerate(verification_pairs[:3]):
                set1 = scene_to_furniture_sets[smaller]
                set2 = scene_to_furniture_sets[larger]
                print(f"  Remaining {i+1}: {smaller}({len(set1)}) ⊆ {larger}({len(set2)})")

# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified dataset processing with inclusion removal")
    parser.add_argument('--dataset', choices=['iqon3000', 'deepfurniture'], required=True, help='Dataset type to process')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for feature extraction')
    parser.add_argument('--input-dir', type=str, help='Input directory for IQON3000 dataset')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for processed dataset')
    parser.add_argument('--user-based', action='store_true', help='Use user-based splitting (recommended)')
    
    # DeepFurniture arguments
    parser.add_argument('--image-dir', type=str, help='DeepFurniture image directory')
    parser.add_argument('--annotations-json', type=str, help='DeepFurniture annotations.json file')
    parser.add_argument('--furnitures-jsonl', type=str, help='DeepFurniture furnitures.jsonl file')
    parser.add_argument('--no-inclusion-removal', action='store_true', help='Disable inclusion relationship removal for DeepFurniture')
    
    args = parser.parse_args()
    
    if args.dataset == 'iqon3000':
        if not args.input_dir:
            parser.error("--input-dir is required for IQON3000 dataset")
        
        if args.user_based:
            print("Processing IQON3000 dataset: 17 categories (USER-BASED SPLIT)")
            print("Key improvements:")
            print("- USER-BASED splitting: No user overlap between splits")
            print("- Split FIRST, then normalize")
            print("- Use training data statistics only")
            print("- Prevent information leakage")
            process_iqon3000(args.input_dir, args.output_dir, args.batch_size)
        else:
            print("Processing IQON3000 dataset: 17 categories (SCENE-BASED SPLIT)")
            print("Warning: This may cause user overlap between splits")
            process_iqon3000(args.input_dir, args.output_dir, args.batch_size)

    elif args.dataset == 'deepfurniture':
        if not all([args.image_dir, args.annotations_json, args.furnitures_jsonl]):
            parser.error("--image-dir, --annotations-json, and --furnitures-jsonl are required for DeepFurniture dataset")
        
        apply_inclusion_removal = not args.no_inclusion_removal
        
        if apply_inclusion_removal:
            print("Processing DeepFurniture dataset: WITH inclusion relationship removal")
        else:
            print("Processing DeepFurniture dataset: WITHOUT inclusion relationship removal")
        
        process_deepfurniture(
            image_dir=args.image_dir,
            annotations_json=args.annotations_json,
            furnitures_jsonl=args.furnitures_jsonl,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            apply_inclusion_removal=apply_inclusion_removal
        )


if __name__ == "__main__":
    main()