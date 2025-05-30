#!/usr/bin/env python3
"""
make_datasets.py
================
Unified dataset generator for both IQON3000 and DeepFurniture datasets.
Creates train.pkl, validation.pkl, test.pkl, and category_centers.pkl.gz files.

Usage:
------
# For IQON3000 dataset (7 categories)
python make_datasets.py --dataset iqon3000 --input-dir data/IQON3000 --output-dir datasets/IQON3000

# For DeepFurniture dataset (11 categories)
python make_datasets.py --dataset deepfurniture --image-dir data/DeepFurniture/furnitures --annotations-json data/DeepFurniture/metadata/annotations.json --furnitures-jsonl data/DeepFurniture/metadata/furnitures.jsonl --output-dir datasets/DeepFurniture
"""

import os
import json
import pickle
import gzip
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

# Suppress TensorFlow warnings
tf.get_logger().setLevel("ERROR")

# =============================================================================
# IQON3000 Dataset Configuration (7 categories)
# =============================================================================

IQON3000_CATEGORIES = {
    1: "outerwear", 2: "tops", 3: "bottoms", 4: "shoes",
    5: "bags", 6: "hats", 7: "accessories"
}

IQON3000_CATEGORY_MAPPING = {
    "ジャケット": 1, "コート": 1, "アウター": 1, "カーディガン": 1, "ブルゾン": 1, "ダウン": 1, "パーカー": 1,
    "Tシャツ": 2, "カットソー": 2, "シャツ": 2, "ブラウス": 2, "ニット": 2, "セーター": 2,
    "ベスト": 2, "タンクトップ": 2, "キャミソール": 2, "チュニック": 2, "トップス": 2, "インナー": 2,
    "ワンピース": 2, "ドレス": 2, "ルームウェア": 2, "水着": 2, "浴衣": 2, "着物": 2,
    "パンツ": 3, "スカート": 3, "ショートパンツ": 3, "ロングパンツ": 3, "ロングスカート": 3,
    "ジーンズ": 3, "デニム": 3, "レギンス": 3, "スラックス": 3,
    "シューズ": 4, "靴": 4, "スニーカー": 4, "サンダル": 4, "ブーツ": 4, "パンプス": 4,
    "ルームシューズ": 4, "ローファー": 4,
    "バッグ": 5, "トートバッグ": 5, "ショルダーバッグ": 5, "ハンドバッグ": 5, "クラッチバッグ": 5,
    "ボストンバッグ": 5, "リュック": 5, "ポーチ": 5,
    "帽子": 6, "ハット": 6, "キャップ": 6, "ニット帽": 6, "ベレー帽": 6, "キャスケット": 6,
    "アクセサリー": 7, "ジュエリー": 7, "ネックレス": 7, "ブレスレット": 7, "イヤリング": 7,
    "リング": 7, "ヘアアクセサリー": 7, "ベルト": 7, "スカーフ": 7, "ストール": 7,
    "マフラー": 7, "手袋": 7, "サングラス": 7, "メガネ": 7, "ファッション雑貨": 7,
    "ソックス・靴下": 7, "タイツ・ストッキング": 7, "傘・日傘": 7, "時計": 7, "財布": 7,
    "キーケース・キーホルダー": 7, "ピアス": 7, "コスメ": 7, "レッグウェア": 7, "傘": 7,
    "インテリア": 7, "ネイル": 7, "フレグランス": 7, "小物": 7, "アンダーウェア": 7,
    "ボディケア": 7, "ファッション小物": 7, "ブローチ": 7, "ステーショナリー": 7,
    "その他": 7,
}

DEFAULT_IQON3000_CATEGORY = 7

# =============================================================================
# DeepFurniture Dataset Configuration (11 categories)
# =============================================================================

DEEPFURNITURE_CATEGORIES = {
    1: "chair", 2: "table", 3: "sofa", 4: "bed", 5: "cabinet", 
    6: "lamp", 7: "bookshelf", 8: "desk", 9: "dresser", 
    10: "nightstand", 11: "other_furniture"
}

# =============================================================================
# IQON3000 Dataset Class
# =============================================================================

class IQON3000Dataset(Dataset):
    def __init__(self, iqon_dir, processor):
        self.iqon_dir = iqon_dir
        self.processor = processor
        self.item_info = {}  # item_id -> (user_id, coordinate_id, category, cat_id, filename)
        self.items = []
        self._load_data()
        print(f"IQON3000: Loaded {len(self.items)} items from {len(self.item_info)} unique entries")
    
    def _map_to_main_category_strict(self, japanese_category_name):
        if not japanese_category_name or japanese_category_name.strip() == '':
            return DEFAULT_IQON3000_CATEGORY, True
        if japanese_category_name in IQON3000_CATEGORY_MAPPING:
            return IQON3000_CATEGORY_MAPPING[japanese_category_name], False
        for key in sorted(IQON3000_CATEGORY_MAPPING.keys(), key=len, reverse=True):
            if key in japanese_category_name:
                return IQON3000_CATEGORY_MAPPING[key], False
        if japanese_category_name.strip().isdigit():
            return DEFAULT_IQON3000_CATEGORY, True
        return DEFAULT_IQON3000_CATEGORY, True
    
    def _load_data(self):
        print(f"Starting IQON3000 data loading from: {self.iqon_dir}")
        if not os.path.isdir(self.iqon_dir):
            print(f"Error: IQON3000 directory not found: {self.iqon_dir}")
            return

        source_category_counts = {}
        unmapped_to_default_count = 0
        json_decode_errors = 0
        items_missing_id_in_json = 0
        items_missing_image = 0
        
        user_id_dirs = [d for d in os.listdir(self.iqon_dir) if os.path.isdir(os.path.join(self.iqon_dir, d))]
        print(f"Found {len(user_id_dirs)} user directories.")

        for user_id_str in tqdm(user_id_dirs, desc="Processing IQON3000 Users"):
            current_user_path = os.path.join(self.iqon_dir, user_id_str)
            try:
                coordinate_id_dirs = [d for d in os.listdir(current_user_path) if os.path.isdir(os.path.join(current_user_path, d))]
            except OSError:
                continue

            for coordinate_id_str in coordinate_id_dirs:
                current_coordinate_path = os.path.join(current_user_path, coordinate_id_str)
                coordinate_json_path = os.path.join(current_coordinate_path, f"{coordinate_id_str}.json")

                if not os.path.exists(coordinate_json_path):
                    continue
                
                try:
                    with open(coordinate_json_path, 'r', encoding='utf-8') as f:
                        outfit_data = json.load(f)
                except json.JSONDecodeError:
                    json_decode_errors += 1
                    continue
                
                items_list = outfit_data.get('items', [])
                for item_detail in items_list:
                    true_item_id = item_detail.get('itemId')
                    if not true_item_id:
                        items_missing_id_in_json += 1
                        continue
                    true_item_id_str = str(true_item_id)

                    # Determine image filename
                    image_filename = f"{true_item_id_str}_m.jpg"
                    full_image_path = os.path.join(current_coordinate_path, image_filename)
                    
                    if not os.path.exists(full_image_path):
                        img_url = item_detail.get('imgUrl', '')
                        if img_url:
                            potential_filename = os.path.basename(img_url)
                            if potential_filename.endswith(("_m.jpg", ".jpg", ".png", ".jpeg")):
                                alt_path = os.path.join(current_coordinate_path, potential_filename)
                                if os.path.exists(alt_path):
                                    image_filename = potential_filename
                                    full_image_path = alt_path
                                else:
                                    items_missing_image += 1; continue
                            else:
                                items_missing_image += 1; continue
                        else:
                            items_missing_image += 1; continue

                    cat_field = item_detail.get('category x color', item_detail.get('categoryName', ''))
                    if not cat_field and 'category' in item_detail and isinstance(item_detail['category'], dict):
                        cat_field = item_detail['category'].get('name', '')
                    if not cat_field:
                        cat_field = item_detail.get('itemName', item_detail.get('name', ''))
                    
                    jp_cat_name = cat_field.split(' × ')[0].strip() if ' × ' in cat_field else cat_field.strip()
                    
                    if jp_cat_name:
                        source_category_counts[jp_cat_name] = source_category_counts.get(jp_cat_name, 0) + 1
                    
                    main_cat_id, was_fallback = self._map_to_main_category_strict(jp_cat_name)
                    if was_fallback:
                        unmapped_to_default_count += 1
                    
                    self.item_info[true_item_id_str] = (user_id_str, coordinate_id_str, jp_cat_name, main_cat_id, image_filename)
                    if true_item_id_str not in self.items:
                        self.items.append(true_item_id_str)
        
        print(f"\nIQON3000 Data Loading Summary:")
        print(f"  Total items processed: {sum(source_category_counts.values()) + unmapped_to_default_count}")
        print(f"  Items skipped (missing 'itemId'): {items_missing_id_in_json}")
        print(f"  Items skipped (missing image): {items_missing_image}")
        print(f"  JSON decode errors: {json_decode_errors}")
        print(f"  Unique item entries: {len(self.item_info)}")
        print(f"  Valid items: {len(self.items)}")
        print(f"  Items mapped to default category: {unmapped_to_default_count}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        true_item_id_str = self.items[idx]
        user_id_str, coordinate_id_str, _, main_cat_id, image_filename = self.item_info[true_item_id_str]
        img_path = os.path.join(self.iqon_dir, user_id_str, coordinate_id_str, image_filename)
        
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            return {key: val.squeeze(0) for key, val in inputs.items()}, true_item_id_str, main_cat_id
        except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
            dummy_size = (224, 224)
            dummy_image = Image.new("RGB", dummy_size, color="gray")
            dummy_input = self.processor(images=dummy_image, return_tensors="pt")
            return {key: val.squeeze(0) for key, val in dummy_input.items()}, true_item_id_str, DEFAULT_IQON3000_CATEGORY

# =============================================================================
# Utility Functions
# =============================================================================

def collate_fn(batch):
    """Collate function for DataLoader"""
    inputs_dict = {}
    item_ids_list, main_cat_ids_list = [], []
    for b_input, item_id, main_cat_id in batch:
        if b_input is None: continue
        for key, val in b_input.items(): 
            inputs_dict.setdefault(key, []).append(val)
        item_ids_list.append(item_id)
        main_cat_ids_list.append(main_cat_id)
    
    if not inputs_dict: 
        return {'pixel_values': torch.empty(0, 3, 224, 224)}, [], []
    
    try:
        final_inputs = {}
        for key, val_list in inputs_dict.items():
            if val_list:
                final_inputs[key] = torch.stack(val_list)
            elif key == 'pixel_values': 
                final_inputs[key] = torch.empty(0, 3, 224, 224)
        return final_inputs, item_ids_list, main_cat_ids_list
    except RuntimeError as e: 
        print(f"collate_fn error: {e}")
        raise e

def pad_or_truncate_df(data_list, max_len, pad_value, dtype_override=None):
    """Pad or truncate data for DeepFurniture format"""
    if not data_list: 
        if isinstance(pad_value, np.ndarray): 
            return np.array([pad_value.copy() for _ in range(max_len)], dtype=dtype_override or pad_value.dtype)
        return np.array([pad_value] * max_len, dtype=dtype_override or type(pad_value))
    
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

def zscore_normalize(x: np.ndarray) -> np.ndarray:
    """Z-score normalization"""
    mean = x.mean(axis=(0, 1), keepdims=True)
    std = x.std(axis=(0, 1), keepdims=True) + 1e-7
    return (x - mean) / std

# =============================================================================
# IQON3000 Data Processing (7 categories)
# =============================================================================

def process_iqon3000(input_dir, output_dir, batch_size=32):
    """Process IQON3000 dataset to 7 categories"""
    print(f"Processing IQON3000 dataset from {input_dir} to 7 categories")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    dataset = IQON3000Dataset(input_dir, processor)
    if len(dataset) == 0:
        print("Dataset is empty. Skipping feature extraction.")
        return
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=4, 
                            pin_memory=device.type == 'cuda')
    
    # Extract features
    item_features = {}
    item_main_categories = {}
    item_to_set = {}
    set_to_items = {}
    
    print(f"Starting feature extraction for {len(dataset)} items...")
    
    with torch.no_grad():
        for batch_inputs, batch_true_item_ids, batch_main_cat_ids in tqdm(dataloader, desc="Extracting IQON3000 features"):
            if not batch_true_item_ids: continue
            if not batch_inputs or 'pixel_values' not in batch_inputs or batch_inputs['pixel_values'].numel() == 0: 
                continue
            
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            
            try: 
                img_feats = model.get_image_features(**batch_inputs)
            except Exception as e: 
                print(f"Error during get_image_features: {e}")
                continue 
            
            # Normalize features
            img_feats_norm = img_feats.norm(dim=1, keepdim=True)
            zero_norm_mask = (img_feats_norm == 0)
            img_feats_norm[zero_norm_mask] = 1e-9 
            img_feats = (img_feats / img_feats_norm).cpu().numpy()
            
            for i, true_item_id in enumerate(batch_true_item_ids):
                if true_item_id is None: continue
                item_features[true_item_id] = img_feats[i]
                item_main_categories[true_item_id] = batch_main_cat_ids[i]
                
                if true_item_id in dataset.item_info:
                    _, coordinate_id_for_item, _, _, _ = dataset.item_info[true_item_id]
                    item_to_set[true_item_id] = coordinate_id_for_item 
                    set_to_items.setdefault(coordinate_id_for_item, []).append(true_item_id)
    
    print(f"Extracted features for {len(item_features)} items from {len(set_to_items)} coordinate sets.")

    # Save category info
    category_info = {'main_categories_def': IQON3000_CATEGORIES, 'main_category_stats': {}}
    for item_id_stat in item_features: 
        main_cat = item_main_categories.get(item_id_stat, DEFAULT_IQON3000_CATEGORY)
        category_info['main_category_stats'][main_cat] = category_info['main_category_stats'].get(main_cat, 0) + 1
    
    with open(os.path.join(output_dir, 'category_info.pkl'), 'wb') as f: 
        pickle.dump(category_info, f)
    
    json_safe = {'main_categories_def': category_info['main_categories_def'], 
                 'main_category_stats': {str(k):v for k,v in category_info['main_category_stats'].items()}}
    with open(os.path.join(output_dir, 'category_info.json'), 'w', encoding='utf-8') as f: 
        json.dump(json_safe, f, ensure_ascii=False, indent=2)

    # Filter valid sets (4+ items with valid categories 1-7)
    valid_coords_with_orig_ids = {} 
    for coord_id, true_item_ids_in_coord in set_to_items.items():
        valid_items_in_this_coord = [tid for tid in true_item_ids_in_coord 
                                   if tid in item_main_categories and 1 <= item_main_categories[tid] <= 7]
        if len(valid_items_in_this_coord) >= 4:
            if true_item_ids_in_coord and true_item_ids_in_coord[0] in dataset.item_info:
                user_id_for_coord, _, _, _, _ = dataset.item_info[true_item_ids_in_coord[0]]
                valid_coords_with_orig_ids[(user_id_for_coord, coord_id)] = valid_items_in_this_coord
    
    print(f"Found {len(valid_coords_with_orig_ids)} valid coordinate sets for DeepFurniture format.")

    # Convert to DeepFurniture format
    set_data_for_deepfurniture_with_ids_as_key = []
    if valid_coords_with_orig_ids:
        for (user_id, coord_id_key), true_item_ids_in_valid_coord in valid_coords_with_orig_ids.items():
            items_data_for_this_coord = []
            for true_item_id in true_item_ids_in_valid_coord:
                if true_item_id in item_main_categories and true_item_id in item_features:
                     items_data_for_this_coord.append((true_item_id, item_main_categories[true_item_id], item_features[true_item_id]))
            if len(items_data_for_this_coord) >= 2:
                 set_data_for_deepfurniture_with_ids_as_key.append(((user_id, coord_id_key), items_data_for_this_coord))

    if set_data_for_deepfurniture_with_ids_as_key:
        train_sets_with_ids, test_val_sets_with_ids = train_test_split(set_data_for_deepfurniture_with_ids_as_key, test_size=0.2, random_state=42)
        val_sets_with_ids, test_sets_with_ids = [], []
        if test_val_sets_with_ids:
            if len(test_val_sets_with_ids) < 2: 
                val_sets_with_ids = test_val_sets_with_ids
            else: 
                val_sets_with_ids, test_sets_with_ids = train_test_split(test_val_sets_with_ids, test_size=0.5, random_state=42)
        
        print(f"IQON3000 Dataset split: Train {len(train_sets_with_ids)}, Validation {len(val_sets_with_ids)}, Test {len(test_sets_with_ids)}")
        
        # Process and save splits
        def process_and_save_iqon3000_split(dataset_with_orig_ids_list, split_name, base_output_dir):
            if not dataset_with_orig_ids_list:
                return None
            actual_item_data_for_split = [item[1] for item in dataset_with_orig_ids_list]
            return convert_to_deepfurniture_format(actual_item_data_for_split, 
                                                 os.path.join(base_output_dir, f'{split_name}.pkl'))

        if train_sets_with_ids:
            process_and_save_iqon3000_split(train_sets_with_ids, 'train', output_dir)
        if val_sets_with_ids:
            process_and_save_iqon3000_split(val_sets_with_ids, 'validation', output_dir)
        if test_sets_with_ids:
            process_and_save_iqon3000_split(test_sets_with_ids, 'test', output_dir)
    else:
        print("No valid coordinate sets for DeepFurniture format conversion.")

    # Compute and save category centers for 7 categories
    compute_iqon3000_category_centers(output_dir)
    
    print(f"IQON3000 processing complete. Files saved to {output_dir}")

def convert_to_deepfurniture_format(sets_of_items_data, output_file, max_item_num=10):
    """Convert sets to DeepFurniture format"""
    q_feats_list, t_feats_list, q_main_cats_list, t_main_cats_list, q_ids_list, t_ids_list = [], [], [], [], [], []
    scene_ids_list, set_sizes_list = [], []
    skipped_sets_count = 0
    
    for set_idx_df, items_in_current_outfit_data in enumerate(tqdm(sets_of_items_data, desc=f"DF Converting ({os.path.basename(output_file)})")):
        if len(items_in_current_outfit_data) < 2: 
            skipped_sets_count += 1
            continue        
        
        random.shuffle(items_in_current_outfit_data)
        split_idx = len(items_in_current_outfit_data) // 2
        if split_idx == 0: 
            skipped_sets_count += 1
            continue
        
        query_item_data_list = items_in_current_outfit_data[:split_idx]
        target_item_data_list = items_in_current_outfit_data[split_idx:]
        
        if not query_item_data_list or not target_item_data_list: 
            skipped_sets_count += 1
            continue
        
        q_ids_raw, q_main_cats_raw, q_feats_raw = zip(*[(d[0], d[1], d[2]) for d in query_item_data_list])
        t_ids_raw, t_main_cats_raw, t_feats_raw = zip(*[(d[0], d[1], d[2]) for d in target_item_data_list])
        
        if not q_feats_raw or not t_feats_raw: 
            skipped_sets_count += 1
            continue
        
        if not (q_feats_raw and isinstance(q_feats_raw[0], np.ndarray) and all(isinstance(f, np.ndarray) and f.shape == q_feats_raw[0].shape for f in q_feats_raw)):
            skipped_sets_count += 1
            continue
        
        if not (t_feats_raw and isinstance(t_feats_raw[0], np.ndarray) and all(isinstance(f, np.ndarray) and f.shape == t_feats_raw[0].shape for f in t_feats_raw)):
            skipped_sets_count += 1
            continue
        
        feature_dim = q_feats_raw[0].shape[0]
        zero_feature_pad = np.zeros(feature_dim, dtype=np.float32)
        
        q_ids_list.append(pad_or_truncate_df(q_ids_raw, max_item_num, '0', object))
        q_main_cats_list.append(pad_or_truncate_df(q_main_cats_raw, max_item_num, 0, np.int32))
        q_feats_list.append(pad_or_truncate_df(q_feats_raw, max_item_num, zero_feature_pad, np.float32))
        t_ids_list.append(pad_or_truncate_df(t_ids_raw, max_item_num, '0', object))
        t_main_cats_list.append(pad_or_truncate_df(t_main_cats_raw, max_item_num, 0, np.int32))
        t_feats_list.append(pad_or_truncate_df(t_feats_raw, max_item_num, zero_feature_pad, np.float32))
        scene_ids_list.append(set_idx_df)
        set_sizes_list.append(len(items_in_current_outfit_data))
    
    if skipped_sets_count > 0: 
        print(f"DF Converting ({os.path.basename(output_file)}): Skipped {skipped_sets_count} sets.")
    
    if not q_feats_list: 
        print(f"Warning: No data to save for {output_file}.")
        return {}
    
    df_tuple = (np.array(q_feats_list, dtype=np.float32), np.array(t_feats_list, dtype=np.float32), 
                np.array(scene_ids_list, dtype=np.int32), np.array(q_main_cats_list, dtype=np.int32), 
                np.array(t_main_cats_list, dtype=np.int32), np.array(set_sizes_list, dtype=np.int32),
                np.array(q_ids_list, dtype=object), np.array(t_ids_list, dtype=object))
    
    try:
        with open(output_file, 'wb') as f: 
            pickle.dump(df_tuple, f)
        print(f"Saved {len(q_feats_list)} sets to {output_file}.")
    except Exception as e_save: 
        print(f"Error saving {output_file}: {e_save}")
        return {}
    
    return {'query_categories': df_tuple[3], 'target_categories': df_tuple[4]}

def compute_iqon3000_category_centers(features_dir):
    """Compute category centers for IQON3000 (7 categories)"""
    train_path = os.path.join(features_dir, 'train.pkl')
    if not os.path.exists(train_path): 
        print(f"Error: Train data not found for category centers ({train_path})")
        return None, None
    
    try:
        with open(train_path, 'rb') as f: 
            data = pickle.load(f)
    except Exception as e: 
        print(f"Error loading train data {train_path}: {e}")
        return None, None
    
    if not (isinstance(data, tuple) and len(data) >= 8): 
        print(f"Error: Invalid train data format train.pkl.")
        return None, None
    
    query_features_all, query_main_category_ids_all = data[0], data[3]
    
    if not (isinstance(query_features_all, np.ndarray) and isinstance(query_main_category_ids_all, np.ndarray)): 
        print(f"Error: Invalid data types in train.pkl.")
        return None, None
    
    if query_features_all.ndim != 3 or query_main_category_ids_all.ndim != 2: 
        print(f"Error: Invalid data dimensions in train.pkl.")
        return None, None
    
    if query_features_all.shape[0] == 0: 
        print(f"Error: No data in train.pkl.")
        return None, None
    
    used_main_cat_ids = set(c_id for cs_set in query_main_category_ids_all for c_id in cs_set if 1 <= c_id <= 7 and c_id != 0)
    if not used_main_cat_ids: 
        print(f"No valid main category IDs (1-7) in train.pkl.")
        return None, None
    
    active_main_cat_ids = sorted(list(used_main_cat_ids))
    embedding_dim = query_features_all.shape[-1]
    print(f"\nCalculating centers for {len(active_main_cat_ids)} main categories (IDs: {min(active_main_cat_ids)}-{max(active_main_cat_ids)})...")
    
    category_centers_dict = {}
    features_per_main_category = {main_id: [] for main_id in active_main_cat_ids}
    
    for set_idx in range(query_features_all.shape[0]):
        for item_idx in range(query_features_all.shape[1]):
            main_cat_id = query_main_category_ids_all[set_idx, item_idx]
            if main_cat_id in active_main_cat_ids and not np.all(query_features_all[set_idx, item_idx] == 0):
                features_per_main_category[main_cat_id].append(query_features_all[set_idx, item_idx])
    
    for main_cat_id in active_main_cat_ids:
        if features_per_main_category[main_cat_id]:
            center_vec = np.mean(np.stack(features_per_main_category[main_cat_id]), axis=0)
            norm = np.linalg.norm(center_vec)
            category_centers_dict[main_cat_id] = (center_vec / norm if norm > 1e-9 else center_vec).astype(np.float32)
        else:
            print(f"Warning: No features for category ID {main_cat_id}. Initializing randomly.")
            rand_vec = np.random.randn(embedding_dim).astype(np.float32)
            norm_rand = np.linalg.norm(rand_vec)
            category_centers_dict[main_cat_id] = rand_vec / norm_rand if norm_rand > 1e-9 else rand_vec
    
    centers_array = np.zeros((len(IQON3000_CATEGORIES), embedding_dim), dtype=np.float32)
    for main_cat_id_map, center_vec_map in category_centers_dict.items():
        if 1 <= main_cat_id_map <= len(IQON3000_CATEGORIES): 
            centers_array[main_cat_id_map - 1] = center_vec_map
    
    for i in range(len(IQON3000_CATEGORIES)):
        if (i + 1) not in category_centers_dict: 
            print(f"Warning: Center for main category ID {i+1} ({IQON3000_CATEGORIES.get(i+1)}) not calculated.")
    
    with gzip.open(os.path.join(features_dir, 'category_centers.pkl.gz'), 'wb') as f: 
        pickle.dump(centers_array, f)
    
    print(f"\nSaved {len(category_centers_dict)} main category centers to category_centers.pkl.gz")
    return category_centers_dict, centers_array

# =============================================================================
# DeepFurniture Data Processing (11 categories)
# =============================================================================

def process_deepfurniture(image_dir, annotations_json, furnitures_jsonl, output_dir, batch_size=32):
    """Process DeepFurniture dataset to 11 categories"""
    print(f"Processing DeepFurniture dataset to 11 categories")
    print(f"  Images: {image_dir}")
    print(f"  Annotations: {annotations_json}")
    print(f"  Furnitures: {furnitures_jsonl}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load furniture metadata
    furniture_to_category = load_jsonl_mapping(Path(furnitures_jsonl), "furniture_id", "category_id")
    
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    model.eval()
    
    # Find and process images
    image_files = []
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        image_files.extend(Path(image_dir).rglob(ext))
    
    valid_images = [img for img in image_files if img.stem in furniture_to_category]
    print(f"Found {len(valid_images)} valid images")
    
    # Extract features
    all_features = []
    all_furniture_ids = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(valid_images), batch_size), desc="Extracting DeepFurniture features"):
            batch_paths = valid_images[i:i+batch_size]
            
            try:
                pil_images = [Image.open(p).convert("RGB") for p in batch_paths]
                inputs = processor(images=pil_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                output = model.get_image_features(**inputs)
                output = output / output.norm(dim=1, keepdim=True)  # Normalize
                batch_feats = output.cpu().numpy()
                
                all_features.append(batch_feats)
                all_furniture_ids.extend([p.stem for p in batch_paths])
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
    
    if not all_features:
        print("No features extracted, exiting")
        return
    
    all_features = np.vstack(all_features)
    print(f"Extracted features: {all_features.shape}")
    
    # Load scene annotations and build scenes
    with open(annotations_json, 'r') as f:
        annotations = json.load(f)
    
    scenes = {}
    category_ids_in_scenes = set()
    
    for scene_record in tqdm(annotations, desc="Building DeepFurniture scenes"):
        scene_id = scene_record.get("scene", {}).get("sceneTaskID")
        if not scene_id:
            continue
        
        scene_features = []
        scene_categories = []
        scene_item_ids = []
        
        for instance in scene_record.get("instances", []):
            furniture_id = instance.get("identityID")
            category_id = instance.get("categoryID")
            
            if not furniture_id or category_id is None:
                continue
            
            # Find feature for this furniture_id
            if furniture_id in all_furniture_ids:
                idx = all_furniture_ids.index(furniture_id)
                scene_features.append(all_features[idx])
                scene_categories.append(category_id)
                scene_item_ids.append(furniture_id)
                category_ids_in_scenes.add(category_id)
        
        if len(scene_features) >= 6:  # Minimum items per scene for DeepFurniture
            scenes[str(scene_id)] = {
                "features": np.stack(scene_features),
                "category_ids": np.array(scene_categories),
                "item_ids": np.array(scene_item_ids, dtype=object)
            }
    
    print(f"Built {len(scenes)} valid DeepFurniture scenes")
    print(f"Found category IDs in scenes: {sorted(list(category_ids_in_scenes))}")
    
    # Convert scenes to DeepFurniture format with z-score normalization
    convert_scenes_to_deepfurniture_format(scenes, output_dir)
    
    # Compute category centers for DeepFurniture (11 categories)
    compute_deepfurniture_category_centers(output_dir)
    
    # Save category mapping
    with open(os.path.join(output_dir, 'category_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'categories': DEEPFURNITURE_CATEGORIES
        }, f, ensure_ascii=False, indent=2)
    
    print(f"DeepFurniture processing complete. Files saved to {output_dir}")

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
                    print(f"[WARN] Skipping line {line_num+1} in '{path.name}' due to error: {e}")
        
        if mapping:
            print(f"[INFO] Loaded mapping from '{path.name}': {len(mapping)} entries.")
            print(f"[INFO] Found {len(unique_values_found)} unique category IDs: {sorted(list(unique_values_found))}")
            if unique_values_found:
                print(f"[INFO] Min/Max Category ID: {min(unique_values_found)} / {max(unique_values_found)}")
        else:
            print(f"[WARN] No valid entries loaded from '{path.name}'.")
    except FileNotFoundError:
        print(f"[ERROR] Metadata file not found: {path}")
    except Exception as e:
        print(f"[ERROR] Failed to load '{path.name}': {e}")
    
    return mapping

def convert_scenes_to_deepfurniture_format(scenes, output_dir, min_items=6, max_items=16, max_item_num=8):
    """Convert scenes to DeepFurniture format with train/val/test split and z-score normalization"""
    from collections import defaultdict
    
    # Build partitions
    q_feats, p_feats, q_cats, p_cats, q_ids, p_ids, s_keys = [], [], [], [], [], [], []
    
    rng = random.Random(42)
    
    for sid, rec in tqdm(scenes.items(), desc="Converting DeepFurniture scenes"):
        feats, cats, ids = rec["features"], rec["category_ids"], rec["item_ids"]
        n = len(feats)
        
        if n < min_items or n > max_items:
            continue
            
        idx = list(range(n))
        rng.shuffle(idx)
        q_idx, p_idx = idx[:n//2], idx[n//2:]
        
        # Pad function
        def pad_df(arr, target, pad_val=0):
            cur = len(arr)
            if cur < target:
                if arr.ndim > 1:
                    pad_shape = (target - cur, *arr.shape[1:])
                else:
                    pad_shape = (target - cur,)
                pad_arr = np.full(pad_shape, pad_val, dtype=arr.dtype)
                arr = np.concatenate([arr, pad_arr], axis=0)
            else:
                arr = arr[:target]
            return arr
        
        # Query
        qf = pad_df(feats[q_idx], max_item_num)
        qi = pad_df(ids[q_idx], max_item_num, pad_val="")
        qc = pad_df(cats[q_idx], max_item_num, pad_val=0)
        
        # Positive
        pf = pad_df(feats[p_idx], max_item_num)
        pi = pad_df(ids[p_idx], max_item_num, pad_val="")
        pc = pad_df(cats[p_idx], max_item_num, pad_val=0)
        
        q_feats.append(qf); q_ids.append(qi); q_cats.append(qc)
        p_feats.append(pf); p_ids.append(pi); p_cats.append(pc)
        s_keys.append(sid)
    
    if not q_feats:
        print("No valid scenes found for conversion")
        return
    
    Q = np.stack(q_feats)
    P = np.stack(p_feats)
    qcat = np.stack(q_cats)
    pcat = np.stack(p_cats)
    qid = np.stack(q_ids)
    pid = np.stack(p_ids)
    sids = np.array(s_keys)
    
    # Category remap (mask=0)
    unique = np.unique(np.concatenate([qcat, pcat]))
    unique = unique[unique > 0]
    cat_map = {cid: i + 1 for i, cid in enumerate(sorted(unique))}
    vect = np.vectorize(lambda c: cat_map.get(c, 0))
    qcat = vect(qcat)
    pcat = vect(pcat)
    
    # Train/val/test split
    (Q_tr, Q_tmp, qcat_tr, qcat_tmp, P_tr, P_tmp, pcat_tr, pcat_tmp,
     qid_tr, qid_tmp, pid_tr, pid_tmp, sid_tr, sid_tmp) = train_test_split(
        Q, qcat, P, pcat, qid, pid, sids,
        test_size=0.3, random_state=42)

    (Q_val, Q_te, qcat_val, qcat_te, P_val, P_te, pcat_val, pcat_te,
     qid_val, qid_te, pid_val, pid_te, sid_val, sid_te) = train_test_split(
        Q_tmp, qcat_tmp, P_tmp, pcat_tmp, qid_tmp, pid_tmp, sid_tmp,
        test_size=0.5, random_state=42)

    # Z-score normalization
    Q_tr, P_tr = zscore_normalize(Q_tr), zscore_normalize(P_tr)
    Q_val, P_val = zscore_normalize(Q_val), zscore_normalize(P_val)
    Q_te, P_te = zscore_normalize(Q_te), zscore_normalize(P_te)
    
    print(f"DeepFurniture split: Train {len(Q_tr)}, Validation {len(Q_val)}, Test {len(Q_te)}")
    
    # Save partitions
    def save_partition(path, objects):
        with open(path, "wb") as f:
            pickle.dump(objects, f)
    
    save_partition(os.path.join(output_dir, "train.pkl"), 
                  (Q_tr, P_tr, sid_tr, qcat_tr, pcat_tr, np.zeros(len(Q_tr)), qid_tr, pid_tr))
    save_partition(os.path.join(output_dir, "validation.pkl"), 
                  (Q_val, P_val, sid_val, qcat_val, pcat_val, np.zeros(len(Q_val)), qid_val, pid_val))
    save_partition(os.path.join(output_dir, "test.pkl"), 
                  (Q_te, P_te, sid_te, qcat_te, pcat_te, np.zeros(len(Q_te)), qid_te, pid_te))

def compute_deepfurniture_category_centers(features_dir):
    """Compute category centers for DeepFurniture (11 categories)"""
    train_path = os.path.join(features_dir, 'train.pkl')
    
    with open(train_path, 'rb') as f:
        Q_tr, P_tr, _, qcat_tr, pcat_tr, _, _, _ = pickle.load(f)
    
    # Category centers (train only)
    flat_feat = np.concatenate([Q_tr.reshape(-1, Q_tr.shape[-1]), P_tr.reshape(-1, P_tr.shape[-1])])
    flat_cat = np.concatenate([qcat_tr.reshape(-1), pcat_tr.reshape(-1)])
    
    unique_cats = np.unique(flat_cat)
    unique_cats = unique_cats[unique_cats > 0]
    
    centers = np.zeros((len(unique_cats), flat_feat.shape[1]))
    for i, cid in enumerate(sorted(unique_cats)):
        mask = flat_cat == cid
        centers[i] = flat_feat[mask].mean(axis=0) if mask.any() else 0
    
    with gzip.open(os.path.join(features_dir, "category_centers.pkl.gz"), "wb") as f:
        pickle.dump(centers, f)
    
    print(f"Saved {len(centers)} DeepFurniture category centers to category_centers.pkl.gz")

# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified dataset generator for IQON3000 and DeepFurniture")
    parser.add_argument('--dataset', choices=['iqon3000', 'deepfurniture'], required=True,
                      help='Dataset type to process')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for feature extraction')
    
    # IQON3000 arguments
    parser.add_argument('--input-dir', type=str,
                      help='Input directory for IQON3000 dataset')
    
    # DeepFurniture arguments
    parser.add_argument('--image-dir', type=str,
                      help='Image directory for DeepFurniture dataset')
    parser.add_argument('--annotations-json', type=str,
                      help='Annotations JSON file for DeepFurniture dataset')
    parser.add_argument('--furnitures-jsonl', type=str,
                      help='Furnitures JSONL file for DeepFurniture dataset')
    
    # Common arguments
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Output directory for processed dataset')
    
    args = parser.parse_args()
    
    if args.dataset == 'iqon3000':
        if not args.input_dir:
            parser.error("--input-dir is required for IQON3000 dataset")
        print("Processing IQON3000 dataset: 7 categories (outerwear, tops, bottoms, shoes, bags, hats, accessories)")
        process_iqon3000(args.input_dir, args.output_dir, args.batch_size)
        
    elif args.dataset == 'deepfurniture':
        if not all([args.image_dir, args.annotations_json, args.furnitures_jsonl]):
            parser.error("--image-dir, --annotations-json, and --furnitures-jsonl are required for DeepFurniture dataset")
        print("Processing DeepFurniture dataset: 11 categories (chair, table, sofa, bed, cabinet, lamp, bookshelf, desk, dresser, nightstand, other_furniture)")
        process_deepfurniture(args.image_dir, args.annotations_json, 
                            args.furnitures_jsonl, args.output_dir, args.batch_size)

if __name__ == "__main__":
    main()