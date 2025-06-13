#!/usr/bin/env python3
"""
修正版 make_datasets.py
========================
重複除去の統一、正規化処理の統一、エラーハンドリングの強化

主な修正点:
1. 重複除去: 同じfurniture_id/item_idは角度に関係なく1つのみ保持
2. 正規化統一: L2正規化のみ（Z-score正規化は削除）
3. カテゴリID体系の完全統一
4. エラーハンドリングとデバッグ情報の強化
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

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
tf.get_logger().setLevel("ERROR")

# =============================================================================
# 統一されたカテゴリ設定
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

DEEPFURNITURE_CATEGORIES = {
    1: "chair", 2: "table", 3: "sofa", 4: "bed", 5: "cabinet", 
    6: "lamp", 7: "bookshelf", 8: "desk", 9: "dresser", 
    10: "nightstand", 11: "other_furniture"
}

# =============================================================================
# 統一された正規化関数
# =============================================================================

def normalize_features_unified(features):
    """
    統一された特徴量正規化
    L2正規化のみを適用（Z-score正規化は削除）
    """
    if isinstance(features, torch.Tensor):
        # PyTorchテンソルの場合
        norm = features.norm(dim=-1, keepdim=True)
        zero_norm_mask = (norm == 0)
        norm[zero_norm_mask] = 1e-9
        return features / norm
    else:
        # NumPy配列の場合
        norm = np.linalg.norm(features, axis=-1, keepdims=True)
        norm[norm == 0] = 1e-9
        return features / norm


# =============================================================================
# IQON3000 Dataset Class (修正版)
# =============================================================================

class IQON3000Dataset(Dataset):
    def __init__(self, iqon_dir, processor):
        self.iqon_dir = iqon_dir
        self.processor = processor
        self.item_info = {}  # item_id -> (user_id, coordinate_id, category, cat_id, filename)
        self.items = []
        self._load_data()
        print(f"IQON3000: Loaded {len(self.items)} unique items from {len(self.item_info)} entries")
    
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

        # 統計情報
        source_category_counts = {}
        unmapped_to_default_count = 0
        json_decode_errors = 0
        items_missing_id_in_json = 0
        items_missing_image = 0
        duplicate_items_skipped = 0
        
        user_id_dirs = [d for d in os.listdir(self.iqon_dir) if os.path.isdir(os.path.join(self.iqon_dir, d))]
        print(f"Found {len(user_id_dirs)} user directories.")

        # 重複追跡用セット
        seen_item_ids = set()

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

                    # 重複チェック: 同じitem_idは1回のみ処理
                    if true_item_id_str in seen_item_ids:
                        duplicate_items_skipped += 1
                        continue
                    seen_item_ids.add(true_item_id_str)

                    # 画像ファイル名を決定
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

                    # カテゴリ情報の抽出
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
                    
                    # アイテム情報を保存（重複なし）
                    self.item_info[true_item_id_str] = (user_id_str, coordinate_id_str, jp_cat_name, main_cat_id, image_filename)
                    self.items.append(true_item_id_str)
        
        print(f"\nIQON3000 Data Loading Summary:")
        print(f"  Total items processed: {sum(source_category_counts.values()) + unmapped_to_default_count}")
        print(f"  Items skipped (duplicates): {duplicate_items_skipped}")
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
# 修正されたCollate関数
# =============================================================================

def collate_fn(batch):
    """修正版collate関数 - エラーハンドリング強化"""
    inputs_dict = {}
    item_ids_list, main_cat_ids_list = [], []
    
    for b_input, item_id, main_cat_id in batch:
        if b_input is None: 
            continue
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

# =============================================================================
# 修正されたパディング関数
# =============================================================================

def pad_or_truncate_df(data_list, max_len, pad_value, dtype_override=None):
    """修正版パディング関数"""
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

# =============================================================================
# IQON3000処理 (修正版)
# =============================================================================

def process_iqon3000(input_dir, output_dir, batch_size=32):
    """修正版IQON3000処理 - 重複除去と正規化統一"""
    print(f"Processing IQON3000 dataset from {input_dir} to 7 categories")
    print("Key improvements: Duplicate removal, unified normalization")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # デバイスとモデルの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    model.eval()
    
    # データセットとデータローダーの作成
    dataset = IQON3000Dataset(input_dir, processor)
    if len(dataset) == 0:
        print("Dataset is empty. Skipping feature extraction.")
        return
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=4, 
                            pin_memory=device.type == 'cuda')
    
    # 特徴量抽出
    item_features = {}
    item_main_categories = {}
    item_to_coordinate = {}
    coordinate_to_items = {}
    
    print(f"Starting feature extraction for {len(dataset)} unique items...")
    
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
            
            # 統一された正規化処理（L2正規化のみ）
            img_feats_normalized = normalize_features_unified(img_feats)
            img_feats_np = img_feats_normalized.cpu().numpy()
            
            for i, true_item_id in enumerate(batch_true_item_ids):
                if true_item_id is None: continue
                
                item_features[true_item_id] = img_feats_np[i]
                item_main_categories[true_item_id] = batch_main_cat_ids[i]
                
                # コーディネート情報を取得
                if true_item_id in dataset.item_info:
                    _, coordinate_id_for_item, _, _, _ = dataset.item_info[true_item_id]
                    item_to_coordinate[true_item_id] = coordinate_id_for_item 
                    coordinate_to_items.setdefault(coordinate_id_for_item, []).append(true_item_id)
    
    print(f"Extracted features for {len(item_features)} unique items from {len(coordinate_to_items)} coordinate sets.")
    
    # カテゴリ情報の保存
    category_info = {
        'main_categories_def': IQON3000_CATEGORIES, 
        'main_category_stats': {}
    }
    for item_id_stat in item_features: 
        main_cat = item_main_categories.get(item_id_stat, DEFAULT_IQON3000_CATEGORY)
        category_info['main_category_stats'][main_cat] = category_info['main_category_stats'].get(main_cat, 0) + 1
    
    with open(os.path.join(output_dir, 'category_info.pkl'), 'wb') as f: 
        pickle.dump(category_info, f)
    
    json_safe = {
        'main_categories_def': category_info['main_categories_def'], 
        'main_category_stats': {str(k):v for k,v in category_info['main_category_stats'].items()}
    }
    with open(os.path.join(output_dir, 'category_info.json'), 'w', encoding='utf-8') as f: 
        json.dump(json_safe, f, ensure_ascii=False, indent=2)

    # 有効なコーディネートの選別（4個以上のアイテム、カテゴリ1-7）
    valid_coordinates = {} 
    for coord_id, true_item_ids_in_coord in coordinate_to_items.items():
        valid_items_in_this_coord = [
            tid for tid in true_item_ids_in_coord 
            if tid in item_main_categories and 1 <= item_main_categories[tid] <= 7
        ]
        if len(valid_items_in_this_coord) >= 4:
            # 最初のアイテムからユーザーIDを取得
            if true_item_ids_in_coord and true_item_ids_in_coord[0] in dataset.item_info:
                user_id_for_coord, _, _, _, _ = dataset.item_info[true_item_ids_in_coord[0]]
                valid_coordinates[(user_id_for_coord, coord_id)] = valid_items_in_this_coord
    
    print(f"Found {len(valid_coordinates)} valid coordinate sets for conversion.")

    # DeepFurniture形式への変換
    set_data_for_conversion = []
    if valid_coordinates:
        for (user_id, coord_id_key), true_item_ids_in_valid_coord in valid_coordinates.items():
            items_data_for_this_coord = []
            for true_item_id in true_item_ids_in_valid_coord:
                if true_item_id in item_main_categories and true_item_id in item_features:
                     items_data_for_this_coord.append((
                         true_item_id, 
                         item_main_categories[true_item_id], 
                         item_features[true_item_id]
                     ))
            if len(items_data_for_this_coord) >= 2:
                 set_data_for_conversion.append(((user_id, coord_id_key), items_data_for_this_coord))

    if set_data_for_conversion:
        # データセット分割
        train_sets, test_val_sets = train_test_split(set_data_for_conversion, test_size=0.2, random_state=42)
        val_sets, test_sets = [], []
        if test_val_sets:
            if len(test_val_sets) < 2: 
                val_sets = test_val_sets
            else: 
                val_sets, test_sets = train_test_split(test_val_sets, test_size=0.5, random_state=42)
        
        print(f"IQON3000 Dataset split: Train {len(train_sets)}, Validation {len(val_sets)}, Test {len(test_sets)}")
        
        # 分割データの保存
        def process_and_save_iqon3000_split(dataset_with_orig_ids_list, split_name, base_output_dir):
            if not dataset_with_orig_ids_list:
                return None
            actual_item_data_for_split = [item[1] for item in dataset_with_orig_ids_list]
            return convert_to_deepfurniture_format_fixed(actual_item_data_for_split, 
                                                 os.path.join(base_output_dir, f'{split_name}.pkl'))

        if train_sets:
            process_and_save_iqon3000_split(train_sets, 'train', output_dir)
        if val_sets:
            process_and_save_iqon3000_split(val_sets, 'validation', output_dir)
        if test_sets:
            process_and_save_iqon3000_split(test_sets, 'test', output_dir)
    else:
        print("No valid coordinate sets for conversion.")

    # カテゴリ中心の計算（7カテゴリ用）
    compute_iqon3000_category_centers_fixed(output_dir)
    
    print(f"IQON3000 processing complete. Files saved to {output_dir}")

# =============================================================================
# 修正版: DeepFurniture形式変換関数
# =============================================================================

def convert_to_deepfurniture_format_fixed(sets_of_items_data, output_file, max_item_num=10):
    """修正版: DeepFurniture形式変換 - クエリ・ポジティブ重複チェック強化"""
    q_feats_list, t_feats_list, q_main_cats_list, t_main_cats_list, q_ids_list, t_ids_list = [], [], [], [], [], []
    scene_ids_list, set_sizes_list = [], []
    skipped_sets_count = 0
    overlap_warning_count = 0
    
    for set_idx_df, items_in_current_outfit_data in enumerate(tqdm(sets_of_items_data, desc=f"Converting ({os.path.basename(output_file)})")):
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
        
        # 重複チェック: クエリとターゲットに同じアイテムが含まれていないか確認
        query_ids = set([d[0] for d in query_item_data_list])
        target_ids = set([d[0] for d in target_item_data_list])
        
        if query_ids & target_ids:  # 重複が存在する場合
            overlap_warning_count += 1
            # 重複を解決: target側から重複するアイテムを除去
            target_item_data_list = [d for d in target_item_data_list if d[0] not in query_ids]
            if not target_item_data_list:  # ターゲットが空になった場合はスキップ
                skipped_sets_count += 1
                continue
        
        q_ids_raw, q_main_cats_raw, q_feats_raw = zip(*[(d[0], d[1], d[2]) for d in query_item_data_list])
        t_ids_raw, t_main_cats_raw, t_feats_raw = zip(*[(d[0], d[1], d[2]) for d in target_item_data_list])
        
        if not q_feats_raw or not t_feats_raw: 
            skipped_sets_count += 1
            continue
        
        # 特徴量の妥当性チェック
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
        print(f"Converting ({os.path.basename(output_file)}): Skipped {skipped_sets_count} sets.")
    if overlap_warning_count > 0:
        print(f"Converting ({os.path.basename(output_file)}): Resolved {overlap_warning_count} query-target overlaps.")
    
    if not q_feats_list: 
        print(f"Warning: No data to save for {output_file}.")
        return {}
    
    # データの整合性最終チェック
    print(f"Final data validation for {os.path.basename(output_file)}:")
    print(f"  Query-Target overlap check...")
    
    final_overlap_count = 0
    for i in range(len(q_ids_list)):
        q_items = set([item for item in q_ids_list[i] if item != '0' and item != 0])
        t_items = set([item for item in t_ids_list[i] if item != '0' and item != 0])
        if q_items & t_items:
            final_overlap_count += 1
    
    if final_overlap_count > 0:
        print(f"  ❌ WARNING: {final_overlap_count} sets still have query-target overlap!")
    else:
        print(f"  ✅ No query-target overlap detected in final data.")
    
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

# =============================================================================
# 修正版: カテゴリ中心計算関数
# =============================================================================

def compute_iqon3000_category_centers_fixed(features_dir):
    """修正版: IQON3000カテゴリ中心計算 - 辞書形式、正規化統一"""
    print("\nComputing IQON3000 category centers (fixed version)...")
    
    train_path = os.path.join(features_dir, 'train.pkl')
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found")
        return
    
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    
    if len(train_data) >= 8:
        query_features, target_features, scene_ids, query_categories, target_categories, set_sizes, query_item_ids, target_item_ids = train_data
    else:
        print(f"Error: Unexpected data format in {train_path}")
        return
    
    print(f"Loaded training data: {len(query_features)} sets")
    
    # IQON3000の7カテゴリID (1-7)
    active_main_cat_ids = list(range(1, 8))  # 1-7
    
    # 特徴量次元を取得
    if len(query_features) > 0 and query_features[0].shape[-1] > 0:
        embedding_dim = query_features[0].shape[-1]
    else:
        print("Error: No valid features found")
        return
    
    print(f"Feature dimension: {embedding_dim}")
    
    # カテゴリ別に全特徴量を収集
    features_per_main_category = {cat_id: [] for cat_id in active_main_cat_ids}
    
    # クエリとターゲットの特徴量を処理
    all_features = np.concatenate([query_features, target_features], axis=0)
    all_categories = np.concatenate([query_categories, target_categories], axis=0)
    
    print("Collecting features by category...")
    valid_feature_count = 0
    invalid_feature_count = 0
    
    for set_idx in tqdm(range(len(all_features)), desc="Processing sets"):
        for item_idx in range(len(all_features[set_idx])):
            feat = all_features[set_idx][item_idx]
            cat = all_categories[set_idx][item_idx]
            
            # パディング（ゼロベクトル）とカテゴリ範囲外をスキップ
            if cat == 0 or np.all(feat == 0) or cat not in active_main_cat_ids:
                invalid_feature_count += 1
                continue
            
            # 特徴量の正規化状態を確認（L2ノルムが1に近いかチェック）
            feat_norm = np.linalg.norm(feat)
            if feat_norm < 0.9 or feat_norm > 1.1:
                # 正規化されていない場合は再正規化
                feat = normalize_features_unified(feat.reshape(1, -1))[0]
            
            features_per_main_category[cat].append(feat)
            valid_feature_count += 1
    
    print(f"Processed {valid_feature_count} valid features, skipped {invalid_feature_count} invalid features")
    
    # カテゴリ中心を計算
    category_centers_dict = {}
    for main_cat_id in active_main_cat_ids:
        if features_per_main_category[main_cat_id]:
            center_vec = np.mean(np.stack(features_per_main_category[main_cat_id]), axis=0)
            # 中心ベクトルも正規化
            center_vec = normalize_features_unified(center_vec.reshape(1, -1))[0]
            category_centers_dict[main_cat_id] = center_vec.tolist()
            print(f"Category {main_cat_id}: {len(features_per_main_category[main_cat_id])} features")
        else:
            print(f"Warning: No features for category ID {main_cat_id}. Initializing randomly.")
            rand_vec = np.random.randn(embedding_dim).astype(np.float32)
            rand_vec = normalize_features_unified(rand_vec.reshape(1, -1))[0]
            category_centers_dict[main_cat_id] = rand_vec.tolist()
    
    # 辞書形式で保存
    output_path = os.path.join(features_dir, 'category_centers.pkl.gz')
    with gzip.open(output_path, 'wb') as f:
        pickle.dump(category_centers_dict, f)
    
    print(f"Saved {len(category_centers_dict)} main category centers to {output_path}")
    
    # カテゴリ名のマッピングも表示
    category_names = {
        1: "outerwear", 2: "tops", 3: "bottoms", 4: "shoes",
        5: "bags", 6: "hats", 7: "accessories"
    }
    
    print("\nCategory center summary:")
    for cat_id in sorted(category_centers_dict.keys()):
        cat_name = category_names.get(cat_id, f"Category {cat_id}")
        feature_count = len(features_per_main_category[cat_id])
        print(f"  {cat_id}: {cat_name} - {feature_count} features")
    
    return category_centers_dict

# =============================================================================
# DeepFurniture処理 (修正版)
# =============================================================================

def process_deepfurniture(image_dir, annotations_json, furnitures_jsonl, output_dir, batch_size=32):
    """修正版DeepFurniture処理 - 重複除去と正規化統一"""
    print(f"Processing DeepFurniture dataset to 11 categories")
    print("Key improvements: Proper duplicate removal, unified normalization")
    print(f"  Images: {image_dir}")
    print(f"  Annotations: {annotations_json}")
    print(f"  Furnitures: {furnitures_jsonl}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 家具メタデータの読み込み
    furniture_to_category = load_jsonl_mapping(Path(furnitures_jsonl), "furniture_id", "category_id")
    
    # デバイスとモデルの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    model.eval()
    
    # 画像ファイルの検索と処理
    image_files = []
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        image_files.extend(Path(image_dir).rglob(ext))
    
    valid_images = [img for img in image_files if img.stem in furniture_to_category]
    print(f"Found {len(valid_images)} valid images")
    
    # 重複除去のための追跡
    processed_furniture_ids = set()
    duplicate_count = 0
    
    # 特徴量抽出
    all_features = []
    all_furniture_ids = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(valid_images), batch_size), desc="Extracting DeepFurniture features"):
            batch_paths = valid_images[i:i+batch_size]
            
            # 重複チェック付きでバッチを処理
            filtered_batch_paths = []
            for path in batch_paths:
                furniture_id = path.stem
                if furniture_id not in processed_furniture_ids:
                    processed_furniture_ids.add(furniture_id)
                    filtered_batch_paths.append(path)
                else:
                    duplicate_count += 1
            
            if not filtered_batch_paths:
                continue
            
            try:
                pil_images = [Image.open(p).convert("RGB") for p in filtered_batch_paths]
                inputs = processor(images=pil_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                output = model.get_image_features(**inputs)
                # 統一された正規化処理
                output_normalized = normalize_features_unified(output)
                batch_feats = output_normalized.cpu().numpy()
                
                all_features.append(batch_feats)
                all_furniture_ids.extend([p.stem for p in filtered_batch_paths])
                
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
    
    if duplicate_count > 0:
        print(f"Removed {duplicate_count} duplicate furniture items")
    
    if not all_features:
        print("No features extracted, exiting")
        return
    
    all_features = np.vstack(all_features)
    print(f"Extracted features: {all_features.shape}")
    
    # シーンアノテーションの読み込みとシーン構築
    with open(annotations_json, 'r') as f:
        annotations = json.load(f)
    
    scenes = {}
    category_ids_in_scenes = set()
    
    for scene_record in tqdm(annotations, desc="Building DeepFurniture scenes"):
        scene_id = scene_record.get("scene", {}).get("sceneTaskID")
        if not scene_id:
            continue
        
        # 重複除去: 同じfurniture_idは1つのみ保持
        scene_items_unique = {}
        
        for instance in scene_record.get("instances", []):
            furniture_id = str(instance.get("identityID"))
            category_id = instance.get("categoryID")
    
            if not furniture_id or category_id is None:
                continue
 
            # 重複チェック: 同じfurniture_idは最初のもののみ保持
            if furniture_id not in scene_items_unique:
                if furniture_id in all_furniture_ids:
                    idx = all_furniture_ids.index(furniture_id)
                    scene_items_unique[furniture_id] = (all_features[idx], category_id)

        # 有効なシーンの構築（4個以上のユニークアイテム）
        if len(scene_items_unique) >= 4:
            item_ids_final = list(scene_items_unique.keys())
            features_final = np.array([v[0] for v in scene_items_unique.values()])
            categories_final = np.array([v[1] for v in scene_items_unique.values()])
          
            scenes[str(scene_id)] = {
                "features": features_final,
                "category_ids": categories_final,
                "item_ids": np.array(item_ids_final, dtype=object)
            }
            category_ids_in_scenes.update(categories_final)
    
    print(f"Built {len(scenes)} valid DeepFurniture scenes")
    print(f"Found category IDs in scenes: {sorted(list(category_ids_in_scenes))}")
    
    # シーンをDeepFurniture形式に変換
    convert_scenes_to_deepfurniture_format_fixed(scenes, output_dir)
    
    # カテゴリ中心の計算（11カテゴリ用）
    compute_deepfurniture_category_centers_fixed(output_dir)
    
    # カテゴリマッピングの保存
    with open(os.path.join(output_dir, 'category_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'categories': DEEPFURNITURE_CATEGORIES
        }, f, ensure_ascii=False, indent=2)
    
    print(f"DeepFurniture processing complete. Files saved to {output_dir}")

def load_jsonl_mapping(path, key_field, value_field):
    """JSONL ファイルマッピングの読み込み"""
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

def convert_scenes_to_deepfurniture_format_fixed(scenes, output_dir, min_items=4, max_items=20, max_item_num=10):
    """修正版: シーンをDeepFurniture形式に変換 - 重複チェック強化"""
    from collections import defaultdict
    import numpy as np
    import random
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm
    import pickle
    import os
    
    # シーンごとにクエリとポジティブを作成
    q_feats, p_feats, q_cats, p_cats, q_ids, p_ids, s_keys = [], [], [], [], [], [], []
    
    rng = random.Random(42)
    
    # 統計情報
    scene_count = 0
    skipped_scenes = 0
    overlap_resolved_count = 0
    
    for sid, rec in tqdm(scenes.items(), desc="Converting DeepFurniture scenes"):
        feats, cats, ids = rec["features"], rec["category_ids"], rec["item_ids"]
        n = len(feats)
        
        if n < min_items or n > max_items:
            skipped_scenes += 1
            continue
        
        # シーン内のアイテムをシャッフルして半分に分ける
        idx = list(range(n))
        rng.shuffle(idx)
        
        # 重複なしで半分ずつに分割
        half = n // 2
        q_idx = idx[:half]      # クエリ用インデックス
        p_idx = idx[half:]      # ポジティブ用インデックス
        
        # 重複チェック（デバッグ用）
        q_items_set = set([ids[i] for i in q_idx])
        p_items_set = set([ids[i] for i in p_idx])
        
        if q_items_set & p_items_set:
            print(f"WARNING: Scene {sid} has overlapping items between query and positive!")
            print(f"  Overlapping items: {q_items_set & p_items_set}")
            overlap_resolved_count += 1
            continue
        
        # パディング関数
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
        
        # クエリ
        qf = pad_df(feats[q_idx], max_item_num)
        qi = pad_df(ids[q_idx], max_item_num, pad_val="")
        qc = pad_df(cats[q_idx], max_item_num, pad_val=0)
        
        # ポジティブ
        pf = pad_df(feats[p_idx], max_item_num)
        pi = pad_df(ids[p_idx], max_item_num, pad_val="")
        pc = pad_df(cats[p_idx], max_item_num, pad_val=0)
        
        q_feats.append(qf); q_ids.append(qi); q_cats.append(qc)
        p_feats.append(pf); p_ids.append(pi); p_cats.append(pc)
        s_keys.append(sid)
        scene_count += 1
    
    print(f"\nProcessed {scene_count} scenes (skipped {skipped_scenes} scenes, resolved {overlap_resolved_count} overlaps)")
    
    if not q_feats:
        print("No valid scenes found for conversion")
        return
    
    # NumPy配列に変換
    Q = np.stack(q_feats)
    P = np.stack(p_feats)
    qcat = np.stack(q_cats)
    pcat = np.stack(p_cats)
    qid = np.stack(q_ids)
    pid = np.stack(p_ids)
    sids = np.array(s_keys)
    
    # カテゴリリマップ（パディング=0）
    unique = np.unique(np.concatenate([qcat, pcat]))
    unique = unique[unique > 0]
    cat_map = {cid: i + 1 for i, cid in enumerate(sorted(unique))}
    vect = np.vectorize(lambda c: cat_map.get(c, 0))
    qcat = vect(qcat)
    pcat = vect(pcat)
    
    print(f"Found {len(unique)} unique categories, remapped to 1-{len(unique)}")
    
    # シーンベースでtrain/val/testに分割
    indices = np.arange(len(Q))
    
    # 70/15/15の割合で分割
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    # 分割データを作成
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
    
    # 各分割内での重複チェック
    def check_query_positive_overlap(split_name, q_ids, p_ids):
        """クエリとポジティブ間の重複をチェック"""
        overlap_count = 0
        for i in range(len(q_ids)):
            q_items = set([item for item in q_ids[i] if item and item != "" and item != "0"])
            p_items = set([item for item in p_ids[i] if item and item != "" and item != "0"])
            if q_items & p_items:
                overlap_count += 1
        
        if overlap_count > 0:
            print(f"⚠️  {split_name}: {overlap_count}/{len(q_ids)} scenes have query-positive overlap")
        else:
            print(f"✅ {split_name}: No query-positive overlap in any scene")
        
        return overlap_count == 0
    
    print("\n=== Query-Positive Overlap Check ===")
    check_query_positive_overlap("Train", qid_tr, pid_tr)
    check_query_positive_overlap("Validation", qid_val, pid_val)
    check_query_positive_overlap("Test", qid_te, pid_te)
    
    # 正規化は既に統一されているためZ-score正規化は削除
    # すべての特徴量は既にL2正規化済み
    print(f"Features are already L2-normalized, skipping additional normalization")
    
    print(f"\nDeepFurniture split: Train {len(Q_tr)}, Validation {len(Q_val)}, Test {len(Q_te)}")
    
    # 分割データの保存
    def save_partition(path, objects):
        with open(path, "wb") as f:
            pickle.dump(objects, f)
    
    # 元の形式を維持してset_sizesは0で初期化
    save_partition(os.path.join(output_dir, "train.pkl"), 
                  (Q_tr, P_tr, sid_tr, qcat_tr, pcat_tr, np.zeros(len(Q_tr)), qid_tr, pid_tr))
    save_partition(os.path.join(output_dir, "validation.pkl"), 
                  (Q_val, P_val, sid_val, qcat_val, pcat_val, np.zeros(len(Q_val)), qid_val, pid_val))
    save_partition(os.path.join(output_dir, "test.pkl"), 
                  (Q_te, P_te, sid_te, qcat_te, pcat_te, np.zeros(len(Q_te)), qid_te, pid_te))
    
    print("\n✅ Dataset saved successfully!")

def compute_deepfurniture_category_centers_fixed(features_dir):
    """修正版: DeepFurnitureカテゴリ中心計算 - 辞書形式、正規化統一"""
    train_path = os.path.join(features_dir, 'train.pkl')
    
    with open(train_path, 'rb') as f:
        Q_tr, P_tr, _, qcat_tr, pcat_tr, _, _, _ = pickle.load(f)
    
    # カテゴリ中心計算（訓練データのみ）
    flat_feat = np.concatenate([Q_tr.reshape(-1, Q_tr.shape[-1]), P_tr.reshape(-1, P_tr.shape[-1])])
    flat_cat = np.concatenate([qcat_tr.reshape(-1), pcat_tr.reshape(-1)])
    
    unique_cats = np.unique(flat_cat)
    unique_cats = unique_cats[unique_cats > 0]
    
    # 辞書形式で保存（互換性のため）
    category_centers_dict = {}
    
    print("Computing DeepFurniture category centers...")
    for cid in sorted(unique_cats):
        mask = flat_cat == cid
        if mask.any():
            center_vec = flat_feat[mask].mean(axis=0)
            # 中心ベクトルも正規化
            center_vec = normalize_features_unified(center_vec.reshape(1, -1))[0]
            category_centers_dict[int(cid)] = center_vec.tolist()
            print(f"Category {cid}: {mask.sum()} features")
        else:
            # 空のカテゴリにはランダムな正規化ベクトルを割り当て
            rand_vec = np.random.randn(flat_feat.shape[1]).astype(np.float32)
            rand_vec = normalize_features_unified(rand_vec.reshape(1, -1))[0]
            category_centers_dict[int(cid)] = rand_vec.tolist()
    
    import gzip
    with gzip.open(os.path.join(features_dir, "category_centers.pkl.gz"), "wb") as f:
        pickle.dump(category_centers_dict, f)
    
    print(f"Saved {len(category_centers_dict)} DeepFurniture category centers to category_centers.pkl.gz")

# =============================================================================
# メイン関数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fixed unified dataset generator for IQON3000 and DeepFurniture")
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
        print("🔥 Processing IQON3000 dataset: 7 categories (FIXED VERSION)")
        print("   Key improvements:")
        print("   - Duplicate removal: same item_id only appears once")
        print("   - Unified normalization: L2 normalization only")
        print("   - Query-target overlap prevention")
        process_iqon3000(args.input_dir, args.output_dir, args.batch_size)
        
    elif args.dataset == 'deepfurniture':
        if not all([args.image_dir, args.annotations_json, args.furnitures_jsonl]):
            parser.error("--image-dir, --annotations-json, and --furnitures-jsonl are required for DeepFurniture dataset")
        print("🔥 Processing DeepFurniture dataset: 11 categories (FIXED VERSION)")
        print("   Key improvements:")
        print("   - Duplicate removal: same furniture_id only appears once")
        print("   - Unified normalization: L2 normalization only")
        print("   - Query-target overlap prevention")
        process_deepfurniture(args.image_dir, args.annotations_json, 
                            args.furnitures_jsonl, args.output_dir, args.batch_size)

if __name__ == "__main__":
    main()