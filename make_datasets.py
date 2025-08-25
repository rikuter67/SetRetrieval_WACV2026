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
tf.get_logger().setLevel("ERROR")

# =============================================================================
# 統一されたカテゴリ設定
# =============================================================================

IQON3000_CATEGORIES = {
    1: "outerwear",      # アウター（和服・ルームウェア含む）
    2: "tops",           # トップス
    3: "dresses",        # ワンピース・ドレス
    4: "pants",          # パンツ
    5: "skirts",         # スカート
    6: "shoes",          # シューズ
    7: "bags",           # バッグ
    8: "hats",           # 帽子
    9: "watches",        # 時計
    10: "accessories",   # ジュエリー・アクセサリー
    11: "fashion_goods", # ファッション雑貨（インテリア含む）
    12: "wallets",       # 財布・小物
    13: "legwear",       # レッグウェア
    14: "underwear",     # アンダーウェア・水着
    15: "beauty",        # ビューティー
    16: "glasses",       # 眼鏡
    17: "others"         # その他
}

# IQON3000実データ基準カテゴリマッピング
IQON3000_CATEGORY_MAPPING = {
    # カテゴリ1: アウター
    "ジャケット": 1, "コート": 1, "アウター": 1, "カーディガン": 1, 
    "ブルゾン": 1, "ダウン": 1, "パーカー": 1, "浴衣": 1, "着物": 1, "ルームウェア": 1,
    # カテゴリ2: トップス
    "Tシャツ": 2, "カットソー": 2, "シャツ": 2, "ブラウス": 2, 
    "ニット": 2, "セーター": 2, "ベスト": 2, "タンクトップ": 2, 
    "キャミソール": 2, "チュニック": 2, "トップス": 2, "インナー": 2,
    # カテゴリ3: ワンピース・ドレス
    "ワンピース": 3, "ドレス": 3,
    # カテゴリ4: パンツ
    "パンツ": 4, "ショートパンツ": 4, "ロングパンツ": 4, 
    "ジーンズ": 4, "デニム": 4, "レギンス": 4, "スラックス": 4,
    # カテゴリ5: スカート
    "スカート": 5, "ロングスカート": 5,
    # カテゴリ6: シューズ
    "シューズ": 6, "靴": 6, "スニーカー": 6, "サンダル": 6, 
    "ブーツ": 6, "パンプス": 6, "ルームシューズ": 6, "ローファー": 6,
    # カテゴリ7: バッグ
    "バッグ": 7, "トートバッグ": 7, "ショルダーバッグ": 7, 
    "ハンドバッグ": 7, "クラッチバッグ": 7, "ボストンバッグ": 7, 
    "リュック": 7, "ポーチ": 7,
    # カテゴリ8: 帽子
    "帽子": 8, "ハット": 8, "キャップ": 8, "ニット帽": 8, 
    "ベレー帽": 8, "キャスケット": 8,
    # カテゴリ9: 時計
    "時計": 9,
    # カテゴリ10: ジュエリー・アクセサリー
    "アクセサリー": 10, "ジュエリー": 10, "ネックレス": 10, 
    "ブレスレット": 10, "イヤリング": 10, "リング": 10, 
    "ピアス": 10, "ブローチ": 10, "ヘアアクセサリー": 10,
    # カテゴリ11: ファッション雑貨
    "ベルト": 11, "スカーフ": 11, "ストール": 11, "マフラー": 11, 
    "手袋": 11, "ファッション雑貨": 11,
    "ファッション小物": 11, "インテリア": 11, "ステーショナリー": 11,
    # カテゴリ12: 財布・小物
    "財布": 12, "キーケース・キーホルダー": 12, "小物": 12,
    # カテゴリ13: レッグウェア
    "ソックス・靴下": 13, "タイツ・ストッキング": 13, "レッグウェア": 13,
    # カテゴリ14: アンダーウェア・水着
    "アンダーウェア": 14, "水着": 14,
    # カテゴリ15: ビューティー
    "コスメ": 15, "ネイル": 15, "フレグランス": 15, "ボディケア": 15,
    # カテゴリ16: サングラス・メガネ
    "サングラス": 16, "メガネ": 16,
    # カテゴリ17: その他
    "傘・日傘": 17, "傘": 17, "その他": 17,
}

DEFAULT_IQON3000_CATEGORY = 17

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
        self.category_corrections = 0  # 修正されたカテゴリの数
        self.validation_stats = {
            'total_processed': 0,
            'corrections_made': 0,
            'conflicts_detected': 0,
            'name_based_inferences': 0
        }
        self._load_data()
        print(f"IQON3000: Loaded {len(self.items)} unique items from {len(self.item_info)} entries")
        self._print_validation_stats()
    
    def _validate_and_correct_category(self, item_detail):
        """
        商品名とカテゴリフィールドの整合性をチェックし、
        必要に応じて商品名から正しいカテゴリを推定する
        """
        self.validation_stats['total_processed'] += 1
        
        # 既存のカテゴリフィールドを取得
        cat_field = item_detail.get('category x color', item_detail.get('categoryName', ''))
        if not cat_field and 'category' in item_detail and isinstance(item_detail['category'], dict):
            cat_field = item_detail['category'].get('name', '')
        
        declared_category = cat_field.split(' × ')[0].strip() if ' × ' in cat_field else cat_field.strip()
        
        # 商品名を取得
        item_name = item_detail.get('itemName', item_detail.get('name', '')).lower()
        
        # 商品名から実際のカテゴリを推定
        inferred_category = self._infer_category_from_name(item_name)
        
        if inferred_category:
            self.validation_stats['name_based_inferences'] += 1
        
        # 整合性チェック
        if inferred_category and declared_category:
            # 両方存在する場合は整合性をチェック
            if self._categories_conflict(declared_category, inferred_category):
                self.validation_stats['conflicts_detected'] += 1
                self.validation_stats['corrections_made'] += 1
                
                # if self.category_corrections < 10:  # 最初の10件のみログ出力
                #     print(f"⚠️ カテゴリ不整合検出 #{self.validation_stats['conflicts_detected']}:")
                #     print(f"   商品名: {item_name[:50]}...")
                #     print(f"   宣言カテゴリ: {declared_category}")
                #     print(f"   推定カテゴリ: {inferred_category}")
                #     print(f"   → 推定カテゴリを採用")
                
                return inferred_category
        
        # 宣言カテゴリが空で推定カテゴリが存在する場合
        if not declared_category and inferred_category:
            self.validation_stats['corrections_made'] += 1
            return inferred_category
        
        # 推定カテゴリが存在し、宣言カテゴリと矛盾しない場合は推定を優先
        return inferred_category or declared_category or ''

    def _infer_category_from_name(self, item_name):
        """商品名からカテゴリを推定"""
        item_name = item_name.lower()
        
        # より具体的なキーワードから優先的にチェック
        # 長いキーワードから短いキーワードの順でチェック
        category_keywords = {
            "サングラス": "サングラス",
            "sunglasses": "サングラス",
            "メガネ": "メガネ",
            "眼鏡": "メガネ",
            "glasses": "メガネ",
            "アイウェア": "メガネ",  
            "eyewear": "サングラス",
            "スニーカー": "シューズ",
            "ナイキ": "シューズ",
            "nike": "シューズ",
            "アディダス": "シューズ", 
            "adidas": "シューズ",
            "コンバース": "シューズ",
            "converse": "シューズ",
            "タンジュン": "シューズ",
            "tanjun": "シューズ",
            "エアマックス": "シューズ",
            "air max": "シューズ",
            "airmax": "シューズ",
            "スタンスミス": "シューズ",
            "stan smith": "シューズ",
            "コート バーロウ": "シューズ",
            "court barrow": "シューズ",
            "ブーツ": "シューズ",
            "boots": "シューズ",
            "パンプス": "シューズ",
            "pumps": "シューズ",
            "サンダル": "シューズ",
            "sandal": "シューズ",
            "ローファー": "シューズ",
            "loafer": "シューズ",
            
            # バッグ系（次に優先）
            "バッグパック": "バッグ",
            "バックパック": "バッグ",
            "backpack": "バッグ",
            "eastpak": "バッグ",
            "リュックサック": "バッグ",
            "リュック": "バッグ",
            "rucksack": "バッグ",
            "トートバッグ": "バッグ",
            "tote bag": "バッグ",
            "ショルダーバッグ": "バッグ",
            "shoulder bag": "バッグ",
            "ハンドバッグ": "バッグ",
            "hand bag": "バッグ",
            "クラッチバッグ": "バッグ",
            "clutch bag": "バッグ",
            
            # 帽子系（キャップとの混同を避けるため早めにチェック）
            "ニット帽": "帽子",
            "knit cap": "帽子",
            "ベレー帽": "帽子",
            "beret": "帽子",
            "キャスケット": "帽子",
            "casquette": "帽子",
            "ハット": "帽子",
            "hat": "帽子",
            "キャップ": "帽子",
            "cap": "帽子",
            
            # アクセサリー系
            "ネックレス": "アクセサリー",
            "necklace": "アクセサリー",
            "ブレスレット": "アクセサリー",
            "bracelet": "アクセサリー",
            "イヤリング": "アクセサリー",
            "earring": "アクセサリー",
            "ピアス": "アクセサリー",
            "pierce": "アクセサリー",
            "指輪": "アクセサリー",
            "リング": "アクセサリー",
            "ring": "アクセサリー",
            "イヤーマフ": "アクセサリー",
            "earmuff": "アクセサリー",
            "手袋": "アクセサリー",
            "glove": "アクセサリー",
            "グローブ": "アクセサリー",
            
            # 一般的なカテゴリ（最後にチェック）
            "プリントtシャツ": "Tシャツ",
            "tシャツ": "Tシャツ",
            "t-shirt": "Tシャツ",
            "カットソー": "Tシャツ",
            "cut sew": "Tシャツ",
            "パーカー": "パーカー",
            "hoodie": "パーカー",
            "プルオーバー": "パーカー",
            "pullover": "パーカー",
            "ジャケット": "ジャケット",
            "jacket": "ジャケット",
            "ブルゾン": "ジャケット",
            "blouson": "ジャケット",
            "コート": "コート",
            "coat": "コート",
            "パンツ": "パンツ",
            "pants": "パンツ",
            "ロングパンツ": "パンツ",
            "long pants": "パンツ",
            "ショートパンツ": "ショートパンツ",
            "short pants": "ショートパンツ",
            "デニム": "パンツ",
            "denim": "パンツ",
            "ジーンズ": "パンツ",
            "jeans": "パンツ",
            "スカート": "スカート",
            "skirt": "スカート",
            "ワンピース": "ワンピース",
            "dress": "ワンピース",
            "ドレス": "ワンピース",
            "ニット": "ニット",
            "knit": "ニット",
            "セーター": "ニット",
            "sweater": "ニット",
        }
        
        # より長い（具体的な）キーワードから順にチェック
        sorted_keywords = sorted(category_keywords.items(), 
                               key=lambda x: len(x[0]), 
                               reverse=True)
        
        for keyword, category in sorted_keywords:
            if keyword in item_name:
                return category
        
        return None

    def _categories_conflict(self, declared_cat, inferred_cat):
        """カテゴリが矛盾しているかチェック"""
        if not declared_cat or not inferred_cat:
            return False
        
        # 同じカテゴリファミリーの場合は矛盾なし
        shoe_family = ["シューズ", "靴", "スニーカー", "ブーツ", "サンダル", "パンプス", "ローファー"]
        bag_family = ["バッグ", "リュック", "トートバッグ", "ショルダーバッグ", "ハンドバッグ", "クラッチバッグ"]
        hat_family = ["帽子", "キャップ", "ハット", "ニット帽", "ベレー帽"]
        accessory_family = ["アクセサリー", "ピアス", "ネックレス", "ブレスレット", "イヤリング", "手袋"]
        top_family = ["Tシャツ", "カットソー", "シャツ", "ブラウス", "ニット", "セーター", "パーカー"]
        eyewear_family = ["サングラス", "メガネ", "眼鏡"]
        
        families = [shoe_family, bag_family, hat_family, accessory_family, top_family, eyewear_family]
        
        # 同じファミリー内なら矛盾なし
        for family in families:
            if declared_cat in family and inferred_cat in family:
                return False
        
        # 明らかに異なるカテゴリファミリー間は矛盾
        declared_family = None
        inferred_family = None
        
        for i, family in enumerate(families):
            if declared_cat in family:
                declared_family = i
            if inferred_cat in family:
                inferred_family = i
        
        # どちらも主要ファミリーに属し、異なるファミリーの場合は矛盾
        if (declared_family is not None and inferred_family is not None and 
            declared_family != inferred_family):
            return True
        
        return False

    def _print_validation_stats(self):
        """カテゴリ検証統計を出力"""
        stats = self.validation_stats
        print(f"\n🔧 カテゴリ検証統計:")
        print(f"  処理されたアイテム: {stats['total_processed']}")
        print(f"  商品名からの推定: {stats['name_based_inferences']}")
        print(f"  矛盾検出: {stats['conflicts_detected']}")
        print(f"  修正適用: {stats['corrections_made']}")
        if stats['total_processed'] > 0:
            print(f"  修正率: {stats['corrections_made']/stats['total_processed']*100:.1f}%")
            print(f"  推定率: {stats['name_based_inferences']/stats['total_processed']*100:.1f}%")

    def _map_to_main_category_strict(self, japanese_category_name):
        # 既存の実装をそのまま維持
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
        print(f"Starting IQON3000 data loading with CATEGORY VALIDATION from: {self.iqon_dir}")
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

        for user_id_str in tqdm(user_id_dirs, desc="Processing IQON3000 Users with validation"):
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

                    # 🆕 カテゴリ検証・修正を追加
                    corrected_category = self._validate_and_correct_category(item_detail)
                    
                    if corrected_category:
                        source_category_counts[corrected_category] = source_category_counts.get(corrected_category, 0) + 1
                    
                    main_cat_id, was_fallback = self._map_to_main_category_strict(corrected_category)
                    if was_fallback:
                        unmapped_to_default_count += 1
                    
                    # アイテム情報を保存（重複なし）
                    self.item_info[true_item_id_str] = (user_id_str, coordinate_id_str, corrected_category, main_cat_id, image_filename)
                    self.items.append(true_item_id_str)
        
        print(f"\nIQON3000 Data Loading Summary (with Category Validation):")
        print(f"  Total items processed: {self.validation_stats['total_processed']}")
        print(f"  Items skipped (duplicates): {duplicate_items_skipped}")
        print(f"  Items skipped (missing 'itemId'): {items_missing_id_in_json}")
        print(f"  Items skipped (missing image): {items_missing_image}")
        print(f"  JSON decode errors: {json_decode_errors}")
        print(f"  Unique item entries: {len(self.item_info)}")
        print(f"  Valid items: {len(self.items)}")
        print(f"  Items mapped to default category: {unmapped_to_default_count}")


    # 既存の __len__ と __getitem__ メソッドはそのまま維持
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

def split_data_by_users(set_data_for_conversion, test_size=0.15, val_size=0.15, random_state=42):
    """
    ユーザーベースでデータセット分割
    同じユーザーのデータが複数の分割に混在しないようにする
    """
    print("🔄 ユーザーベースでデータセット分割を実行...")
    
    # ユーザーごとにデータをグループ化
    user_to_data = {}
    for data_item in set_data_for_conversion:
        if isinstance(data_item, tuple) and len(data_item) == 2:
            (user_id, coord_id), items_data = data_item
            if user_id not in user_to_data:
                user_to_data[user_id] = []
            user_to_data[user_id].append(data_item)
        else:
            # フォールバック: 従来形式の場合は警告
            print(f"⚠️ 警告: 予期しないデータ形式です")
            continue
    
    # ユーザー統計
    user_data_counts = {user_id: len(data_list) for user_id, data_list in user_to_data.items()}
    total_users = len(user_to_data)
    total_scenes = sum(user_data_counts.values())
    
    print(f"📊 ユーザー統計:")
    print(f"  総ユーザー数: {total_users}")
    print(f"  総シーン数: {total_scenes}")
    print(f"  ユーザーあたり平均シーン数: {total_scenes/total_users:.1f}")
    
    # データ量の多いユーザーから順にソート
    sorted_users = sorted(user_to_data.keys(), 
                         key=lambda u: len(user_to_data[u]), 
                         reverse=True)
    
    # ユーザーを分割
    np.random.seed(random_state)
    shuffled_users = np.random.permutation(sorted_users)
    
    # 累積データ量で分割点を決定
    cumulative_scenes = 0
    train_users, val_users, test_users = [], [], []
    
    for user_id in shuffled_users:
        user_scene_count = len(user_to_data[user_id])
        
        # 現在の累積率を計算
        current_ratio = cumulative_scenes / total_scenes
        
        if current_ratio < (1 - test_size - val_size):
            train_users.append(user_id)
        elif current_ratio < (1 - test_size):
            val_users.append(user_id)
        else:
            test_users.append(user_id)
        
        cumulative_scenes += user_scene_count
    
    # 各分割にデータを割り当て
    train_data = []
    val_data = []
    test_data = []
    
    for user_id in train_users:
        train_data.extend(user_to_data[user_id])
    
    for user_id in val_users:
        val_data.extend(user_to_data[user_id])
    
    for user_id in test_users:
        test_data.extend(user_to_data[user_id])
    
    # 統計を表示
    print(f"\n📈 ユーザーベース分割結果:")
    print(f"  Train: {len(train_users)} users, {len(train_data)} scenes")
    print(f"  Val:   {len(val_users)} users, {len(val_data)} scenes")
    print(f"  Test:  {len(test_users)} users, {len(test_data)} scenes")
    
    # 重複チェック
    train_user_set = set(train_users)
    val_user_set = set(val_users)
    test_user_set = set(test_users)
    
    train_val_overlap = train_user_set & val_user_set
    train_test_overlap = train_user_set & test_user_set
    val_test_overlap = val_user_set & test_user_set
    
    print(f"\n✅ 重複チェック:")
    print(f"  Train-Val重複: {len(train_val_overlap)} users")
    print(f"  Train-Test重複: {len(train_test_overlap)} users")
    print(f"  Val-Test重複: {len(val_test_overlap)} users")
    
    if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
        print("  🎉 ユーザー重複なし！完全分離成功")
    else:
        print("  ⚠️ ユーザー重複が検出されました")
    
    return train_data, val_data, test_data

def process_iqon3000(input_dir, output_dir, batch_size=32):
    """修正版: ユーザーベース分割 + 正規化統一"""
    print(f"Processing IQON3000 dataset with USER-BASED splitting")
    print("Key improvements: User-based split + proper normalization")
    
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
    
    # 特徴量抽出（正規化なし）
    item_features = {}
    item_main_categories = {}
    item_to_coordinate = {}
    coordinate_to_items = {}
    
    print(f"Starting RAW feature extraction for {len(dataset)} unique items...")
    
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
            
            # ✅ 正規化を削除 - 生の特徴量を保存
            img_feats_np = img_feats.cpu().numpy()
            
            for i, true_item_id in enumerate(batch_true_item_ids):
                if true_item_id is None: continue
                
                item_features[true_item_id] = img_feats_np[i]
                item_main_categories[true_item_id] = batch_main_cat_ids[i]
                
                # コーディネート情報を取得
                if true_item_id in dataset.item_info:
                    _, coordinate_id_for_item, _, _, _ = dataset.item_info[true_item_id]
                    item_to_coordinate[true_item_id] = coordinate_id_for_item 
                    coordinate_to_items.setdefault(coordinate_id_for_item, []).append(true_item_id)
    
    print(f"Extracted RAW features for {len(item_features)} unique items from {len(coordinate_to_items)} coordinate sets.")
    
    # 有効なコーディネートの選別（パス形式のキーを使用）
    valid_coordinates = {} 
    for coord_id, true_item_ids_in_coord in coordinate_to_items.items():
        valid_items_in_this_coord = [
            tid for tid in true_item_ids_in_coord 
            if tid in item_main_categories and 1 <= item_main_categories[tid] <= 17
        ]
        if len(valid_items_in_this_coord) >= 4:
            if true_item_ids_in_coord and true_item_ids_in_coord[0] in dataset.item_info:
                user_id_for_coord, _, _, _, _ = dataset.item_info[true_item_ids_in_coord[0]]
                path_key = (user_id_for_coord, coord_id)
                valid_coordinates[path_key] = valid_items_in_this_coord
    
    print(f"Found {len(valid_coordinates)} valid coordinate sets for conversion.")

    # データセット変換
    set_data_for_conversion = []
    if valid_coordinates:
        for (user_id, coord_id_key), true_item_ids_in_valid_coord in valid_coordinates.items():
            items_data_for_this_coord = []
            for true_item_id in true_item_ids_in_valid_coord:
                if true_item_id in item_main_categories and true_item_id in item_features:
                     items_data_for_this_coord.append((
                         true_item_id, 
                         item_main_categories[true_item_id], 
                         item_features[true_item_id]  # 生の特徴量
                     ))
            if len(items_data_for_this_coord) >= 2:
                 set_data_for_conversion.append(((user_id, coord_id_key), items_data_for_this_coord))

    if set_data_for_conversion:
        # ✅ ユーザーベース分割
        train_sets, val_sets, test_sets = split_data_by_users(
            set_data_for_conversion, 
            test_size=0.2, 
            val_size=0.1, 
            random_state=42
        )
        
        print(f"USER-BASED Dataset split: Train {len(train_sets)}, Validation {len(val_sets)}, Test {len(test_sets)}")
        
        # ✅ 学習データから正規化統計量を計算
        train_normalization_stats = compute_normalization_stats(train_sets)
        
        # ✅ 統一された正規化で全分割を処理
        def process_and_save_iqon3000_split(dataset_with_orig_ids_list, split_name, norm_stats, base_output_dir):
            if not dataset_with_orig_ids_list:
                return None
            return convert_to_deepfurniture_format_fixed(
                dataset_with_orig_ids_list, 
                os.path.join(base_output_dir, f'{split_name}.pkl'),
                norm_stats
            )

        if train_sets:
            process_and_save_iqon3000_split(train_sets, 'train', train_normalization_stats, output_dir)
        if val_sets:
            process_and_save_iqon3000_split(val_sets, 'validation', train_normalization_stats, output_dir)
        if test_sets:
            process_and_save_iqon3000_split(test_sets, 'test', train_normalization_stats, output_dir)
    else:
        print("No valid coordinate sets for conversion.")
    
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

    # カテゴリ中心の計算
    compute_iqon3000_category_centers_fixed(output_dir)
    
    print(f"IQON3000 processing complete with USER-BASED split. Files saved to {output_dir}")

def compute_normalization_stats(train_sets):
    """学習データから正規化統計量を計算"""
    print("Computing normalization statistics from training data...")
    
    all_train_features = []
    
    for (user_id, coord_id), items_data in train_sets:
        for item_id, category, feature in items_data:
            all_train_features.append(feature)
    
    if not all_train_features:
        print("Warning: No training features found")
        return None
    
    # 特徴量を結合
    train_features_array = np.stack(all_train_features)
    
    # L2正規化統計量（実際にはL2正規化は統計量不要だが、一貫性のため）
    feature_norms = np.linalg.norm(train_features_array, axis=1, keepdims=True)
    feature_norms[feature_norms == 0] = 1e-9
    
    # Z-score正規化用統計量（オプション）
    feature_mean = np.mean(train_features_array, axis=0)
    feature_std = np.std(train_features_array, axis=0)
    feature_std[feature_std == 0] = 1e-9
    
    normalization_stats = {
        'method': 'l2_norm',  # 'l2_norm' or 'z_score'
        'train_mean': feature_mean,
        'train_std': feature_std,
        'train_feature_count': len(all_train_features)
    }
    
    print(f"Computed normalization stats from {len(all_train_features)} training features")
    print(f"Feature mean magnitude: {np.mean(np.abs(feature_mean)):.6f}")
    print(f"Feature std mean: {np.mean(feature_std):.6f}")
    
    return normalization_stats

def apply_normalization_with_stats(features, normalization_stats):
    """統計量を使って正規化を適用"""
    if normalization_stats is None:
        # フォールバック: L2正規化
        norm = np.linalg.norm(features, axis=-1, keepdims=True)
        norm[norm == 0] = 1e-9
        return features / norm
    
    method = normalization_stats.get('method', 'l2_norm')
    
    if method == 'l2_norm':
        # L2正規化（統計量不要）
        norm = np.linalg.norm(features, axis=-1, keepdims=True)
        norm[norm == 0] = 1e-9
        return features / norm
    
    elif method == 'z_score':
        # Z-score正規化（学習データ統計量使用）
        train_mean = normalization_stats['train_mean']
        train_std = normalization_stats['train_std']
        return (features - train_mean) / train_std
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

# =============================================================================
# 修正版: DeepFurniture形式変換関数
# =============================================================================

def convert_to_deepfurniture_format_fixed(sets_of_items_data, output_file, normalization_stats=None, max_item_num=10):
    """正規化統計量を使用してDeepFurniture形式に変換"""
    q_feats_list, t_feats_list, q_main_cats_list, t_main_cats_list, q_ids_list, t_ids_list = [], [], [], [], [], []
    scene_ids_list, set_sizes_list = [], []
    skipped_sets_count = 0
    overlap_warning_count = 0
    
    print(f"Converting with normalization stats: {normalization_stats['method'] if normalization_stats else 'fallback_l2'}")
    
    for set_idx_df, items_in_current_outfit_data in enumerate(tqdm(sets_of_items_data, desc=f"Converting ({os.path.basename(output_file)})")):
        # データ構造の処理（既存と同じ）
        scene_id = None
        items_to_process = None
        
        if isinstance(items_in_current_outfit_data, tuple) and len(items_in_current_outfit_data) == 2:
            first_elem, second_elem = items_in_current_outfit_data
            if isinstance(first_elem, tuple) and len(first_elem) == 2:
                user_id, coord_id = first_elem
                scene_id = f"{user_id}/{coord_id}"
                items_to_process = second_elem
            else:
                items_to_process = items_in_current_outfit_data
                scene_id = f"scene_{set_idx_df}"
        else:
            items_to_process = items_in_current_outfit_data
            scene_id = f"scene_{set_idx_df}"
        
        if len(items_to_process) < 2: 
            skipped_sets_count += 1
            continue
        
        random.shuffle(items_to_process)
        split_idx = len(items_to_process) // 2
        if split_idx == 0: 
            skipped_sets_count += 1
            continue
        
        query_item_data_list = items_to_process[:split_idx]
        target_item_data_list = items_to_process[split_idx:]
        
        if not query_item_data_list or not target_item_data_list: 
            skipped_sets_count += 1
            continue
        
        # 重複チェック
        query_ids = set([d[0] for d in query_item_data_list])
        target_ids = set([d[0] for d in target_item_data_list])
        
        if query_ids & target_ids:
            overlap_warning_count += 1
            target_item_data_list = [d for d in target_item_data_list if d[0] not in query_ids]
            if not target_item_data_list:
                skipped_sets_count += 1
                continue
        
        q_ids_raw, q_main_cats_raw, q_feats_raw = zip(*[(d[0], d[1], d[2]) for d in query_item_data_list])
        t_ids_raw, t_main_cats_raw, t_feats_raw = zip(*[(d[0], d[1], d[2]) for d in target_item_data_list])
        
        if not q_feats_raw or not t_feats_raw: 
            skipped_sets_count += 1
            continue
        
        # ✅ 統一された正規化を適用
        q_feats_normalized = [apply_normalization_with_stats(feat, normalization_stats) for feat in q_feats_raw]
        t_feats_normalized = [apply_normalization_with_stats(feat, normalization_stats) for feat in t_feats_raw]
        
        feature_dim = q_feats_normalized[0].shape[0]
        zero_feature_pad = np.zeros(feature_dim, dtype=np.float32)
        
        q_ids_list.append(pad_or_truncate_df(q_ids_raw, max_item_num, '0', object))
        q_main_cats_list.append(pad_or_truncate_df(q_main_cats_raw, max_item_num, 0, np.int32))
        q_feats_list.append(pad_or_truncate_df(q_feats_normalized, max_item_num, zero_feature_pad, np.float32))
        t_ids_list.append(pad_or_truncate_df(t_ids_raw, max_item_num, '0', object))
        t_main_cats_list.append(pad_or_truncate_df(t_main_cats_raw, max_item_num, 0, np.int32))
        t_feats_list.append(pad_or_truncate_df(t_feats_normalized, max_item_num, zero_feature_pad, np.float32))
        
        scene_ids_list.append(scene_id)
        set_sizes_list.append(len(items_to_process))
    
    if skipped_sets_count > 0: 
        print(f"Converting ({os.path.basename(output_file)}): Skipped {skipped_sets_count} sets.")
    if overlap_warning_count > 0:
        print(f"Converting ({os.path.basename(output_file)}): Resolved {overlap_warning_count} query-target overlaps.")
    
    if not q_feats_list: 
        print(f"Warning: No data to save for {output_file}.")
        return {}
    
    # 正規化統計量も保存
    normalization_info = {
        'applied_normalization': normalization_stats,
        'split_name': os.path.basename(output_file).replace('.pkl', ''),
        'feature_count': len(q_feats_list) * max_item_num * 2
    }
    
    df_tuple = (
        np.array(q_feats_list, dtype=np.float32),      # 0: query_features
        np.array(t_feats_list, dtype=np.float32),      # 1: target_features  
        np.array(scene_ids_list, dtype=object),        # 2: scene_ids
        np.array(q_main_cats_list, dtype=np.int32),    # 3: query_categories
        np.array(t_main_cats_list, dtype=np.int32),    # 4: target_categories
        np.array(set_sizes_list, dtype=np.int32),      # 5: set_sizes
        np.array(q_ids_list, dtype=object),            # 6: query_item_ids
        np.array(t_ids_list, dtype=object),            # 7: target_item_ids
        normalization_info                             # 8: normalization_info
    )
    
    try:
        with open(output_file, 'wb') as f: 
            pickle.dump(df_tuple, f)
        print(f"✅ Saved {len(q_feats_list)} sets to {output_file} with unified normalization.")
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
    
    if len(train_data) >= 9:
        query_features, target_features, scene_ids, query_categories, target_categories, set_sizes, query_item_ids, target_item_ids, normalization_info = train_data
    elif len(train_data) >= 8:
        query_features, target_features, scene_ids, query_categories, target_categories, set_sizes, query_item_ids, target_item_ids = train_data
    else:
        print(f"Error: Unexpected data format in {train_path}")
        return
    
    print(f"Loaded training data: {len(query_features)} sets")
    
    # IQON3000の7カテゴリID (1-7)
    active_main_cat_ids = list(range(1, 18))  # 1-17
    
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
    
    category_names = {
        1: "outerwear", 2: "tops", 3: "dresses", 4: "pants", 5: "skirts",
        6: "shoes", 7: "bags", 8: "hats", 9: "watches", 10: "accessories",
        11: "fashion_goods", 12: "wallets", 13: "legwear", 
        14: "underwear", 15: "beauty", 16: "glasses", 17: "others"
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

# def process_deepfurniture(image_dir, annotations_json, furnitures_jsonl, output_dir, batch_size=32):
#     """修正版DeepFurniture処理 - 重複除去と正規化統一"""
#     print(f"Processing DeepFurniture dataset to 11 categories")
#     print("Key improvements: Proper duplicate removal, unified normalization")
#     print(f"  Images: {image_dir}")
#     print(f"  Annotations: {annotations_json}")
#     print(f"  Furnitures: {furnitures_jsonl}")
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 家具メタデータの読み込み
#     furniture_to_category = load_jsonl_mapping(Path(furnitures_jsonl), "furniture_id", "category_id")
    
#     # デバイスとモデルの設定
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#     processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#     model = model.to(device)
#     model.eval()
    
#     # 画像ファイルの検索と処理
#     image_files = []
#     for ext in ("*.jpg", "*.png", "*.jpeg"):
#         image_files.extend(Path(image_dir).rglob(ext))
    
#     valid_images = [img for img in image_files if img.stem in furniture_to_category]
#     print(f"Found {len(valid_images)} valid images")
    
#     # 重複除去のための追跡
#     processed_furniture_ids = set()
#     duplicate_count = 0
    
#     # 特徴量抽出
#     all_features = []
#     all_furniture_ids = []
    
#     with torch.no_grad():
#         for i in tqdm(range(0, len(valid_images), batch_size), desc="Extracting DeepFurniture features"):
#             batch_paths = valid_images[i:i+batch_size]
            
#             # 重複チェック付きでバッチを処理
#             filtered_batch_paths = []
#             for path in batch_paths:
#                 furniture_id = path.stem
#                 if furniture_id not in processed_furniture_ids:
#                     processed_furniture_ids.add(furniture_id)
#                     filtered_batch_paths.append(path)
#                 else:
#                     duplicate_count += 1
            
#             if not filtered_batch_paths:
#                 continue
            
#             try:
#                 pil_images = [Image.open(p).convert("RGB") for p in filtered_batch_paths]
#                 inputs = processor(images=pil_images, return_tensors="pt", padding=True)
#                 inputs = {k: v.to(device) for k, v in inputs.items()}
                
#                 output = model.get_image_features(**inputs)
#                 # 統一された正規化処理
#                 output_normalized = normalize_features_unified(output)
#                 batch_feats = output_normalized.cpu().numpy()
                
#                 all_features.append(batch_feats)
#                 all_furniture_ids.extend([p.stem for p in filtered_batch_paths])
                
#             except Exception as e:
#                 print(f"Error processing batch: {e}")
#                 continue
    
#     if duplicate_count > 0:
#         print(f"Removed {duplicate_count} duplicate furniture items")
    
#     if not all_features:
#         print("No features extracted, exiting")
#         return
    
#     all_features = np.vstack(all_features)
#     print(f"Extracted features: {all_features.shape}")
    
#     # シーンアノテーションの読み込みとシーン構築
#     with open(annotations_json, 'r') as f:
#         annotations = json.load(f)
    
#     scenes = {}
#     category_ids_in_scenes = set()
    
#     for scene_record in tqdm(annotations, desc="Building DeepFurniture scenes"):
#         scene_id = scene_record.get("scene", {}).get("sceneTaskID")
#         if not scene_id:
#             continue
        
#         # 重複除去: 同じfurniture_idは1つのみ保持
#         scene_items_unique = {}
        
#         for instance in scene_record.get("instances", []):
#             furniture_id = str(instance.get("identityID"))
#             category_id = instance.get("categoryID")
    
#             if not furniture_id or category_id is None:
#                 continue
 
#             # 重複チェック: 同じfurniture_idは最初のもののみ保持
#             if furniture_id not in scene_items_unique:
#                 if furniture_id in all_furniture_ids:
#                     idx = all_furniture_ids.index(furniture_id)
#                     scene_items_unique[furniture_id] = (all_features[idx], category_id)

#         # 有効なシーンの構築（4個以上のユニークアイテム）
#         if len(scene_items_unique) >= 4:
#             item_ids_final = list(scene_items_unique.keys())
#             features_final = np.array([v[0] for v in scene_items_unique.values()])
#             categories_final = np.array([v[1] for v in scene_items_unique.values()])
          
#             scenes[str(scene_id)] = {
#                 "features": features_final,
#                 "category_ids": categories_final,
#                 "item_ids": np.array(item_ids_final, dtype=object)
#             }
#             category_ids_in_scenes.update(categories_final)
    
#     print(f"Built {len(scenes)} valid DeepFurniture scenes")
#     print(f"Found category IDs in scenes: {sorted(list(category_ids_in_scenes))}")
    
#     # シーンをDeepFurniture形式に変換
#     convert_scenes_to_deepfurniture_format_fixed(scenes, output_dir)
    
#     # カテゴリ中心の計算（11カテゴリ用）
#     compute_deepfurniture_category_centers_fixed(output_dir)
    
#     # カテゴリマッピングの保存
#     with open(os.path.join(output_dir, 'category_mapping.json'), 'w', encoding='utf-8') as f:
#         json.dump({
#             'categories': DEEPFURNITURE_CATEGORIES
#         }, f, ensure_ascii=False, indent=2)
    
#     print(f"DeepFurniture processing complete. Files saved to {output_dir}")

def process_deepfurniture_with_inclusion_removal(image_dir, annotations_json, furnitures_jsonl, output_dir, batch_size=32, apply_inclusion_removal=True):
    """
    包含関係除去を統合したDeepFurniture処理
    1. シーン構築
    2. 包含関係除去（オプション）
    3. 特徴量抽出
    4. 分割・保存
    """
    print(f"🔥 DeepFurniture processing with inclusion relationship removal")
    print(f"  Images: {image_dir}")
    print(f"  Annotations: {annotations_json}")  
    print(f"  Furnitures: {furnitures_jsonl}")
    print(f"  Apply inclusion removal: {apply_inclusion_removal}")

    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: シーンアノテーションの読み込みとシーン構築
    print(f"\n📖 Step 1: シーンアノテーション読み込み")
    
    with open(annotations_json, 'r') as f:
        annotations = json.load(f)
\
    print(f"総アノテーション数: {len(annotations)}")
    
    # 家具メタデータの読み込み
    furniture_to_category = load_jsonl_mapping(Path(furnitures_jsonl), "furniture_id", "category_id")
    
    # 初期シーン構築（特徴量抽出前）
    raw_scenes = {}
    category_ids_in_scenes = set()
    
    for scene_record in tqdm(annotations, desc="Raw scenes building"):
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
                # 画像ファイルが存在するかチェック
                if furniture_id in furniture_to_category:
                    scene_items_unique[furniture_id] = category_id
        
        # 有効なシーンの構築（4個以上のユニークアイテム）
        if len(scene_items_unique) >= 4:
            # (furniture_id, category_id)のタプルリストとして保存
            raw_scenes[str(scene_id)] = [(fid, cid) for fid, cid in scene_items_unique.items()]
            category_ids_in_scenes.update(scene_items_unique.values())
    
    print(f"構築された生シーン数: {len(raw_scenes)}")
    print(f"発見されたカテゴリID: {sorted(list(category_ids_in_scenes))}")
    
    # Step 2: 包含関係除去（オプション）
    if apply_inclusion_removal:
        print(f"\n✂️  Step 2: 包含関係除去実行")
        
        remover = InclusionRelationshipRemover(debug_mode=True)
        filtered_scenes = remover.remove_inclusion_relationships(
            raw_scenes, 
            min_items=4, 
            debug_limit=15
        )
        
        print(f"包含関係除去結果:")
        print(f"  除去前: {len(raw_scenes)} シーン")
        print(f"  除去後: {len(filtered_scenes)} シーン")
        print(f"  除去されたシーン: {len(raw_scenes) - len(filtered_scenes)}")
        print(f"  除去率: {(len(raw_scenes) - len(filtered_scenes))/len(raw_scenes)*100:.1f}%")
        
        # 包含関係除去結果を中間保存
        with open(os.path.join(output_dir, 'inclusion_removal_report.json'), 'w') as f:
            json.dump({
                'original_scenes': len(raw_scenes),
                'filtered_scenes': len(filtered_scenes),
                'removed_scenes': len(raw_scenes) - len(filtered_scenes),
                'removal_rate': (len(raw_scenes) - len(filtered_scenes))/len(raw_scenes)*100
            }, f, indent=2)
        
        scenes_to_process = filtered_scenes
    else:
        print(f"\n⏭️  Step 2: 包含関係除去をスキップ")
        scenes_to_process = raw_scenes
    
    # Step 3: 特徴量抽出
    print(f"\n🎨 Step 3: CLIP特徴量抽出")
    
    # デバイスとモデルの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    model.eval()
    
    # 画像ファイルの検索と処理
    image_files = []
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        image_files.extend(Path(image_dir).rglob(ext))
    
    valid_images = [img for img in image_files if img.stem in furniture_to_category]
    print(f"発見された有効画像数: {len(valid_images)}")
    
    # フィルタ済みシーンで実際に使用される家具IDのみを処理
    used_furniture_ids = set()
    for scene_items in scenes_to_process.values():
        for furniture_id, _ in scene_items:
            used_furniture_ids.add(furniture_id)
    
    # 使用される画像のみに絞り込み
    relevant_images = [img for img in valid_images if img.stem in used_furniture_ids]
    print(f"実際に使用される画像数: {len(relevant_images)}")
    
    # 重複除去のための追跡
    processed_furniture_ids = set()
    duplicate_count = 0
    
    # 特徴量抽出
    all_features = []
    all_furniture_ids = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(relevant_images), batch_size), desc="CLIP特徴量抽出"):
            batch_paths = relevant_images[i:i+batch_size]
            
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
                print(f"バッチ処理エラー: {e}")
                continue
    
    if duplicate_count > 0:
        print(f"除去された重複家具アイテム: {duplicate_count}")
    
    if not all_features:
        print("特徴量が抽出されませんでした。処理を終了します。")
        return
    
    all_features = np.vstack(all_features)
    print(f"抽出された特徴量: {all_features.shape}")
    
    # Step 4: シーンの再構築（特徴量付き）
    print(f"\n🏗️  Step 4: 特徴量付きシーン再構築")
    
    final_scenes = {}
    
    for scene_id, scene_items in tqdm(scenes_to_process.items(), desc="特徴量統合"):
        scene_features = []
        scene_categories = []
        scene_furniture_ids = []
        
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
    
    print(f"最終シーン数: {len(final_scenes)}")
    
    # Step 5: DeepFurniture形式への変換と保存
    print(f"\n💾 Step 5: DeepFurniture形式変換・保存")
    
    convert_scenes_to_deepfurniture_format_fixed(final_scenes, output_dir)
    
    # カテゴリ中心の計算
    compute_deepfurniture_category_centers_fixed(output_dir)
    
    # カテゴリマッピングの保存
    with open(os.path.join(output_dir, 'category_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'categories': DEEPFURNITURE_CATEGORIES
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✅ DeepFurniture処理完了（包含関係除去統合版）")
    print(f"   結果保存先: {output_dir}")
    
    return final_scenes



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


class InclusionRelationshipRemover:
    """包含関係を除去するクラス（直接統合版）"""
    
    def __init__(self, debug_mode=True):
        self.debug_mode = debug_mode
        
    def remove_inclusion_relationships(self, scenes_dict, min_items=4, debug_limit=10):
        """
        包含関係を除去する統合関数
        
        Args:
            scenes_dict: シーンID -> アイテムリストの辞書
            min_items: 最小アイテム数
            debug_limit: デバッグ出力する例数
            
        Returns:
            filtered_scenes: 包含関係を除去したシーン辞書
        """
        
        if self.debug_mode:
            print(f"🔍 包含関係除去開始")
            print(f"入力シーン数: {len(scenes_dict)}")
        
        # Step 1: シーンをfurniture_idのセットに変換
        scene_to_furniture_sets = {}
        
        for scene_id, scene_data in scenes_dict.items():
            furniture_ids = self._extract_furniture_ids(scene_data)
            
            # 最小アイテム数チェック
            if len(furniture_ids) >= min_items:
                scene_to_furniture_sets[scene_id] = set(furniture_ids)
        
        if self.debug_mode:
            print(f"最小アイテム数({min_items})以上のシーン: {len(scene_to_furniture_sets)}")
        
        # Step 2: 包含関係を検出
        inclusion_pairs = self._detect_inclusion_pairs(scene_to_furniture_sets, debug_limit)
        
        if not inclusion_pairs:
            if self.debug_mode:
                print("✅ 包含関係は検出されませんでした")
            return scenes_dict
        
        # Step 3: 除去対象シーンを決定
        scenes_to_remove = self._determine_removal_targets(inclusion_pairs, scene_to_furniture_sets)
        
        # Step 4: 除去実行
        filtered_scenes = self._apply_removal(scenes_dict, scenes_to_remove)
        
        # Step 5: 除去後の検証
        if self.debug_mode:
            self._validate_removal(filtered_scenes, min_items)
        
        return filtered_scenes
    
    def _extract_furniture_ids(self, scene_data):
        """シーンデータからfurniture_idを抽出"""
        
        furniture_ids = []
        
        if isinstance(scene_data, dict):
            # 辞書形式の場合
            if 'item_ids' in scene_data:
                furniture_ids = list(scene_data['item_ids'])
            elif 'features' in scene_data and 'item_ids' in scene_data:
                furniture_ids = list(scene_data['item_ids'])
            else:
                # その他の辞書キーを探索
                for key, value in scene_data.items():
                    if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                        if isinstance(value[0], str):
                            furniture_ids = list(value)
                            break
        
        elif isinstance(scene_data, list):
            # リスト形式の場合
            for item in scene_data:
                if isinstance(item, tuple) and len(item) >= 1:
                    furniture_ids.append(str(item[0]))
                elif isinstance(item, str):
                    furniture_ids.append(item)
                elif isinstance(item, dict) and 'furniture_id' in item:
                    furniture_ids.append(str(item['furniture_id']))
        
        # 重複除去と文字列化
        return list(set([str(fid) for fid in furniture_ids if fid]))
    
    def _detect_inclusion_pairs(self, scene_to_furniture_sets, debug_limit):
        """包含関係のペアを検出"""
        
        inclusion_pairs = []
        scene_ids = list(scene_to_furniture_sets.keys())
        
        if self.debug_mode:
            print(f"\n🔍 包含関係検出中... ({len(scene_ids)}シーン)")
        
        debug_count = 0
        
        # 全ペアをチェック（効率化可能だが、まず正確性を重視）
        for i, scene_id1 in enumerate(tqdm(scene_ids, desc="包含関係チェック", disable=not self.debug_mode)):
            set1 = scene_to_furniture_sets[scene_id1]
            
            for scene_id2 in scene_ids[i+1:]:
                set2 = scene_to_furniture_sets[scene_id2]
                
                # 包含関係チェック
                if set1.issubset(set2) and set1 != set2:
                    inclusion_pairs.append((scene_id1, scene_id2, 'subset'))
                    
                    if self.debug_mode and debug_count < debug_limit:
                        print(f"\n包含関係発見: {scene_id1} ⊆ {scene_id2}")
                        print(f"  {scene_id1}({len(set1)}個): {sorted(list(set1))[:5]}...")
                        print(f"  {scene_id2}({len(set2)}個): {sorted(list(set2))[:5]}...")
                        print(f"  差分({len(set2-set1)}個): {sorted(list(set2-set1))[:3]}...")
                        debug_count += 1
                        
                elif set2.issubset(set1) and set1 != set2:
                    inclusion_pairs.append((scene_id2, scene_id1, 'subset'))
                    
                    if self.debug_mode and debug_count < debug_limit:
                        print(f"\n包含関係発見: {scene_id2} ⊆ {scene_id1}")
                        print(f"  {scene_id2}({len(set2)}個): {sorted(list(set2))[:5]}...")
                        print(f"  {scene_id1}({len(set1)}個): {sorted(list(set1))[:5]}...")
                        print(f"  差分({len(set1-set2)}個): {sorted(list(set1-set2))[:3]}...")
                        debug_count += 1
        
        if self.debug_mode:
            print(f"\n📊 包含関係検出結果: {len(inclusion_pairs)}ペア")
        
        return inclusion_pairs
    
    def _determine_removal_targets(self, inclusion_pairs, scene_to_furniture_sets):
        """除去対象シーンを決定"""
        
        # 戦略: より小さいシーン（部分集合）を除去
        scenes_to_remove = set()
        
        for smaller_scene, larger_scene, relation_type in inclusion_pairs:
            scenes_to_remove.add(smaller_scene)
        
        if self.debug_mode:
            print(f"\n📋 除去戦略: 小さいシーン（部分集合）を除去")
            print(f"除去対象シーン数: {len(scenes_to_remove)}")
            
            # 除去対象の例を表示
            if scenes_to_remove:
                print(f"除去対象例:")
                for i, scene_id in enumerate(list(scenes_to_remove)[:5]):
                    furniture_set = scene_to_furniture_sets[scene_id]
                    print(f"  {scene_id}: {len(furniture_set)}個のアイテム")
        
        return scenes_to_remove
    
    def _apply_removal(self, scenes_dict, scenes_to_remove):
        """除去を実行"""
        
        original_count = len(scenes_dict)
        
        filtered_scenes = {
            scene_id: scene_data 
            for scene_id, scene_data in scenes_dict.items() 
            if scene_id not in scenes_to_remove
        }
        
        if self.debug_mode:
            removed_count = original_count - len(filtered_scenes)
            print(f"\n✂️  除去実行完了")
            print(f"  元のシーン数: {original_count}")
            print(f"  除去されたシーン: {removed_count}")
            print(f"  残存シーン数: {len(filtered_scenes)}")
            print(f"  除去率: {removed_count/original_count*100:.1f}%")
        
        return filtered_scenes
    
    def _validate_removal(self, filtered_scenes, min_items):
        """除去後の検証"""
        
        print(f"\n🔍 除去後の検証...")
        
        # 再度包含関係をチェック
        verification_remover = InclusionRelationshipRemover(debug_mode=False)
        scene_to_furniture_sets = {}
        
        for scene_id, scene_data in filtered_scenes.items():
            furniture_ids = verification_remover._extract_furniture_ids(scene_data)
            if len(furniture_ids) >= min_items:
                scene_to_furniture_sets[scene_id] = set(furniture_ids)
        
        verification_pairs = verification_remover._detect_inclusion_pairs(scene_to_furniture_sets, debug_limit=3)
        
        if len(verification_pairs) == 0:
            print("✅ 包含関係が完全に除去されました！")
        else:
            print(f"⚠️  まだ{len(verification_pairs)}個の包含関係が残っています")
            # 残存する包含関係の例を表示
            for i, (smaller, larger, _) in enumerate(verification_pairs[:3]):
                set1 = scene_to_furniture_sets[smaller]
                set2 = scene_to_furniture_sets[larger]
                print(f"  残存例{i+1}: {smaller}({len(set1)}) ⊆ {larger}({len(set2)})")


def test_inclusion_removal_on_real_data(annotations_json, limit_scenes=None):
    """
    実際のDeepFurnitureデータで包含関係除去をテスト
    
    Args:
        annotations_json: アノテーションファイルのパス
        limit_scenes: テストするシーン数の上限（Noneの場合は全データ）
    """
    
    print("🧪 実データでの包含関係除去テスト開始")
    print(f"ファイル: {annotations_json}")
    if limit_scenes:
        print(f"テスト対象: 最初の{limit_scenes}シーン")
    else:
        print(f"テスト対象: 全シーン")
    
    # 1. アノテーションファイル読み込み
    with open(annotations_json, 'r') as f:
        annotations = json.load(f)
    
    print(f"総アノテーション数: {len(annotations)}")
    
    # 2. 全シーンを構築（制限なしまたは制限あり）
    test_scenes = {}
    processed_count = 0
    
    for scene_record in tqdm(annotations, desc="シーン構築中"):
        if limit_scenes and processed_count >= limit_scenes:
            break
            
        scene_id = scene_record.get("scene", {}).get("sceneTaskID")
        if not scene_id:
            continue
        
        # アイテム収集
        scene_items = []
        for instance in scene_record.get("instances", []):
            furniture_id = str(instance.get("identityID"))
            category_id = instance.get("categoryID")
            
            if furniture_id and category_id is not None:
                scene_items.append((furniture_id, category_id))
        
        if len(scene_items) >= 4:
            test_scenes[str(scene_id)] = scene_items
            processed_count += 1
    
    print(f"テスト用シーン構築完了: {len(test_scenes)}シーン")
    
    remover = InclusionRelationshipRemover(debug_mode=True)
    filtered_scenes = remover.remove_inclusion_relationships(test_scenes, min_items=4, debug_limit=15)
    
    # 4. 結果サマリー
    print(f"\n📊 テスト結果サマリー:")
    print(f"  テスト対象シーン: {len(test_scenes)}")
    print(f"  除去後シーン: {len(filtered_scenes)}")
    print(f"  除去されたシーン: {len(test_scenes) - len(filtered_scenes)}")
    print(f"  除去率: {(len(test_scenes) - len(filtered_scenes))/len(test_scenes)*100:.1f}%")
    
    return test_scenes, filtered_scenes


# =============================================================================
# メイン関数
# =============================================================================

# メイン関数も更新
def main():
    parser = argparse.ArgumentParser(description="Dataset generator with inclusion relationship removal")
    parser.add_argument('--dataset', choices=['iqon3000', 'deepfurniture'], required=True, help='Dataset type to process')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for feature extraction')
    parser.add_argument('--input-dir', type=str, help='Input directory for IQON3000 dataset')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for processed dataset')
    parser.add_argument('--user-based', action='store_true', help='Use user-based splitting (recommended)')
    
    # DeepFurniture関連の引数
    parser.add_argument('--image-dir', type=str, help='DeepFurniture image directory')
    parser.add_argument('--annotations-json', type=str, help='DeepFurniture annotations.json file')
    parser.add_argument('--furnitures-jsonl', type=str, help='DeepFurniture furnitures.jsonl file')
    parser.add_argument('--no-inclusion-removal', action='store_true', help='Disable inclusion relationship removal for DeepFurniture')
    
    args = parser.parse_args()
    
    if args.dataset == 'iqon3000':
        if not args.input_dir:
            parser.error("--input-dir is required for IQON3000 dataset")
        
        if args.user_based:
            print("🔥 Processing IQON3000 dataset: 7 categories (USER-BASED SPLIT)")
            print("   Key improvements:")
            print("   - USER-BASED splitting: No user overlap between splits")
            print("   - Split FIRST, then normalize")
            print("   - Use training data statistics only")
            print("   - Prevent information leakage")
            process_iqon3000(args.input_dir, args.output_dir, args.batch_size)
        else:
            print("🔥 Processing IQON3000 dataset: 7 categories (SCENE-BASED SPLIT)")
            print("   ⚠️ Warning: This may cause user overlap between splits")
            process_iqon3000(args.input_dir, args.output_dir, args.batch_size)

    elif args.dataset == 'deepfurniture':
        if not all([args.image_dir, args.annotations_json, args.furnitures_jsonl]):
            parser.error("--image-dir, --annotations-json, and --furnitures-jsonl are required for DeepFurniture dataset")
        
        apply_inclusion_removal = not args.no_inclusion_removal
        
        if apply_inclusion_removal:
            print("🔥 Processing DeepFurniture dataset: WITH inclusion relationship removal")
        else:
            print("🔥 Processing DeepFurniture dataset: WITHOUT inclusion relationship removal")
        
        process_deepfurniture_with_inclusion_removal(
            image_dir=args.image_dir,
            annotations_json=args.annotations_json,
            furnitures_jsonl=args.furnitures_jsonl,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            apply_inclusion_removal=apply_inclusion_removal
        )


if __name__ == "__main__":
    main()