# enhanced_util.py - Complete evaluation pipeline with all visualizations
import os
import json
import time
import pickle
import gzip
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

# Import visualization libraries with fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib not available, visualizations disabled")

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] sklearn not available, some visualizations disabled")

# CORRECTED Dataset configurations with proper category mappings
DATASET_CONFIGS = {
    'DeepFurniture': {
        'num_categories': 11,
        'category_range': (1, 11),
        'category_names': {
            1: "Chairs",
            2: "Tables", 
            3: "Storage",
            4: "Beds",
            5: "Sofas",
            6: "Lighting",
            7: "Decor",
            8: "Electronics",
            9: "Kitchen",
            10: "Outdoor",
            11: "Others"
        },
        'image_paths': {
            'furniture': 'data/DeepFurniture/furnitures',
            'scenes': 'data/DeepFurniture/scenes'
        }
    },
    'IQON3000': {
        'num_categories': 7,  # CORRECTED: 7 categories instead of 11
        'category_range': (1, 7),  # CORRECTED: 1-7 instead of 1-11
        'category_names': {
            1: "インナー系",
            2: "ボトムス系", 
            3: "シューズ系",
            4: "バッグ系",
            5: "アクセサリー系",
            6: "帽子",
            7: "トップス系"  # CORRECTED: simplified category names
        },
        'image_paths': {
            'furniture': 'data/IQON3000',  # Two-level structure: setId/itemId
            'scenes': 'data/IQON3000'
        }
    }
}

def detect_dataset_from_generator(test_generator):
    """Detect dataset type from data generator"""
    try:
        if hasattr(test_generator, 'dataset_name'):
            return test_generator.dataset_name
        if hasattr(test_generator, 'data_path'):
            if 'DeepFurniture' in test_generator.data_path:
                return 'DeepFurniture'
            elif 'IQON3000' in test_generator.data_path:
                return 'IQON3000'
        print("[WARN] Could not detect dataset type, defaulting to DeepFurniture")
        return 'DeepFurniture'
    except Exception as e:
        print(f"[WARN] Dataset detection failed: {e}, defaulting to DeepFurniture")
        return 'DeepFurniture'

# ------------------------------------------------------------------
# 🖼️ Enhanced real image loading utilities
# ------------------------------------------------------------------
def load_real_furniture_image(item_id: str, dataset_type: str = "DeepFurniture", thumb_size=(150, 150)):
    """Load real furniture/item image from dataset with robust path handling"""
    # デバッグ出力を削除または簡素化
    # print(f"[DEBUG] Attempting to load image. Received ID: '{item_id}' (Type: {type(item_id)}), Dataset: {dataset_type}")
    if item_id == "0" or item_id == 0 or not item_id:
        return None
    
    try:
        item_id_str = str(item_id).strip()
        if not item_id_str or item_id_str == "0":
            return None
            
        config = DATASET_CONFIGS[dataset_type]
        image_paths = config['image_paths']
        base_dir = image_paths['furniture']
        
        if dataset_type == "IQON3000":
            # IQON3000の処理（変更なし）
            possible_base_dirs = [
                base_dir, 
                "data/IQON3000", 
                "./data/IQON3000", 
                "~/SetRetrieval_WACV2026/data/IQON3000",
                "~/setRetrieval/Datasets/IQON3000",
                "IQON3000"
            ]
            
            for base in possible_base_dirs:
                expanded_base = os.path.expanduser(base)
                if not os.path.exists(expanded_base):
                    continue
                    
                try:
                    for set_dir in os.listdir(expanded_base):
                        set_path = os.path.join(expanded_base, set_dir)
                        if not os.path.isdir(set_path):
                            continue
                            
                        possible_files = [
                            f"{item_id_str}_m.jpg",
                            f"{item_id_str}_s.jpg",
                            f"{item_id_str}_l.jpg",
                            f"{item_id_str}.jpg", 
                            f"{item_id_str}.png", 
                            f"{item_id_str}.jpeg"
                        ]
                        
                        for filename in possible_files:
                            img_path = os.path.join(set_path, filename)
                            if os.path.exists(img_path):
                                try:
                                    img = Image.open(img_path).convert("RGB")
                                    # print(f"[SUCCESS] 👗 Loaded IQON3000 real image: {img_path}")
                                    img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                                    result = Image.new("RGB", thumb_size, "white")
                                    paste_x = (thumb_size[0] - img.width) // 2
                                    paste_y = (thumb_size[1] - img.height) // 2
                                    result.paste(img, (paste_x, paste_y))
                                    return result
                                except Exception as e:
                                    continue
                except Exception as e:
                    continue
                    
        else:  # DeepFurniture
            possible_base_dirs = [
                base_dir,
                "data/DeepFurniture/furnitures",
                "./data/DeepFurniture/furnitures",
                "~/SetRetrieval_WACV2026/data/DeepFurniture/furnitures",
                "DeepFurniture/furnitures",
            ]
            
            possible_extensions = [".png", ".jpg", ".jpeg"]
            
            for base in possible_base_dirs:
                expanded_base = os.path.expanduser(base)
                for ext in possible_extensions:
                    img_path = os.path.join(expanded_base, f"{item_id_str}{ext}")
                    if os.path.exists(img_path):
                        try:
                            img = Image.open(img_path).convert("RGB")
                            # print(f"[SUCCESS] 🪑 Loaded DeepFurniture real image: {img_path}")
                            img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                            result = Image.new("RGB", thumb_size, "white")
                            paste_x = (thumb_size[0] - img.width) // 2
                            paste_y = (thumb_size[1] - img.height) // 2
                            result.paste(img, (paste_x, paste_y))
                            return result
                        except Exception as e:
                            continue
        # print(f"[WARN] 🔍 Real image not found for {dataset_type} ID {item_id_str}")
            
    except Exception as e:
        print(f"[ERROR] Error loading real image {item_id}: {e}")
    
    return None

def create_category_placeholder(category_id: int, item_type: str, thumb_size=(150, 150), dataset_type: str = "DeepFurniture"):
    """Create visually appealing category-specific placeholder images"""
    
    config = DATASET_CONFIGS[dataset_type]
    category_names = config['category_names']
    
    # Category-specific colors
    if dataset_type == "DeepFurniture":
        category_colors = {
            1: "#8B4513", 2: "#D2691E", 3: "#708090", 4: "#9370DB", 5: "#20B2AA",
            6: "#FFD700", 7: "#FF69B4", 8: "#4169E1", 9: "#32CD32", 10: "#228B22", 11: "#808080"
        }
    else:  # IQON3000
        category_colors = {
            1: "#FFF8DC", 2: "#4169E1", 3: "#8B4513", 4: "#DAA520", 5: "#FF69B4", 
            6: "#2F4F4F", 7: "#98FB98", 8: "#FFB6C1", 9: "#DDA0DD", 10: "#F0E68C", 11: "#D3D3D3"
        }
    
    # Item type specific colors
    type_colors = {
        "Query": "#90EE90", "Target": "#FFB6C1", "Retrieval": "#FFFFE0", "Scene": "#87CEEB"
    }
    
    base_color = category_colors.get(category_id, "#D3D3D3")
    type_color = type_colors.get(item_type, "#F0F0F0")
    
    img = Image.new("RGB", thumb_size, base_color)
    draw = ImageDraw.Draw(img)
    
    try:
        font_large = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Add border
    border_color = "#333333"
    draw.rectangle([0, 0, thumb_size[0]-1, thumb_size[1]-1], outline=border_color, width=2)
    
    # Add type indicator background
    type_height = 25
    draw.rectangle([5, 5, thumb_size[0]-5, 5+type_height], fill=type_color, outline=border_color)
    if font_small:
        draw.text((10, 8), item_type, fill="black", font=font_small)
    
    # Add category info
    cat_name = category_names.get(category_id, "Unknown")
    text_y = thumb_size[1] // 2 - 15
    if font_large:
        draw.text((10, text_y), f"Cat {category_id}", fill="white", font=font_large)
    if font_small:
        # Truncate long category names
        display_name = cat_name[:12] + "..." if len(cat_name) > 12 else cat_name
        draw.text((10, text_y + 20), display_name, fill="white", font=font_small)
        draw.text((10, thumb_size[1] - 35), "NO IMAGE", fill="red", font=font_small)
    
    # Add decorative elements
    corner_size = 8
    draw.rectangle([thumb_size[0]-corner_size-5, thumb_size[1]-corner_size-5, 
                   thumb_size[0]-5, thumb_size[1]-5], fill="white", outline=border_color)
    
    return img

def safe_load_furniture_image(item_id: str, furn_root: str, dataset_type: str = "DeepFurniture", 
                            thumb_size=(150, 150), item_type: str = "Item", category_id: int = 1):
    """Main function: Try to load real image first, then enhanced placeholder"""
    real_img = load_real_furniture_image(item_id, dataset_type, thumb_size)
    if real_img is not None:
        return real_img
    return create_category_placeholder(category_id, item_type, thumb_size, dataset_type)

def load_scene_image(scene_id: str, scene_root: str, dataset_type: str = "DeepFurniture", thumb_size=(400, 300)):
    """
    DeepFurniture用のシーン画像読み込み
    data/DeepFurniture/scenes/[scene_id]/image.jpg から読み込み
    """
    try:
        # 可能なベースパス
        possible_base_dirs = [
            "data/DeepFurniture/scenes",
            "./data/DeepFurniture/scenes",
            "~/SetRetrieval_WACV2026/data/DeepFurniture/scenes",
            "/home/yamazono/SetRetrieval_WACV2026/data/DeepFurniture/scenes"
        ]
        
        for base_dir in possible_base_dirs:
            expanded_base = os.path.expanduser(base_dir)
            scene_image_path = os.path.join(expanded_base, str(scene_id), "image.jpg")
            
            if os.path.exists(scene_image_path):
                try:
                    img = Image.open(scene_image_path).convert("RGB")
                    print(f"[SUCCESS] 🏠 Loaded DeepFurniture scene: {scene_image_path}")
                    img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                    result = Image.new("RGB", thumb_size, "white")
                    paste_x = (thumb_size[0] - img.width) // 2
                    paste_y = (thumb_size[1] - img.height) // 2
                    result.paste(img, (paste_x, paste_y))
                    return result
                except Exception as e:
                    print(f"[ERROR] Failed to load scene {scene_image_path}: {e}")
                    continue
        
        print(f"[WARN] 🔍 Scene image not found for scene ID: {scene_id}")
        
    except Exception as e:
        print(f"[ERROR] Error loading scene image {scene_id}: {e}")
    
    # フォールバック: プレースホルダー画像を作成
    return create_scene_placeholder(scene_id, thumb_size)

def create_scene_placeholder(scene_id: str, thumb_size=(400, 300)):
    """シーン用プレースホルダー画像を作成"""
    img = Image.new("RGB", thumb_size, "#E6F3FF")
    draw = ImageDraw.Draw(img)
    
    try:
        font_large = ImageFont.truetype("arial.ttf", 24)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # 枠線を描画
    draw.rectangle([0, 0, thumb_size[0]-1, thumb_size[1]-1], outline="#2E86C1", width=3)
    
    # テキストを描画
    text_lines = [
        "DeepFurniture Scene",
        f"ID: {scene_id}",
        "Image not found"
    ]
    
    y_offset = thumb_size[1] // 2 - 40
    for i, line in enumerate(text_lines):
        font = font_large if i == 0 else font_small
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (thumb_size[0] - text_width) // 2
        draw.text((x, y_offset), line, fill="#2E86C1", font=font)
        y_offset += 30 if i == 0 else 25
    
    return img

# ------------------------------------------------------------------
# Data collection and processing
# ------------------------------------------------------------------
def gather_test_items_fixed(test_generator):
    """Fixed version of gather_test_items with robust error handling AND detailed logging"""
    print("[INFO] Starting fixed test item collection...")
    
    items = []
    successful_batches = 0
    total_batches = 0
    
    try:
        max_batches = len(test_generator)
        print(f"[INFO] Generator length: {max_batches}, attempting to process all batches.")
    except TypeError:
        max_batches = float('inf')
        print("[WARN] Generator length not available, processing until iterator is exhausted.")

    if hasattr(test_generator, 'on_epoch_end'):
        test_generator.on_epoch_end()
    
    iterator = iter(test_generator)
    
    while total_batches < max_batches:
        # print(f"[DEBUG] Loop START: total_batches={total_batches}, successful_batches={successful_batches}")
        try:
            batch_data = next(iterator)
            total_batches += 1
            
            if batch_data is None:
                print("[DEBUG] batch_data is None. Skipping.")
                continue
            
            if not isinstance(batch_data, (list, tuple)) or len(batch_data) < 2:
                print(f"[DEBUG] Invalid batch_data format. Type: {type(batch_data)}. Skipping.")
                continue
            
            features_tuple, _ = batch_data
            
            if features_tuple is None or len(features_tuple) < 7:
                print("[DEBUG] features_tuple is invalid. Skipping.")
                continue
            
            Xconcat = features_tuple[0]
            if hasattr(Xconcat, 'numpy'): Xconcat = Xconcat.numpy()

            if len(Xconcat.shape) != 3:
                print(f"[DEBUG] Invalid Xconcat shape: {Xconcat.shape}. Skipping.")
                continue
            
            half = Xconcat.shape[0] // 2
            if half == 0:
                print("[DEBUG] Batch half-size is 0. Skipping.")
                continue
            
            # (item extraction logic remains the same)
            catQ, catP, q_ids, t_ids = (features_tuple[i] for i in [3, 4, 5, 6])
            if hasattr(catQ, 'numpy'): catQ = catQ.numpy()
            if hasattr(catP, 'numpy'): catP = catP.numpy()
            if hasattr(q_ids, 'numpy'): q_ids = q_ids.numpy()
            if hasattr(t_ids, 'numpy'): t_ids = t_ids.numpy()

            for i in range(half):
                item = (Xconcat[i], catQ[i], Xconcat[i + half], catP[i], q_ids[i], t_ids[i])
                items.append(item)
            
            successful_batches += 1
            # print(f"[DEBUG] Successfully processed batch. Items collected so far: {len(items)}")

        except StopIteration:
            print("[DEBUG] 'StopIteration' caught. This means the generator is exhausted. Breaking loop.")
            break
        except Exception as e:
            print(f"\n--- !!! UNEXPECTED ERROR in gather_test_items_fixed !!! ---")
            print(f"An error occurred while processing batch number: {total_batches}")
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc()
            print(f"-------------------------------------------------------------\n")
            break

    # This final print is outside the loop
    # print(f"[INFO] Collection loop finished. Collected {len(items)} items from {successful_batches} successful batches out of {total_batches} total batches attempted.")
    return items if len(items) > 0 else None

def build_item_feature_dict(model, test_gen) -> Dict[str, np.ndarray]:
    """Build a dictionary of item_id → feature from all test batches."""
    vecs: Dict[str, np.ndarray] = {}
    
    try:
        batch_count = 0
        for batch_data in test_gen:
            if batch_count >= 10:  # Limit to avoid memory issues
                break
                
            if batch_data is None or len(batch_data) < 2:
                continue
                
            features_tuple = batch_data[0]
            if features_tuple is None or len(features_tuple) < 7:
                continue
                
            concat_ft, _, _, _, _, q_ids, t_ids = features_tuple[:7]
            
            cf = concat_ft.numpy() if hasattr(concat_ft, 'numpy') else concat_ft
            q_ids_np = q_ids.numpy() if hasattr(q_ids, 'numpy') else q_ids
            t_ids_np = t_ids.numpy() if hasattr(t_ids, 'numpy') else t_ids
            
            B2, N, _ = cf.shape
            half = B2 // 2

            # Process both query and target items
            for b in range(half):
                for vec, iid in zip(cf[b], q_ids_np[b]):
                    if iid > 0: 
                        vecs[str(int(iid))] = vec
            
            for b in range(half, B2):
                for vec, iid in zip(cf[b], t_ids_np[b-half]):
                    if iid > 0: 
                        vecs[str(int(iid))] = vec
            
            batch_count += 1
                        
    except Exception as e:
        print(f"[ERROR] Failed to build feature dict: {e}")
        
    print(f"[INFO] Built feature dict: {len(vecs)} items")
    return vecs

def find_topk_similar_items_by_euclidean(query_vec: np.ndarray,
                                         item_dict: Dict[str, np.ndarray],
                                         k: int = 3,
                                         exclude: Set[str] = None) -> List[Tuple[str, float]]:
    """Find top-k similar items using Euclidean distance"""
    exclude = exclude or set()
    
    # クエリベクトルを正規化
    query_norm = query_vec / np.linalg.norm(query_vec)
    
    similarities = []
    for iid, v in item_dict.items():
        if iid not in exclude and iid != "0":
            # アイテムベクトルを正規化
            v_norm = v / np.linalg.norm(v)
            # コサイン類似度を計算（内積）
            cosine_sim = float(np.dot(query_norm, v_norm))
            similarities.append((iid, cosine_sim))
    
    # 類似度の高い順（降順）でソート
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

def create_item_placeholder(item_id: str, item_type: str, thumb_size=(150, 150)):
    """個別アイテム用のプレースホルダー画像"""
    type_colors = {
        "Query": "#90EE90",
        "Target": "#FFB6C1", 
        "Pred-1": "#FFE4B5",
        "Pred-2": "#E0E0E0",
        "Pred-3": "#F0F0F0"
    }
    
    bg_color = type_colors.get(item_type, "#F5F5F5")
    
    img = Image.new("RGB", thumb_size, bg_color)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # 枠線
    draw.rectangle([0, 0, thumb_size[0]-1, thumb_size[1]-1], outline="black", width=2)
    
    # テキスト
    text_lines = [item_type, f"ID: {item_id}"]
    y_start = thumb_size[1] // 2 - 20
    
    for i, line in enumerate(text_lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (thumb_size[0] - text_width) // 2
        y = y_start + i * 20
        draw.text((x, y), line, fill="black", font=font)
    
    return img

# ------------------------------------------------------------------
# Enhanced Visualization functions
# ------------------------------------------------------------------
def create_retrieval_collage(scene_img: Image.Image,
                             query_imgs: List[Image.Image],
                             target_imgs: List[Image.Image],
                             topk: List[List[Tuple[Image.Image, float]]],
                             save_path: str,
                             thumb=(150, 150),
                             dataset_type: str = "DeepFurniture",
                             scene_id: str = "unknown"):
    """
    シーンベースのレイアウトで修正したcreate_retrieval_collage
    全ての未定義変数エラーを解決した完全版
    """
    thumb_size = thumb
    scene_size = (300, 300)
    
    # レイアウト計算 - 未定義変数を修正
    max_items = max(len(query_imgs), len(target_imgs)) if query_imgs or target_imgs else 3
    top_k = 3
    
    scene_width = scene_size[0]
    items_width = max_items * thumb_size[0]
    
    canvas_width = scene_width + items_width + 20
    canvas_height = max(scene_size[1], (1 + 1 + top_k) * thumb_size[1]) + 80
    
    # キャンバス作成
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    
    try:
        font_title = ImageFont.truetype("arial.ttf", 16)
        font_label = ImageFont.truetype("arial.ttf", 12)
    except:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
    
    # タイトル描画
    title = f"DeepFurniture Scene Retrieval - {scene_id}"
    draw.text((10, 10), title, fill="black", font=font_title)
    
    y_offset = 50
    
    # 1. シーン画像（左側に1つだけ）
    if scene_img:
        # scene_imgのサイズ調整
        scene_resized = scene_img.copy()
        scene_resized.thumbnail(scene_size, Image.Resampling.LANCZOS)
        canvas.paste(scene_resized, (10, y_offset))
        draw.text((15, y_offset - 20), "Scene", fill="black", font=font_label)
    
    # アイテム表示開始位置
    items_x_start = scene_width + 30
    
    # 2. クエリアイテム行
    y_query = y_offset
    draw.text((items_x_start, y_query - 20), "Query Items:", fill="green", font=font_label)
    for i, img in enumerate(query_imgs[:max_items]):
        x = items_x_start + i * thumb_size[0]
        if img:
            img_resized = img.copy()
            img_resized.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            canvas.paste(img_resized, (x, y_query))
        # クエリラベル
        draw.rectangle((x, y_query + thumb_size[1] - 20, x + 40, y_query + thumb_size[1]), fill="green")
        draw.text((x + 5, y_query + thumb_size[1] - 18), f"Q{i+1}", fill="white", font=font_label)
    
    # 3. ターゲットアイテム行
    y_target = y_offset + thumb_size[1]
    draw.text((items_x_start, y_target - 20), "Target Items:", fill="red", font=font_label)
    for i, img in enumerate(target_imgs[:max_items]):
        x = items_x_start + i * thumb_size[0]
        if img:
            img_resized = img.copy()
            img_resized.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            canvas.paste(img_resized, (x, y_target))
        # ターゲットラベル
        draw.rectangle((x, y_target + thumb_size[1] - 20, x + 40, y_target + thumb_size[1]), fill="red")
        draw.text((x + 5, y_target + thumb_size[1] - 18), f"T{i+1}", fill="white", font=font_label)
    
    # 4. 予測結果行
    for k in range(top_k):
        y_pred = y_offset + (k + 2) * thumb_size[1]
        draw.text((items_x_start, y_pred - 20), f"Top-{k+1} Predictions:", fill="blue", font=font_label)
        
        for i in range(max_items):
            x = items_x_start + i * thumb_size[0]
            
            if i < len(topk) and k < len(topk[i]):
                img, dist = topk[i][k]
                if img:
                    img_resized = img.copy()
                    img_resized.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                    canvas.paste(img_resized, (x, y_pred))
                    
                    # 類似度スコア
                    score_text = f"{dist:.3f}"
                    bbox = draw.textbbox((0, 0), score_text, font=font_label)
                    text_width = bbox[2] - bbox[0]
                    score_x = x + (thumb_size[0] - text_width) // 2
                    draw.text((score_x, y_pred + thumb_size[1] + 5), score_text, fill="blue", font=font_label)
            else:
                # 空のプレースホルダー
                placeholder = Image.new("RGB", thumb_size, "lightgray")
                placeholder_draw = ImageDraw.Draw(placeholder)
                placeholder_draw.rectangle([0, 0, thumb_size[0]-1, thumb_size[1]-1], outline="gray")
                canvas.paste(placeholder, (x, y_pred))
    
    canvas.save(save_path)
    print(f"[INFO] 🎨 Scene-based collage saved → {save_path}")

def visualize_test_sets_and_collages(model, test_generator, item_vecs: dict,
                                     scene_root: str = "data",
                                     furn_root: str = "data", 
                                     out_dir: str = "visuals",
                                     top_k: int = 3,
                                     dataset_type: str = "DeepFurniture"):
    """元の関数構造を保ちつつ、シーンベース対応"""
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"[INFO] 🎨 Creating SCENE-BASED visualizations for {dataset_type}")
    
    if dataset_type != "DeepFurniture":
        print(f"[INFO] Skipping visualization for {dataset_type}")
        return
    
    try:
        batch_data = next(iter(test_generator))
        ((Xc, _, _, catQ, catP, qIDs, tIDs, _), setIDs) = batch_data
    except Exception as e:
        print(f"[ERROR] Failed to get batch: {e}")
        return
    
    B = Xc.shape[0] // 2
    config = DATASET_CONFIGS[dataset_type]

    setIDs = [test_generator.get_scene_id_from_index(idx) for idx in setIDs[:B]]
    
    for s in range(min(B, 5)):
        try:
            scene_id = str(setIDs[s])
            
            # モデル予測
            Xin = Xc[s].numpy()
            pred_vecs = model.infer_single_set(Xin)
            
            # 🔧 重要な修正：load_scene_image を使ってシーン画像を読み込み
            scene = load_scene_image(scene_id, scene_root, dataset_type, thumb_size=(400, 300))
            
            # クエリ・ターゲット画像（プレースホルダーを使用）
            q_imgs = [safe_load_furniture_image(str(int(i)), furn_root, dataset_type, 
                                              item_type="Query", category_id=int(c)) 
                     for i, c in zip(qIDs[s].numpy(), catQ[s].numpy()) if int(i) > 0]
            
            t_imgs = [safe_load_furniture_image(str(int(i)), furn_root, dataset_type,
                                              item_type="Target", category_id=int(c)) 
                     for i, c in zip(tIDs[s].numpy(), catP[s].numpy()) if int(i) > 0]
            
            # 予測結果の取得
            target_categories = [int(cat) for cat in catP[s].numpy() if int(cat) > 0]
            query_item_ids = [str(int(id)) for id in qIDs[s].numpy() if int(id) > 0]
            target_item_ids = [str(int(id)) for id in tIDs[s].numpy() if int(id) > 0]
            
            min_cat, max_cat = config['category_range']
            topks = []
            
            for i, cat_id in enumerate(target_categories):
                if not min_cat <= cat_id <= max_cat:
                    topks.append([])
                    continue
                    
                pred_vec = pred_vecs[cat_id - min_cat]
                exclude_items = set(query_item_ids + target_item_ids)
                
                # 類似アイテム検索
                similar_items = find_topk_similar_items_by_euclidean(
                    pred_vec, item_vecs, k=top_k, exclude=exclude_items
                )
                
                # 画像付きの結果に変換
                topk_with_imgs = []
                for item_id, similarity in similar_items:
                    img = safe_load_furniture_image(item_id, furn_root, dataset_type,
                                                  item_type="Retrieval", category_id=cat_id)
                    topk_with_imgs.append((img, similarity))
                
                topks.append(topk_with_imgs)
            
            # 🔧 重要：元のcreate_retrieval_collage関数を呼び出し（scene_idを追加）
            collage_fn = os.path.join(out_dir, f"scene_{scene_id}_retrieval.jpg")
            create_retrieval_collage(
                scene, q_imgs, t_imgs, topks, collage_fn, 
                thumb=(150, 150), dataset_type=dataset_type, scene_id=scene_id
            )
            
        except Exception as e:
            print(f"[ERROR] Failed to process scene {s}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"[INFO] ✅ Scene-based visualizations completed")

# ------------------------------------------------------------------
# Main evaluation pipeline
# ------------------------------------------------------------------
# util.py の main_evaluation_pipeline 関数の修正箇所

def main_evaluation_pipeline(model, test_generator, output_dir="output", 
                           checkpoint_path=None, hard_negative_threshold=0.9,
                           top_k_percentages=[1, 3, 5, 10, 20],
                           combine_directions=True, enable_visualization=True):
    """
    メイン評価パイプライン（修正版 - シーンベース可視化対応）
    """
    print(f"[INFO] 🎯 Starting percentage-based evaluation pipeline")
    
    try:
        # Auto-detect dataset type
        dataset_type = detect_dataset_from_generator(test_generator)
        config = DATASET_CONFIGS[dataset_type]
        print(f"[INFO] Detected dataset type: {dataset_type} ({config['num_categories']} categories)")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. テストアイテム収集
        print("[INFO] Gathering test items...")
        test_items = gather_test_items_fixed(test_generator)
        
        if test_items is None or len(test_items) == 0:
            print("[ERROR] No test items could be collected")
            return
        
        print(f"[INFO] Successfully collected {len(test_items)} test items")
        
        # 2. 評価メトリクス計算
        print("[INFO] Computing percentage-based metrics...")
        metrics = compute_comprehensive_metrics(model, test_items, dataset_type)
        
        # 3. 結果表作成
        print("[INFO] Creating percentage-based results table...")
        results_df = create_quantitative_results_table(metrics, output_dir, dataset_type)
        
        # 4. 可視化（DeepFurnitureのみ、シーンベース）
        if enable_visualization and dataset_type == "DeepFurniture":
            print("[INFO] 🎨 Creating SCENE-BASED visualizations for DeepFurniture...")
            
            # アイテム特徴量辞書を構築
            item_vecs = build_item_feature_dict(model, test_generator)
            
            # シーンベース可視化を実行
            vis_dir = os.path.join(output_dir, "visualizations")
            visualize_test_sets_and_collages(
                model=model,
                test_generator=test_generator,
                item_vecs=item_vecs,
                scene_root="data/DeepFurniture/scenes",
                furn_root="data/DeepFurniture/furnitures",
                out_dir=vis_dir,
                top_k=3,
                dataset_type=dataset_type
            )
            
            print(f"[INFO] ✅ Scene-based visualizations saved to: {vis_dir}")
        elif enable_visualization and dataset_type != "DeepFurniture":
            print(f"[INFO] Skipping visualization for {dataset_type} (only DeepFurniture scene visualization supported)")
        
        # 5. 結果出力
        print(f"\n{'='*80}")
        print(f"🎯 PERCENTAGE-BASED EVALUATION REPORT - {dataset_type}")
        print(f"{'='*80}")
        print(f"📊 Dataset: {dataset_type}")
        print(f"📊 Total Queries: {metrics['overall']['total_queries']}")
        print(f"📊 Categories: {len(metrics['categories'])}")
        print(f"📊 Evaluation Method: Top-K% (percentage of gallery)")
        print(f"")
        print(f"📈 OVERALL PERFORMANCE")
        print(f"----------------------------------------")
        overall = metrics['overall']
        print(f"Mean Reciprocal Rank (MRR): {overall['mrr']:.3f}")
        
        if 'top1_pct' in overall:
            print(f"Top-1% Accuracy: {overall['top1_pct']:.3f} ({overall['top1_pct']*100:.1f}%)")
            print(f"Top-3% Accuracy: {overall['top3_pct']:.3f} ({overall['top3_pct']*100:.1f}%)")
            print(f"Top-5% Accuracy: {overall['top5_pct']:.3f} ({overall['top5_pct']*100:.1f}%)")
            print(f"Top-10% Accuracy: {overall['top10_pct']:.3f} ({overall['top10_pct']*100:.1f}%)")
            print(f"Top-20% Accuracy: {overall['top20_pct']:.3f} ({overall['top20_pct']*100:.1f}%)")
            if 'avg_percentile_rank' in overall:
                print(f"Average Percentile Rank: {overall['avg_percentile_rank']:.1f}%")
        
        print(f"\n📁 Results saved to: {output_dir}/")
        if dataset_type == "DeepFurniture":
            print(f"   - 🎨 Scene visualizations: {output_dir}/visualizations/")
        print(f"   - 📊 CSV results: {output_dir}/{dataset_type.lower()}_percentage_results.csv")
        print(f"   - 📋 Table: {output_dir}/{dataset_type.lower()}_percentage_table.txt")
        print(f"{'='*80}")
        
        print(f"[INFO] ✅ Evaluation completed successfully!")
        
        return metrics
        
    except Exception as e:
        print(f"[ERROR] ❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# util.py の compute_comprehensive_metrics 関数内に追加

def compute_comprehensive_metrics(model, test_items, dataset_type, max_items=10000):
    """
    K%ベースの評価メトリクス計算（学習時と一致した中心基準の評価）
    """
    print("[INFO] Computing percentage-based evaluation metrics with cluster center correction...")
    
    config = DATASET_CONFIGS[dataset_type]
    min_cat, max_cat = config['category_range']
    
    items_to_process = min(len(test_items), max_items)
    test_items_subset = test_items[:items_to_process]

    # ==================================================================
    # STEP 0: Get cluster centers from model
    # ==================================================================
    try:
        cluster_centers = model.get_cluster_center().numpy()  # (num_categories, dim)
        print(f"[INFO] Retrieved cluster centers: {cluster_centers.shape}")
    except Exception as e:
        print(f"[ERROR] Could not get cluster centers: {e}")
        print("[WARN] Falling back to evaluation without cluster center correction")
        # ここで元の評価に戻すか、エラーを出すか
        return compute_comprehensive_metrics_original(model, test_items, dataset_type, max_items)

    # ==================================================================
    # STEP 1: Build the gallery of all items in the test set
    # ==================================================================
    print("[INFO] Step 1/3: Building a gallery of all test items...")
    gallery_by_cat = defaultdict(list)
    for item in test_items_subset:
        _, _, t_feats, t_cats, _, t_ids = item
        for i, cat_id in enumerate(t_cats):
            if cat_id > 0: # Valid category
                gallery_by_cat[cat_id].append((str(int(t_ids[i])), t_feats[i]))

    # Convert to numpy arrays for efficiency
    for cat_id in gallery_by_cat:
        ids, feats = zip(*gallery_by_cat[cat_id])
        gallery_by_cat[cat_id] = {'ids': np.array(ids), 'feats': np.array(feats)}
    
    print(f"[INFO] Gallery built. Found items in {len(gallery_by_cat)} categories.")
    
    # カテゴリ別のギャラリーサイズを表示
    for cat_id, gallery in gallery_by_cat.items():
        print(f"  Category {cat_id}: {len(gallery['ids'])} items")

    # ==================================================================
    # STEP 2: Iterate through queries and compute percentage-based ranks
    # ==================================================================
    print("[INFO] Step 2/3: Processing queries with CENTER-CORRECTED evaluation...")
    
    # Initialize metrics storage
    category_metrics = {cat: {'ranks': [], 'percentile_ranks': []} for cat in range(min_cat, max_cat + 1)}
    
    debug_sample_count = 0
    max_debug_samples = 3
    
    for i, item in enumerate(test_items_subset):
        if i % 1000 == 0:
            print(f"  Processing query {i}/{items_to_process}...")
            
        try:
            q_feats, q_cats, t_feats, t_cats, q_ids, t_ids = item
            
            # デバッグ情報（最初の数サンプルのみ）
            if debug_sample_count < max_debug_samples:
                print(f"\n=== DEBUG: Sample {debug_sample_count + 1} (CENTER-CORRECTED) ===")
                debug_sample_count += 1
            
            # Get model prediction for all categories
            pred_vecs = model.infer_single_set(q_feats[None, :])  # (num_categories, dim)
            
            # For each target item in the current scene
            for j, cat_id in enumerate(t_cats):
                if not (min_cat <= cat_id <= max_cat):
                    continue

                target_id_str = str(int(t_ids[j]))
                
                # Get the prediction and cluster center for this specific category
                pred_vec = pred_vecs[cat_id - min_cat]  # 0-based index
                center_vec = cluster_centers[cat_id - min_cat]  # クラスタ中心
                
                # 🔧 重要: 学習時と同じく中心からの残差ベクトルを計算
                pred_residual = pred_vec - center_vec

                # Get the gallery for this category
                gallery = gallery_by_cat.get(cat_id)
                if not gallery or len(gallery['ids']) == 0:
                    continue

                gallery_ids = gallery['ids']
                gallery_feats = gallery['feats']
                gallery_size = len(gallery_ids)

                # 🔧 重要: ギャラリーアイテムも中心からの残差ベクトルに変換
                gallery_residuals = gallery_feats - center_vec

                # Compute cosine similarity using residual vectors (学習時と同じ)
                pred_residual_norm = np.linalg.norm(pred_residual)
                gallery_residuals_norm = np.linalg.norm(gallery_residuals, axis=1)
                
                if pred_residual_norm == 0: 
                    continue
                
                # 残差ベクトル同士の類似度計算（学習時と一致）
                similarities = np.dot(gallery_residuals, pred_residual) / (gallery_residuals_norm * pred_residual_norm + 1e-8)
                
                # Get sorted indices (highest similarity first)
                sorted_indices = np.argsort(similarities)[::-1]
                
                # Find the rank of the true target item
                rank_list = np.where(gallery_ids[sorted_indices] == target_id_str)[0]
                
                if len(rank_list) > 0:
                    # 1-based rank
                    rank = rank_list[0] + 1
                    
                    # Convert to percentile rank (0-100%)
                    percentile_rank = (rank / gallery_size) * 100
                    
                    category_metrics[cat_id]['ranks'].append(rank)
                    category_metrics[cat_id]['percentile_ranks'].append(percentile_rank)
                    
                    # デバッグ情報: 修正前後の比較
                    if debug_sample_count <= max_debug_samples and j < 3:
                        # print(f"\n--- Target {j+1} Debug (CENTER-CORRECTED) ---")
                        # print(f"Target ID: {target_id_str}, Category: {cat_id}")
                        # print(f"Gallery size: {gallery_size}")
                        # print(f"Rank: {rank}/{gallery_size} ({percentile_rank:.1f}%)")
                        
                        # 修正前の方法との比較のため、生ベクトルでの順位も計算
                        similarities_raw = np.dot(gallery_feats, pred_vec) / (np.linalg.norm(gallery_feats, axis=1) * np.linalg.norm(pred_vec) + 1e-8)
                        sorted_indices_raw = np.argsort(similarities_raw)[::-1]
                        rank_list_raw = np.where(gallery_ids[sorted_indices_raw] == target_id_str)[0]
                        rank_raw = rank_list_raw[0] + 1 if len(rank_list_raw) > 0 else gallery_size
                        percentile_raw = (rank_raw / gallery_size) * 100
                        
                        # print(f"COMPARISON:")
                        # print(f"  Raw vectors rank: {rank_raw}/{gallery_size} ({percentile_raw:.1f}%)")
                        # print(f"  Residual vectors rank: {rank}/{gallery_size} ({percentile_rank:.1f}%)")
                        # print(f"  Improvement: {percentile_raw - percentile_rank:.1f} percentile points")
                        
                        # Top-5の結果を表示
                        # print("Top-5 similar items (CENTER-CORRECTED):")
                        top5_indices = sorted_indices[:5]
                        for k, idx in enumerate(top5_indices):
                            sim_score = similarities[idx]
                            item_id = gallery_ids[idx]
                            is_target = "✅ TARGET" if item_id == target_id_str else ""
                            # print(f"  {k+1}. ID:{item_id} Sim:{sim_score:.4f} {is_target}")

        except Exception as e:
            print(f"  [ERROR] Error processing query {i}: {e}")
            continue

    # ==================================================================
    # STEP 3: Compute percentage-based metrics
    # ==================================================================
    print("[INFO] Step 3/3: Computing percentage-based metrics...")
    results = {'dataset': dataset_type, 'overall': {}, 'categories': {}}
    all_percentile_ranks = []

    for cat_id, metrics_data in category_metrics.items():
        percentile_ranks = np.array(metrics_data['percentile_ranks'])
        if len(percentile_ranks) == 0:
            continue
            
        all_percentile_ranks.extend(percentile_ranks)
        
        # Calculate percentage-based accuracies
        top1_pct = np.mean(percentile_ranks <= 1.0)    # Top 1%
        top3_pct = np.mean(percentile_ranks <= 3.0)    # Top 3%
        top5_pct = np.mean(percentile_ranks <= 5.0)    # Top 5%
        top10_pct = np.mean(percentile_ranks <= 10.0)  # Top 10%
        top20_pct = np.mean(percentile_ranks <= 20.0)  # Top 20%
        
        # MRR based on ranks (traditional)
        ranks = np.array(metrics_data['ranks'])
        mrr = np.mean(1.0 / ranks)
        
        results['categories'][cat_id] = {
            'mrr': mrr,
            'top1_pct': top1_pct,
            'top3_pct': top3_pct, 
            'top5_pct': top5_pct,
            'top10_pct': top10_pct,
            'top20_pct': top20_pct,
            'count': len(ranks),
            'avg_percentile_rank': np.mean(percentile_ranks)
        }

    # Overall metrics
    if all_percentile_ranks:
        overall_percentile_ranks = np.array(all_percentile_ranks)
        
        all_ranks = []
        for cat_id, metrics_data in category_metrics.items():
            all_ranks.extend(metrics_data['ranks'])
        
        overall_ranks = np.array(all_ranks)
        
        results['overall'] = {
            'mrr': np.mean(1.0 / overall_ranks),
            'top1_pct': np.mean(overall_percentile_ranks <= 1.0),
            'top3_pct': np.mean(overall_percentile_ranks <= 3.0),
            'top5_pct': np.mean(overall_percentile_ranks <= 5.0),
            'top10_pct': np.mean(overall_percentile_ranks <= 10.0),
            'top20_pct': np.mean(overall_percentile_ranks <= 20.0),
            'total_queries': len(overall_ranks),
            'avg_percentile_rank': np.mean(overall_percentile_ranks)
        }
    else:
        results['overall'] = {
            'total_queries': 0, 'mrr': 0, 
            'top1_pct': 0, 'top3_pct': 0, 'top5_pct': 0, 'top10_pct': 0, 'top20_pct': 0,
            'avg_percentile_rank': 0
        }

    print(f"[INFO] CENTER-CORRECTED metrics computed for {results['overall']['total_queries']} queries.")
    print(f"[INFO] Average percentile rank: {results['overall']['avg_percentile_rank']:.1f}%")
    print(f"[INFO] Expected improvement: Much better alignment with training (target: ~12.7%)")

    return results

def create_pca_visualization(model, test_items, output_dir, dataset_type, sample_size=3):
    """Create PCA embedding visualization"""
    if not HAS_SKLEARN or not HAS_MATPLOTLIB:
        print("[WARN] PCA visualization skipped - missing dependencies")
        return
    
    print(f"[INFO] Creating PCA visualization for {dataset_type}")
    
    config = DATASET_CONFIGS[dataset_type]
    min_cat, max_cat = config['category_range']
    colors = plt.cm.tab10.colors
    
    # Collect all embeddings
    all_embeddings = []
    all_categories = []
    all_types = []  # 'query', 'target', 'prediction'
    
    sample_items = test_items[:sample_size]
    
    for i, item in enumerate(sample_items):
        try:
            q_feats, q_cats, t_feats, t_cats, _, _ = item
            
            # Get model predictions
            pred_vecs = model.infer_single_set(q_feats[None, :])  # (num_categories, dim)
            
            # Add query items
            for j, (feat, cat) in enumerate(zip(q_feats, q_cats)):
                if min_cat <= cat <= max_cat:
                    all_embeddings.append(feat)
                    all_categories.append(cat)
                    all_types.append('query')
            
            # Add target items
            for j, (feat, cat) in enumerate(zip(t_feats, t_cats)):
                if min_cat <= cat <= max_cat:
                    all_embeddings.append(feat)
                    all_categories.append(cat)
                    all_types.append('target')
            
            # Add predictions
            for cat_idx in range(config['num_categories']):
                all_embeddings.append(pred_vecs[cat_idx])
                all_categories.append(cat_idx + min_cat)
                all_types.append('prediction')
                
        except Exception as e:
            print(f"[ERROR] Failed to process item {i}: {e}")
            continue
    
    if len(all_embeddings) < 10:
        print("[WARN] Not enough embeddings for PCA visualization")
        return
    
    # Apply PCA
    embeddings_np = np.array(all_embeddings)
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_np)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot points by type
    for point_type in ['query', 'target', 'prediction']:
        mask = np.array(all_types) == point_type
        if not np.any(mask):
            continue
            
        x = embeddings_2d[mask, 0]
        y = embeddings_2d[mask, 1]
        cats = np.array(all_categories)[mask]
        
        if point_type == 'query':
            marker, label = 'o', '● Query items'
            size, alpha = 150, 0.8
        elif point_type == 'target':
            marker, label = '^', '▲ Target items'
            size, alpha = 150, 0.8
        else:  # prediction
            marker, label = 'x', '✕ Predictions'
            size, alpha = 200, 1.0
        
        # Color by category
        for cat in range(min_cat, max_cat + 1):
            cat_mask = cats == cat
            if np.any(cat_mask):
                color = colors[(cat - min_cat) % len(colors)]
                plt.scatter(x[cat_mask], y[cat_mask], c=[color], marker=marker, 
                           s=size, alpha=alpha, edgecolors='black', linewidth=1)
    
    plt.title(f"{dataset_type} PCA Embedding Space", fontsize=14, fontweight='bold')
    plt.xlabel(f"PC1 (explained variance: {pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 (explained variance: {pca.explained_variance_ratio_[1]:.2%})")
    plt.grid(True, alpha=0.3)
    
    # Create legend
    legend_elements = [
        plt.scatter([], [], marker="o", c="black", s=150, label="● Query items"),
        plt.scatter([], [], marker="^", c="black", s=150, label="▲ Target items"),
        plt.scatter([], [], marker="x", c="black", s=200, label="✕ Predictions"),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    pca_path = os.path.join(output_dir, f"{dataset_type.lower()}_pca_embedding.png")
    plt.savefig(pca_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] PCA visualization saved → {pca_path}")

def create_quantitative_results_table(metrics, output_dir, dataset_type):
    """K%ベースの結果表を作成"""
    print(f"[INFO] Creating percentage-based results table for {dataset_type}")
    
    config = DATASET_CONFIGS[dataset_type]
    category_names = config['category_names']
    
    # Prepare table data
    table_data = []
    
    # Add category rows
    for cat_id in sorted(metrics['categories'].keys()):
        cat_metrics = metrics['categories'][cat_id]
        cat_name = category_names.get(cat_id, f'Category {cat_id}')
        
        table_data.append({
            'Category': f"{cat_id}",
            'Name': cat_name[:15] + "..." if len(cat_name) > 15 else cat_name,
            'Count': cat_metrics['count'],
            'MRR': f"{cat_metrics['mrr']:.3f}",
            'Top1%': f"{cat_metrics['top1_pct']:.3f}",
            'Top3%': f"{cat_metrics['top3_pct']:.3f}",
            'Top5%': f"{cat_metrics['top5_pct']:.3f}",
            'Top10%': f"{cat_metrics['top10_pct']:.3f}",
            'Top20%': f"{cat_metrics['top20_pct']:.3f}",
            'AvgPercentile': f"{cat_metrics['avg_percentile_rank']:.1f}%"
        })
    
    # Add overall row
    overall = metrics['overall']
    table_data.append({
        'Category': 'Overall',
        'Name': 'All Categories',
        'Count': overall['total_queries'],
        'MRR': f"{overall['mrr']:.3f}",
        'Top1%': f"{overall['top1_pct']:.3f}",
        'Top3%': f"{overall['top3_pct']:.3f}",
        'Top5%': f"{overall['top5_pct']:.3f}",
        'Top10%': f"{overall['top10_pct']:.3f}",
        'Top20%': f"{overall['top20_pct']:.3f}",
        'AvgPercentile': f"{overall['avg_percentile_rank']:.1f}%"
    })
    
    # Save as CSV
    df = pd.DataFrame(table_data)
    csv_path = os.path.join(output_dir, f"{dataset_type.lower()}_percentage_results.csv")
    df.to_csv(csv_path, index=False)
    
    # Create formatted text table
    table_path = os.path.join(output_dir, f"{dataset_type.lower()}_percentage_table.txt")
    
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(f"📊 **{dataset_type} Percentage-Based Results (Top-K%)**\n")
        f.write("=" * 100 + "\n\n")
        
        # Header
        f.write(f"{'Category':<10} {'Name':<20} {'Count':<8} {'MRR':<8} {'Top1%':<8} {'Top3%':<8} {'Top5%':<8} {'Top10%':<8} {'Top20%':<8} {'AvgPct':<8}\n")
        f.write("-" * 100 + "\n")
        
        # Category rows
        for _, row in df.iterrows():
            f.write(f"{row['Category']:<10} {row['Name']:<20} {row['Count']:<8} {row['MRR']:<8} "
                   f"{row['Top1%']:<8} {row['Top3%']:<8} {row['Top5%']:<8} {row['Top10%']:<8} {row['Top20%']:<8} {row['AvgPercentile']:<8}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"Dataset: {dataset_type}\n")
        f.write(f"Total Queries: {overall['total_queries']}\n")
        f.write(f"Categories: {len(metrics['categories'])}\n")
        f.write(f"Evaluation Method: Percentage-based (Top-K% of gallery)\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"[INFO] Percentage-based results saved:")
    print(f"  - CSV: {csv_path}")
    print(f"  - Table: {table_path}")
    
    return df

def create_comprehensive_report(metrics, output_dir, dataset_type):
    """Create comprehensive evaluation report"""
    report_path = os.path.join(output_dir, f"{dataset_type.lower()}_comprehensive_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"🎯 COMPREHENSIVE EVALUATION REPORT - {dataset_type}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Dataset: {dataset_type}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Queries: {metrics['overall']['total_queries']}\n")
        f.write(f"Categories Evaluated: {len(metrics['categories'])}\n\n")
        
        # Overall performance
        f.write("📊 OVERALL PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        overall = metrics['overall']
        f.write(f"Mean Reciprocal Rank (MRR): {overall['mrr']:.3f}\n")
        f.write(f"Top-1 Accuracy: {overall['top1_acc']:.3f} ({overall['top1_acc']*100:.1f}%)\n")
        f.write(f"Top-5 Accuracy: {overall['top5_acc']:.3f} ({overall['top5_acc']*100:.1f}%)\n")
        f.write(f"Top-10 Accuracy: {overall['top10_acc']:.3f} ({overall['top10_acc']*100:.1f}%)\n")
        f.write(f"Top-20 Accuracy: {overall['top20_acc']:.3f} ({overall['top20_acc']*100:.1f}%)\n\n")
        
        # Category breakdown
        f.write("📈 CATEGORY BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        config = DATASET_CONFIGS[dataset_type]
        category_names = config['category_names']
        
        for cat_id in sorted(metrics['categories'].keys()):
            cat_metrics = metrics['categories'][cat_id]
            cat_name = category_names.get(cat_id, f'Category {cat_id}')
            
            f.write(f"\nCategory {cat_id}: {cat_name}\n")
            f.write(f"  Queries: {cat_metrics['count']}\n")
            f.write(f"  MRR: {cat_metrics['mrr']:.3f}\n")
            f.write(f"  Top-1: {cat_metrics['top1_acc']:.3f} ({cat_metrics['top1_acc']*100:.1f}%)\n")
            f.write(f"  Top-5: {cat_metrics['top5_acc']:.3f} ({cat_metrics['top5_acc']*100:.1f}%)\n")
            f.write(f"  Top-10: {cat_metrics['top10_acc']:.3f} ({cat_metrics['top10_acc']*100:.1f}%)\n")
            f.write(f"  Top-20: {cat_metrics['top20_acc']:.3f} ({cat_metrics['top20_acc']*100:.1f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("📁 Generated Files:\n")
        f.write(f"  - Quantitative Results: {dataset_type.lower()}_quantitative_results.csv\n")
        f.write(f"  - Results Table: {dataset_type.lower()}_results_table.txt\n")
        f.write(f"  - Retrieval Collages: *_collage.jpg\n")
        f.write(f"  - PCA Embeddings: {dataset_type.lower()}_pca_embedding.png\n")
        f.write("=" * 80 + "\n")
    
    print(f"[INFO] Comprehensive report saved → {report_path}")

# Legacy compatibility function
def compute_global_rank(model, test_generator, output_dir="output", 
                       checkpoint_path=None, hard_negative_threshold=0.9,
                       top_k_values=[5, 10, 20], 
                       top_k_percentages=[5, 10, 20],
                       combine_directions=True, enable_visualization=False,
                       dataset_type=None):
    """Legacy function - redirects to main pipeline"""
    print("[INFO] Redirecting to comprehensive evaluation pipeline...")
    return main_evaluation_pipeline(
        model=model,
        test_generator=test_generator,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        hard_negative_threshold=hard_negative_threshold,
        top_k_percentages=top_k_percentages,
        combine_directions=combine_directions,
        enable_visualization=enable_visualization
    )