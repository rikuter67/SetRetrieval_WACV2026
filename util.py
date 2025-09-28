"""
util.py - Core evaluation logic for Set Retrieval.
"""
import os
import gc
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
import random
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
            # IQON3000のカテゴリ数16とDeepFurnitureのカテゴリ数11を考慮
            if max_cat <= len(IQON3000_CATEGORIES): # 16以下ならIQON3000と判定
                return 'IQON3000'
            elif max_cat <= len(DEEPFURNITURE_CATEGORIES): # 11以下ならDeepFurnitureと判定 (この順序が重要)
                return 'DeepFurniture'
            else:
                return 'DeepFurniture' # デフォルトはDeepFurniture
    except Exception as e:
        print(f"[WARN] Could not detect dataset type: {e}")
    return 'IQON3000' # デフォルト


def collect_test_data(test_data) -> Tuple[List[Dict], Dict, str]:
    """テストデータ収集（DataGeneratorから直接取得）"""
    print("[INFO] 📥 Collecting test data and building gallery...")
    
    # ★★★ 修正: DataGeneratorから直接取得を優先 ★★★
    if hasattr(test_data, '_data_generator'):
        print("[INFO] Using DataGenerator for test data collection...")
        data_gen = test_data._data_generator
        dataset_type = data_gen.dataset_name or 'Unknown'
        
        # DataGeneratorから直接全バッチを取得
        all_batches = []
        print(f"[INFO] Collecting {len(data_gen)} batches from DataGenerator...")
        for i in range(len(data_gen)):
            batch = data_gen[i]  # DataGeneratorから直接取得（set_idsを含む）
            if batch is not None:
                all_batches.append(batch)
                
    else:
        # フォールバック：tf.data.Dataset経由
        print("[INFO] Using tf.data.Dataset iterator...")
        dataset_type = detect_dataset_type(test_data)
        all_batches = list(tqdm(test_data.as_numpy_iterator(), desc="Collecting batches"))
    
    # データセット設定を取得
    config = get_dataset_config(dataset_type)
    min_cat, max_cat = config['category_range']
    
    test_items = []
    gallery_by_category = defaultdict(dict)
    
    print("[INFO] Processing sets and building gallery...")
    for batch in tqdm(all_batches, desc="Processing batches"):
        batch_size = len(batch['query_features'])
        
        # ★★★ デバッグ: バッチに含まれるキーを確認 ★★★
        if len(test_items) == 0:  # 最初のバッチでのみ表示
            print(f"[DEBUG] Available keys in batch: {list(batch.keys())}")
        
        for i in range(batch_size):
            # 基本的なアイテムセットを作成
            item_set = {
                'query_features': batch['query_features'][i],
                'target_features': batch['target_features'][i],
                'query_categories': batch['query_categories'][i],
                'target_categories': batch['target_categories'][i],
                'query_item_ids': batch['query_item_ids'][i],
                'target_item_ids': batch['target_item_ids'][i],
            }
            
            # ★★★ 修正: set_idsの安全な取得 ★★★
            if 'set_ids' in batch:
                set_id_raw = batch['set_ids'][i]
                if isinstance(set_id_raw, bytes):
                    item_set['set_id'] = set_id_raw.decode('utf-8')
                elif isinstance(set_id_raw, (np.str_, np.bytes_)):
                    item_set['set_id'] = str(set_id_raw)
                else:
                    item_set['set_id'] = str(set_id_raw)
            else:
                # set_idsがない場合は代替IDを生成
                item_set['set_id'] = f"test_set_{len(test_items)}"
                if len(test_items) == 0:
                    print("[WARNING] set_ids not found in batch, using generated IDs")
            
            test_items.append(item_set)

            # ギャラリーを構築
            for feat, cat, item_id in zip(item_set['target_features'], item_set['target_categories'], item_set['target_item_ids']):
                cat_int = int(cat)
                if min_cat <= cat_int <= max_cat and np.any(feat):
                    gallery_by_category[cat_int][str(item_id)] = feat.astype(np.float32)

    # ギャラリーの最終処理
    for cat_id in list(gallery_by_category.keys()):
        items = gallery_by_category[cat_id]
        if items:
            gallery_by_category[cat_id] = {
                'ids': np.array(list(items.keys())), 
                'features': np.array(list(items.values()))
            }
    
    print(f"✅ Collected {len(test_items)} test items and {len(gallery_by_category)} categories")
    return test_items, dict(gallery_by_category), dataset_type


def collect_training_items_by_category(train_data_path):
    """学習データから各カテゴリのアイテムを収集"""
    
    with open(train_data_path, 'rb') as f:
        data = pickle.load(f)
    
    query_features = np.array(data[0], dtype=np.float32)
    positive_features = np.array(data[1], dtype=np.float32)
    query_categories = np.array(data[3], dtype=np.int32)
    positive_categories = np.array(data[4], dtype=np.int32)
    
    # カテゴリ別アイテム辞書
    items_by_category = defaultdict(list)
    
    # 全シーンから全アイテムを収集
    for scene_idx in range(len(query_features)):
        # queryアイテム
        for item_idx in range(len(query_features[scene_idx])):
            if query_categories[scene_idx, item_idx] > 0:
                cat = int(query_categories[scene_idx, item_idx])
                feat = query_features[scene_idx, item_idx]
                item_id = f"q_{scene_idx}_{item_idx}"
                items_by_category[cat].append({
                    'id': item_id,
                    'features': feat,
                    'scene_idx': scene_idx,
                    'item_idx': item_idx,
                    'type': 'query'
                })
        
        # positiveアイテム
        for item_idx in range(len(positive_features[scene_idx])):
            if positive_categories[scene_idx, item_idx] > 0:
                cat = int(positive_categories[scene_idx, item_idx])
                feat = positive_features[scene_idx, item_idx]
                item_id = f"p_{scene_idx}_{item_idx}"
                items_by_category[cat].append({
                    'id': item_id,
                    'features': feat,
                    'scene_idx': scene_idx,
                    'item_idx': item_idx,
                    'type': 'positive'
                })
    
    return items_by_category



# util.py に追加する関数群

class HardNegativeMiner:
    """高速化されたHard Negative事前計算とキャッシュ管理"""
    
    def __init__(self, train_data_path, whitening_params=None, 
                 max_items_per_category=1500, progress_interval=100, 
                 max_negatives_per_item=15):
        self.whitening_params = whitening_params
        self.max_items_per_category = max_items_per_category
        self.progress_interval = progress_interval
        self.max_negatives_per_item = max_negatives_per_item
        
        self.items_by_category = {}
        self.hard_negatives_cache = {}
        self.scheduler = CurriculumScheduler()
        
        print(f"🚀 Starting Optimized Hard Negative computation...")
        print(f"   Max items per category: {max_items_per_category}")
        print(f"   Max negatives per item: {max_negatives_per_item}")
        print(f"   Progress interval: {progress_interval}")
        
        # 学習データから事前計算実行
        self._precompute_from_training_data(train_data_path)
        
        print(f"✅ Optimized Hard Negative mining completed!")
        print(f"   Categories: {len(self.items_by_category)}")
        print(f"   Cached relationships: {len(self.hard_negatives_cache):,}")
    
    def _precompute_from_training_data(self, train_data_path):
        """学習データから全Hard Negative関係を事前計算"""
        # Step 1: 学習データ収集（高速版）
        print("  Step 1: Fast data collection with sampling...")
        
        self.items_by_category = collect_training_items_by_category_fast(
            train_data_path, self.max_items_per_category
        )
        
        # Step 2: ホワイトニング適用
        if self.whitening_params:
            print("  Step 2: Applying whitening...")
            self._apply_whitening_to_items()
        
        # Step 3: 高速類似度計算
        print("  Step 3: Fast similarity computation...")
        self._compute_all_similarities_fast()
    
    def _apply_whitening_to_items(self):
        """ホワイトニング適用（バッチ処理）"""
        for cat_id, items in self.items_by_category.items():
            if len(items) > 0:
                print(f"    Applying whitening to category {cat_id}: {len(items)} items")
                
                # バッチ処理でホワイトニング
                features = np.array([item['features'] for item in items])
                whitened_features = np.dot(features - self.whitening_params['mean'], 
                                         self.whitening_params['matrix'])
                
                # 結果を元のアイテムに書き戻し
                for i, item in enumerate(items):
                    item['features'] = whitened_features[i]
    
    def _compute_all_similarities_fast(self):
        """高速化された類似度計算"""
        total_processed = 0
        
        for cat_id, items in self.items_by_category.items():
            if len(items) < 2:
                continue
            
            print(f"    Processing category {cat_id}: {len(items)} items")
            
            # ★ 代表的なアイテムのみ処理（さらなる高速化）
            if len(items) > 1000:
                # 大きなカテゴリは代表サンプルのみ処理
                sample_size = min(500, len(items) // 3)
                sample_indices = np.random.choice(len(items), sample_size, replace=False)
                process_items = [items[i] for i in sample_indices]
                print(f"      Large category detected, processing {len(process_items)} representative items")
            else:
                process_items = items
            
            # 各アイテムのHard Negative計算
            for item in process_items:
                hard_negatives = compute_hard_negatives_for_item_fast(
                    item, items, 
                    similarity_threshold=0.1,  # 最低閾値
                    max_negatives=self.max_negatives_per_item
                )
                
                self.hard_negatives_cache[item['id']] = hard_negatives
                total_processed += 1
                
                # ★ 高頻度進捗表示
                if total_processed % self.progress_interval == 0:
                    print(f"      Processed {total_processed:,} items...")
            
            # メモリ解放
            if len(items) > 1000:
                # 大きなカテゴリの処理後はメモリ解放
                gc.collect()
        
        print(f"    Total processed items: {total_processed:,}")
    
    def get_hard_negatives_for_item(self, item_id, epoch, max_samples=60):
        """指定アイテム・エポックのHard Negativeを取得"""
        current_threshold = self.scheduler.get_threshold(epoch)
        all_hard_negatives = self.hard_negatives_cache.get(item_id, [])
        
        # 現在エポックに適した困難度のものを選択
        epoch_hard_negatives = [
            hn for hn in all_hard_negatives 
            if hn['similarity'] >= current_threshold
        ]
        
        # 不足している場合は低い閾値のものも追加
        if len(epoch_hard_negatives) < max_samples // 2 and len(all_hard_negatives) > 0:
            remaining = max_samples - len(epoch_hard_negatives)
            lower_threshold_negatives = [
                hn for hn in all_hard_negatives 
                if hn['similarity'] < current_threshold
            ]
            if lower_threshold_negatives:
                additional = random.sample(
                    lower_threshold_negatives, 
                    min(remaining, len(lower_threshold_negatives))
                )
                epoch_hard_negatives.extend(additional)
        
        return epoch_hard_negatives[:max_samples]


def save_hard_negative_cache(cache_obj, output_path):
    """Hard Negativeキャッシュ保存"""
    if hasattr(cache_obj, 'hard_negatives_cache'):
        # 新形式（クラスオブジェクト）
        save_data = {
            'hard_negatives_cache': cache_obj.hard_negatives_cache,
            'scheduler': cache_obj.scheduler,
            'max_items_per_category': getattr(cache_obj, 'max_items_per_category', 3000),
            'max_negatives_per_item': getattr(cache_obj, 'max_negatives_per_item', 30)
        }
    else:
        # 旧形式（辞書）
        save_data = cache_obj
    
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    if hasattr(cache_obj, 'hard_negatives_cache'):
        print(f"[INFO] Saved {len(cache_obj.hard_negatives_cache):,} hard negative relationships")
    else:
        print(f"[INFO] Saved hard negative cache")


def compute_hard_negatives_for_item(positive_item, category_items, similarity_threshold=0.3):
    """指定アイテムに対するHard Negativeを計算"""
    
    positive_features = positive_item['features']
    hard_negatives = []
    
    for candidate_item in category_items:
        # 自分自身は除外
        if candidate_item['id'] == positive_item['id']:
            continue
        
        # コサイン類似度計算
        candidate_features = candidate_item['features']
        similarity = np.dot(positive_features, candidate_features) / (
            np.linalg.norm(positive_features) * np.linalg.norm(candidate_features)
        )
        
        # 閾値以上をHard Negativeとして選択
        if similarity >= similarity_threshold:
            hard_negatives.append({
                'item': candidate_item,
                'similarity': similarity
            })
    
    # 類似度降順でソート
    hard_negatives.sort(key=lambda x: x['similarity'], reverse=True)
    
    return hard_negatives

    
class CurriculumScheduler:
    """段階的困難度調整"""
    
    def __init__(self, init_threshold=0.3, max_threshold=0.8, increment=0.05):
        self.init_threshold = init_threshold
        self.max_threshold = max_threshold
        self.increment = increment
    
    def get_threshold(self, epoch):
        return min(self.max_threshold, self.init_threshold + epoch * self.increment)

def compute_similarities_vectorized(features_a, features_b, batch_size=2500):
    """
    ベクトル化された高速類似度計算
    
    Args:
        features_a: (N, D) 特徴量配列A
        features_b: (M, D) 特徴量配列B
        batch_size: バッチサイズ
    
    Returns:
        similarities: (N, M) 類似度行列
    """
    N, D = features_a.shape
    M = features_b.shape[0]
    
    # 正規化
    features_a_norm = features_a / (np.linalg.norm(features_a, axis=1, keepdims=True) + 1e-8)
    features_b_norm = features_b / (np.linalg.norm(features_b, axis=1, keepdims=True) + 1e-8)
    
    # GPU使用可能かチェック
    if tf.config.list_physical_devices('GPU') and N > 500 and M > 500:
        try:
            # GPU版：バッチ処理
            similarities = np.zeros((N, M), dtype=np.float32)
            
            for i in range(0, N, batch_size):
                end_i = min(i + batch_size, N)
                batch_a = tf.constant(features_a_norm[i:end_i])
                batch_sim = tf.matmul(batch_a, tf.constant(features_b_norm), transpose_b=True)
                similarities[i:end_i] = batch_sim.numpy()
            
            return similarities
        except:
            # GPU計算失敗時はCPUにフォールバック
            pass
    
    # CPU版：NumPy（高速）
    return np.dot(features_a_norm, features_b_norm.T)

def collect_training_items_by_category_fast(train_data_path, max_items_per_category=2000):
    """高速化された学習データ収集（サンプリング付き）"""
    
    with open(train_data_path, 'rb') as f:
        data = pickle.load(f)
    
    query_features = np.array(data[0], dtype=np.float32)
    positive_features = np.array(data[1], dtype=np.float32)
    query_categories = np.array(data[3], dtype=np.int32)
    positive_categories = np.array(data[4], dtype=np.int32)
    
    # ★ 修正：実際のアイテムIDを使用
    query_item_ids = np.array(data[6])      # 実際のID
    positive_item_ids = np.array(data[7])   # 実際のID

    # 一時的な収集用辞書
    temp_features = defaultdict(list)
    temp_metadata = defaultdict(list)
    
    print("  Collecting items by category...")
    
    # 全シーンから全アイテムを収集
    for scene_idx in range(len(query_features)):
        # queryアイテム
        for item_idx in range(len(query_features[scene_idx])):
            if query_categories[scene_idx, item_idx] > 0:
                cat = int(query_categories[scene_idx, item_idx])
                feat = query_features[scene_idx, item_idx]
                item_id = str(query_item_ids[scene_idx, item_idx])
                
                temp_features[cat].append(feat)
                temp_metadata[cat].append({
                    'id': item_id, 'scene_idx': scene_idx, 'item_idx': item_idx, 'type': 'query'
                })
        
        # positiveアイテム
        for item_idx in range(len(positive_features[scene_idx])):
            if positive_categories[scene_idx, item_idx] > 0:
                cat = int(positive_categories[scene_idx, item_idx])
                feat = positive_features[scene_idx, item_idx]
                item_id = str(positive_item_ids[scene_idx, item_idx])
                
                temp_features[cat].append(feat)
                temp_metadata[cat].append({
                    'id': item_id, 'scene_idx': scene_idx, 'item_idx': item_idx, 'type': 'positive'
                })
    
    # ★ スマートサンプリング適用
    items_by_category = {}
    
    for cat_id, features in temp_features.items():
        original_count = len(features)
        
        if original_count > max_items_per_category:
            print(f"    Category {cat_id}: Sampling {max_items_per_category} from {original_count} items")
            
            # ランダムサンプリング（シンプルで高速）
            selected_indices = np.random.choice(
                original_count, max_items_per_category, replace=False
            )
            
            # 選択されたアイテムのみ保持
            sampled_items = []
            for idx in selected_indices:
                sampled_items.append({
                    'id': temp_metadata[cat_id][idx]['id'],
                    'features': features[idx],
                    'scene_idx': temp_metadata[cat_id][idx]['scene_idx'],
                    'item_idx': temp_metadata[cat_id][idx]['item_idx'],
                    'type': temp_metadata[cat_id][idx]['type']
                })
            
            items_by_category[cat_id] = sampled_items
        else:
            # サンプリング不要
            full_items = []
            for idx, feat in enumerate(features):
                full_items.append({
                    'id': temp_metadata[cat_id][idx]['id'],
                    'features': feat,
                    'scene_idx': temp_metadata[cat_id][idx]['scene_idx'],
                    'item_idx': temp_metadata[cat_id][idx]['item_idx'],
                    'type': temp_metadata[cat_id][idx]['type']
                })
            
            items_by_category[cat_id] = full_items
        
        print(f"    Category {cat_id}: {len(items_by_category[cat_id])} items (from {original_count})")
    
    return items_by_category

def compute_hard_negatives_for_item_fast(positive_item, category_items, similarity_threshold=0.3, max_negatives=50):
    """高速化されたHard Negative計算"""
    
    if len(category_items) <= 1:
        return []
    
    positive_features = positive_item['features']
    positive_id = positive_item['id']
    
    # 候補特徴量を抽出（自分以外）
    candidate_features = []
    candidate_items = []
    
    for item in category_items:
        if item['id'] != positive_id:
            candidate_features.append(item['features'])
            candidate_items.append(item)
    
    if len(candidate_features) == 0:
        return []
    
    candidate_features = np.array(candidate_features)
    
    # ベクトル化されたコサイン類似度計算
    positive_norm = positive_features / (np.linalg.norm(positive_features) + 1e-8)
    candidate_norms = candidate_features / (np.linalg.norm(candidate_features, axis=1, keepdims=True) + 1e-8)
    similarities = np.dot(candidate_norms, positive_norm)
    
    # 閾値以上のもののみ選択
    valid_mask = similarities >= similarity_threshold
    valid_indices = np.where(valid_mask)[0]
    valid_similarities = similarities[valid_mask]
    
    if len(valid_indices) == 0:
        return []
    
    # Top-K選択（類似度降順）
    k = min(max_negatives, len(valid_indices))
    top_k_indices = np.argsort(valid_similarities)[::-1][:k]
    
    hard_negatives = []
    for i in top_k_indices:
        original_idx = valid_indices[i]
        hard_negatives.append({
            'item': candidate_items[original_idx],
            'similarity': float(valid_similarities[i])
        })
    
    return hard_negatives


def load_hard_negative_cache(cache_path):
    """Hard Negativeキャッシュ読み込み"""
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            save_data = pickle.load(f)
        
        # 新形式の場合
        if isinstance(save_data, dict) and 'hard_negatives_cache' in save_data:
            class CacheWrapper:
                def __init__(self, data):
                    self.hard_negatives_cache = data['hard_negatives_cache']
                    self.scheduler = data.get('scheduler', CurriculumScheduler())
                    self.max_items_per_category = data.get('max_items_per_category', 3000)
                    self.max_negatives_per_item = data.get('max_negatives_per_item', 30)
                
                def get_hard_negatives_for_item(self, item_id, epoch, max_samples=60):
                    current_threshold = self.scheduler.get_threshold(epoch)
                    all_hard_negatives = self.hard_negatives_cache.get(item_id, [])
                    
                    epoch_hard_negatives = [
                        hn for hn in all_hard_negatives 
                        if hn['similarity'] >= current_threshold
                    ]
                    
                    # 不足分を補完
                    if len(epoch_hard_negatives) < max_samples // 2 and len(all_hard_negatives) > 0:
                        remaining = max_samples - len(epoch_hard_negatives)
                        lower_threshold_negatives = [
                            hn for hn in all_hard_negatives 
                            if hn['similarity'] < current_threshold
                        ]
                        if lower_threshold_negatives:
                            additional = random.sample(
                                lower_threshold_negatives, 
                                min(remaining, len(lower_threshold_negatives))
                            )
                            epoch_hard_negatives.extend(additional)
                    
                    return epoch_hard_negatives[:max_samples]
            
            cache_obj = CacheWrapper(save_data)
            print(f"[INFO] Loaded {len(cache_obj.hard_negatives_cache):,} hard negative relationships")
            return cache_obj
        else:
            # 旧形式（後方互換性）
            return save_data
    
    return None

def compute_training_whitening_params(train_generator, model, num_categories=7):
    """
    訓練データからカテゴリ別ホワイトニングパラメータを計算
    
    Args:
        train_generator: 訓練データジェネレータ
        model: 学習済みモデル
        num_categories: カテゴリ数
    
    Returns:
        カテゴリ別ホワイトニングパラメータ
    """
    print("[INFO] Computing whitening parameters from TRAINING data...")
    
    # カテゴリ別特徴量収集
    category_embeddings = {cat_id: [] for cat_id in range(1, num_categories + 1)}
    
    # 訓練データから特徴量収集
    batch_count = 0
    for batch in train_generator.take(100):  # 最初の100バッチ使用
        query_features = batch['query_features']
        query_categories = batch['query_categories']
        target_features = batch['target_features']
        target_categories = batch['target_categories']
        
        # モデルで予測（各カテゴリの埋め込みを取得）
        predictions = model({'query_features': query_features}, training=False)
        
        # ターゲット特徴量から各カテゴリの埋め込みを収集
        for sample_idx in range(target_features.shape[0]):
            for item_idx in range(target_features.shape[1]):
                cat_id = int(target_categories[sample_idx, item_idx])
                if 1 <= cat_id <= num_categories:
                    # この特徴量からモデルが生成するであろう埋め込み
                    item_features = target_features[sample_idx, item_idx:item_idx+1]
                    item_embedding = model.infer_single_set(item_features)[cat_id - 1]
                    category_embeddings[cat_id].append(item_embedding.numpy())
        
        batch_count += 1
        if batch_count % 20 == 0:
            print(f"  Processed {batch_count} training batches...")
    
    # カテゴリ別ホワイトニングパラメータ計算
    category_whitening_params = {}
    
    for cat_id in range(1, num_categories + 1):
        if cat_id in category_embeddings and len(category_embeddings[cat_id]) > 0:
            embeddings = np.array(category_embeddings[cat_id])
            mean_vec, whitening_matrix = compute_whitening_params(embeddings)
            category_whitening_params[cat_id] = {
                'mean': mean_vec, 
                'matrix': whitening_matrix
            }
            print(f"  Category {cat_id}: {len(embeddings)} training samples")
        else:
            print(f"  Category {cat_id}: No training samples, skipping")
    
    print(f"[INFO] Training-based whitening parameters computed for {len(category_whitening_params)} categories")
    return category_whitening_params


def save_whitening_params(whitening_params, filepath):
    """ホワイトニングパラメータを保存"""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(whitening_params, f)
    print(f"[INFO] Whitening parameters saved to {filepath}")


def load_whitening_params(filepath):
    """ホワイトニングパラメータを読み込み"""
    import pickle
    with open(filepath, 'rb') as f:
        whitening_params = pickle.load(f)
    print(f"[INFO] Whitening parameters loaded from {filepath}")
    return whitening_params


def compute_whitening_params(features, epsilon=1e-6):
    """シンプルなホワイトニングパラメータ計算"""
    mean_vec = np.mean(features, axis=0)
    centered = features - mean_vec
    cov_matrix = np.cov(centered.T)
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    # 数値安定化：0に近い固有値を修正
    eigenvals = np.maximum(eigenvals, epsilon)
    whitening_matrix = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
    return mean_vec, whitening_matrix

def apply_whitening(features, mean_vec, whitening_matrix):
    """シンプルなホワイトニング適用"""
    centered = features - mean_vec
    whitened = centered @ whitening_matrix.T
    norms = np.linalg.norm(whitened, axis=1, keepdims=True)
    return whitened / np.maximum(norms, 1e-8)

# def compute_training_whitening_stats(model, train_dataset):
#     """学習データ全体から統計量計算"""
#     print("[INFO] Computing whitening statistics from ALL training data...")
    
#     all_embeddings = []
#     batch_count = 0
    
#     for batch in train_dataset:
#         # バッチ全体を一度に推論
#         predictions = model({'query_features': batch['query_features']}, training=False)
#         # 全カテゴリの埋め込みを収集（形状: [batch_size, num_categories, embed_dim]）
#         batch_embeddings = predictions.numpy().reshape(-1, predictions.shape[-1])
#         all_embeddings.append(batch_embeddings)
        
#         batch_count += 1
#         if batch_count % 10 == 0:  # 10バッチごとに進捗表示
#             print(f"  Processed {batch_count} batches...")
    
#     # 全埋め込みを結合
#     all_embeddings = np.concatenate(all_embeddings, axis=0)
#     print(f"  Collected {len(all_embeddings):,} embeddings for whitening from {batch_count} batches")
    
#     # 統計量計算
#     mean_vec, whitening_matrix = compute_whitening_params(all_embeddings)
    
#     return {'mean': mean_vec, 'matrix': whitening_matrix}


# def apply_whitening_to_evaluation(model, test_items, gallery_by_category, whitening_params):
#     """テスト時にホワイトニング適用"""
#     # ギャラリーにホワイトニング適用
#     for cat_id, gallery in gallery_by_category.items():
#         if 'features' in gallery:
#             # ギャラリー特徴量 → モデル → ホワイトニング
#             gallery_embeddings = []
#             for feat in gallery['features']:
#                 pred = model({'query_features': feat.reshape(1, 1, -1)}, training=False)
#                 gallery_embeddings.append(pred[0, cat_id-1].numpy())
            
#             gallery_embeddings = np.array(gallery_embeddings)
#             gallery['features'] = apply_whitening(gallery_embeddings, whitening_params['mean'], whitening_params['matrix'])
    
#     return gallery_by_category

def apply_whitening_to_predictions(predictions, whitening_params):
    """予測ベクトルにホワイトニング適用（高速版）"""
    if whitening_params is None:
        return predictions
        
    original_shape = predictions.shape
    # [batch_size, num_categories, embed_dim] -> [batch_size * num_categories, embed_dim]
    flat_predictions = predictions.reshape(-1, original_shape[-1])
    
    # ホワイトニング適用
    whitened = apply_whitening(flat_predictions, whitening_params['mean'], whitening_params['matrix'])
    
    # 元の形状に戻す
    return whitened.reshape(original_shape)


import gzip
import pickle


def compute_input_whitening_stats(dataset_path: str, feature_dim: int):
    """
    データセットのpklファイルから生の入力特徴量を読み込み、
    ホワイトニングに必要な統計量（平均と共分散行列）を計算する。
    """
    import gzip
    import pickle
    import numpy as np
    from tqdm import tqdm
    import os

    print("🔧 Collecting all raw features from training data for whitening stats...")
    
    train_split_path = os.path.join(dataset_path, 'train.pkl')
    
    # ★★★ 修正: ファイルが.gzで終わるか確認して開く方法を変える ★★★
    if train_split_path.endswith('.gz'):
        opener = gzip.open
    else:
        # .gzでなければ通常のopenを使用
        opener = open
        
    try:
        with opener(train_split_path, 'rb') as f:
            # data_generator._load_data と同じ形式でロード
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {train_split_path}")
        return None
    except Exception as e:
        print(f"Error loading data from {train_split_path}: {e}")
        raise # 他のエラーは再スローしてデバッグを助ける
    
    # 既存の_load_dataロジックを簡略化して特徴量だけを収集
    if not (isinstance(data, tuple) and len(data) >= 8):
        raise ValueError(f"Unsupported data format in {train_split_path}. Expected a tuple with at least 8 elements.")

    query_features = np.array(data[0], dtype=np.float32)
    positive_features = np.array(data[1], dtype=np.float32)

    # query featuresとpositive featuresを全て統合
    all_features = np.concatenate([query_features.reshape(-1, feature_dim), positive_features.reshape(-1, feature_dim)], axis=0)

    # 0ベクトル（パディング部分）を除外
    # norm = np.linalg.norm(all_features, axis=1)
    # all_features = all_features[norm > 1e-8]
    
    print(f"✅ Collected {len(all_features)} features with shape {all_features.shape}")
    
    # 平均と共分散行列を計算
    mean_vector = np.mean(all_features, axis=0)
    
    # データ点数が次元数より多いことを確認
    num_samples, num_dims = all_features.shape
    if num_samples <= num_dims:
        print(f"⚠️ Warning: Number of samples ({num_samples}) is not greater than feature dimension ({num_dims}).")
        print("         Covariance matrix will be singular. Consider using a smaller dimension or more data.")

    # 共分散行列の計算 (データ数が次元数より多いことを仮定)
    cov_matrix = np.cov(all_features, rowvar=False, bias=False)
    
    # 固有値分解 (np.linalg.eigh は対称行列用で安定)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 固有値が負の値にならないようにクリップ（数値安定性のため）
    eigenvalues[eigenvalues < 1e-8] = 1e-8
    
    # ホワイトニング変換行列（ZCA Whitening）
    # ZCA = V * D^-0.5 * V^T
    whitening_matrix = eigenvectors @ np.diag(1. / np.sqrt(eigenvalues)) @ eigenvectors.T
    
    return {'mean': mean_vector, 'matrix': whitening_matrix}



def precompute_gallery_embeddings(model, gallery_by_category, whitening_params=None):
    """ギャラリーの埋め込みを事前計算（高速版）"""
    print("[INFO] Pre-computing gallery embeddings...")
    
    processed_gallery = {}
    
    for cat_id, gallery in gallery_by_category.items():
        if 'features' not in gallery or len(gallery['features']) == 0:
            continue
            
        gallery_features = gallery['features']  # 既にCLIP特徴量
        print(f"  Category {cat_id}: Processing {len(gallery_features)} items")
        
        # バッチ処理でギャラリー特徴量をモデル埋め込みに変換
        batch_size = 64
        all_embeddings = []
        
        for i in range(0, len(gallery_features), batch_size):
            batch_features = gallery_features[i:i + batch_size]
            # [batch_size, 1, feature_dim] の形状でモデルに入力
            batch_input = np.expand_dims(batch_features, axis=1)
            
            # モデルで埋め込み計算
            batch_predictions = model({'query_features': tf.constant(batch_input)}, training=False)
            # 該当カテゴリの埋め込みを取得
            batch_embeddings = batch_predictions[:, cat_id - 1, :].numpy()
            all_embeddings.append(batch_embeddings)
        
        gallery_embeddings = np.concatenate(all_embeddings, axis=0)
        
        # ホワイトニング適用（オプション）
        if whitening_params is not None:
            gallery_embeddings = apply_whitening(
                gallery_embeddings, 
                whitening_params['mean'], 
                whitening_params['matrix']
            )
            print(f"    Applied whitening to category {cat_id}")
        
        processed_gallery[cat_id] = {
            'ids': gallery['ids'],
            'features': gallery_embeddings  # これで埋め込み済み
        }
    
    return processed_gallery


# def evaluate_model_comprehensive(model, test_items, gallery_by_category, dataset_type, use_weighted_topk=False, category_centers=None):
#     print("[INFO] 🎯 Starting comprehensive evaluation...")
#     config = get_dataset_config(dataset_type)
#     min_cat, max_cat = config['category_range']

#     # 重み付きメトリクス用の閾値
#     ALPHA_RANK_THRESHOLD_PERCENT = 5  # 許容アイテム基準（正解との類似度順位）
#     BETA_THRESHOLD = 0.90             # 予測成功基準（予測との類似度）
#     EPSILON = 1e-8

#     evaluated_queries_details = []

#     queries_by_category = defaultdict(list)

#     for item in tqdm(test_items, desc="Organizing queries"):
#         for target_cat, target_id in zip(item['target_categories'], item['target_item_ids']):
#             target_cat_int = int(target_cat)
#             if not (min_cat <= target_cat_int <= max_cat): continue
#             target_id_str = str(target_id)
#             if target_cat_int in gallery_by_category and target_id_str in gallery_by_category[target_cat_int]['ids']:
#                 query_data = {'query_features': item['query_features'], 'target_id': target_id_str}
#                 queries_by_category[target_cat_int].append(query_data)

#     # # ホワイトニングパラメータ読み込み
#     # whitening_params_path = 'training_whitening_params.pkl'
#     # if os.path.exists(whitening_params_path):
#     #     whitening_params = load_whitening_params(whitening_params_path)
#     #     print("[INFO] ✅ Loaded training-based whitening parameters")
#     #     processed_gallery = precompute_gallery_embeddings(model, gallery_by_category, whitening_params)
#     #     use_whitening = True
#     # else:
#     #     print("[INFO] ⚠️ No whitening parameters found, using original features")
#     #     whitening_params = None
#     #     processed_gallery = None
#     #     use_whitening = False
    
#     category_results = {}
#     all_ranks = []
#     all_gallery_sizes = []

#     predictions_by_category = defaultdict(list)

#     # 重み付きメトリクス用の蓄積変数
#     if use_weighted_topk:
#         all_weighted_scores = {k: [] for k in [1, 5, 10, 20]}
#         all_acceptance_counts = []
#         all_success_counts = []
    
#     # デバッグ用
#     debug_count = 0
#     total_processed = 0
    
#     for cat_id, queries in tqdm(queries_by_category.items(), desc="Evaluating categories"):
#         if not queries: 
#             continue
        
#         gallery = gallery_by_category[cat_id]
            
#         gallery_size = len(gallery['ids'])
#         all_gallery_sizes.append(gallery_size)
        
#         # K%の閾値を計算（Standard用）
#         top_1_percent_threshold = max(1, int(gallery_size * 0.01))
#         top_5_percent_threshold = max(1, int(gallery_size * 0.05))
#         top_10_percent_threshold = max(1, int(gallery_size * 0.10))
#         top_20_percent_threshold = max(1, int(gallery_size * 0.20))
        
#         query_features_batch = np.array([q['query_features'] for q in queries])
#         predictions_batch = model({'query_features': tf.constant(query_features_batch)}, training=False).numpy()
#         pred_vectors = predictions_batch[:, cat_id - 1, :]

#         # ホワイトニング有効時のみ予測ベクトルにホワイトニング適用
#         # if use_whitening and whitening_params is not None:
#         #     pred_vectors = apply_whitening(pred_vectors, whitening_params['mean'], whitening_params['matrix'])

#         # 予測ベクトルを正規化
#         # pred_norms = np.linalg.norm(pred_vectors, axis=1, keepdims=True)
#         # pred_vectors_normalized = pred_vectors / np.maximum(pred_norms, EPSILON)

#         # ★★★ 追加: 正規化された予測ベクトルをカテゴリごとに収集 ★★★
#         predictions_by_category[cat_id].extend(pred_vectors)

#         # ==========================================================
#         # ★★★ ここからデバッグ出力の追加 (修正版) ★★★
#         # ==========================================================
#         if cat_id in predictions_by_category and len(predictions_by_category[cat_id]) > 1:
#             cat_pred_vectors = np.array(predictions_by_category[cat_id])
#             num_samples = len(cat_pred_vectors)
#             num_dimensions = cat_pred_vectors.shape[1]
            
#             # ★★★ 修正: 中心化されたベクトルを計算 ★★★
#             cat_mean = np.mean(cat_pred_vectors, axis=0)
#             centered_vectors = cat_pred_vectors - cat_mean
            
#             # 予測ベクトルのノルムの平均と標準偏差
#             pred_vector_norms = np.linalg.norm(cat_pred_vectors, axis=1)
#             # print(f"\n[DEBUG-CAT-{cat_id}] ----------------------------------------")
#             # print(f"[DEBUG-CAT-{cat_id}] Analysis of Predicted Vectors ({num_samples} samples, {num_dimensions} dims)")
#             # print(f"[DEBUG-CAT-{cat_id}] - Average Norm (Original): {np.mean(pred_vector_norms):.6f}")
#             # print(f"[DEBUG-CAT-{cat_id}] - Norm Std Dev (Original): {np.std(pred_vector_norms):.6f}")
            
#             # 各次元の平均 (中心化後)
#             # centered_mean = np.mean(centered_vectors, axis=0) # これはほぼ0になるはず
#             # print(f"[DEBUG-CAT-{cat_id}] - Overall Mean Norm (Original): {np.linalg.norm(cat_mean):.6f}")
            
#             # 共分散行列の確認 (中心化されたベクトルを使用)
#             if num_samples > num_dimensions:
#                 try:
#                     # 共分散行列を計算 (中心化済みデータを使用)
#                     cov_matrix = np.cov(centered_vectors, rowvar=False, bias=False)
#                     # print(f"[DEBUG-CAT-{cat_id}] - Covariance Matrix (top-left 5x5, from centered vectors):\n{cov_matrix[:5, :5]}")
                    
#                     # 固有値の確認
#                     eigenvalues = np.linalg.eigvalsh(cov_matrix)
#                     # print(f"[DEBUG-CAT-{cat_id}] - Top 5 Eigenvalues: {np.sort(eigenvalues)[::-1][:5]}")
#                     # print(f"[DEBUG-CAT-{cat_id}] - Smallest 5 Eigenvalues: {np.sort(eigenvalues)[:5]}")

#                     # 行列式
#                     generalized_variance = np.linalg.det(cov_matrix)
#                     # print(f"[DEBUG-CAT-{cat_id}] - Generalized Variance (Determinant): {generalized_variance:.6e}")
#                     sign, log_det = np.linalg.slogdet(cov_matrix)
#                     # print(f"[DEBUG-CAT-{cat_id}] - Log Determinant: {log_det:.6f}")
#                 except np.linalg.LinAlgError as e:
#                     print(f"[DEBUG-CAT-{cat_id}] ⚠️ LinAlgError: Could not compute covariance matrix/eigenvalues. Error: {e}")
#             else:
#                 print(f"[DEBUG-CAT-{cat_id}] ℹ️ Not enough samples ({num_samples}) to compute covariance matrix for {num_dimensions} dimensions.")
#         # ==========================================================
#         # ★★★ ここまでデバッグ出力ブロックを修正 ★★★
#         # ==========================================================
        
#         # 類似度計算（正規化された予測ベクトルとギャラリーの類似度）
#         pred_similarities_batch = np.dot(pred_vectors, gallery['features'].T)
#         sorted_indices_batch = np.argsort(pred_similarities_batch, axis=1)[:, ::-1]
        
#         # ... (以降のコードは変更なし) ...
#         # (以降は元のコードと同じなので省略します。上記で追加したデバッグ出力が、
#         # 各カテゴリの評価ループ内で実行されます。)

#         # 重み付きメトリクス計算用
#         if use_weighted_topk:
#             category_weighted_scores = {k: [] for k in [1, 5, 10, 20]}
#             category_acceptance_counts = []
#             category_success_counts = []
        
#         ranks = []
#         for i, query in enumerate(queries):
#             sorted_gallery_ids = gallery['ids'][sorted_indices_batch[i]]
#             pred_similarities = pred_similarities_batch[i]
            
#             # 正解アイテムのランクを取得（Standard用）
#             rank_list = np.where(sorted_gallery_ids == query['target_id'])[0]
#             if len(rank_list) > 0: 
#                 true_rank = rank_list[0] + 1
#                 ranks.append(true_rank)
#                 total_processed += 1
                
#                 # ★★★ 重み付きメトリクス計算（順位ベースに修正） ★★★
#                 if use_weighted_topk:
#                     # 正解アイテムのインデックスと特徴量を取得
#                     correct_idx = np.where(gallery['ids'] == query['target_id'])[0][0]
#                     correct_feature = gallery['features'][correct_idx]
                    
#                     # Step 1: 正解アイテムとの類似度を計算し、類似度順に並び替え
#                     correct_similarities = np.dot(gallery['features'], correct_feature)
                    
#                     # ★★★ 修正箇所: 類似度の絶対値ではなく順位で許容アイテムを定義 ★★★
#                     # Step 2: 類似度でソートし、Top K% の閾値を決定
#                     acceptable_rank_threshold = max(1, int(gallery_size * ALPHA_RANK_THRESHOLD_PERCENT / 100))
                    
#                     # 類似度が高い順のインデックスを取得
#                     sorted_correct_indices = np.argsort(correct_similarities)[::-1]
                    
#                     # Top K% のインデックスを許容アイテムとする
#                     acceptable_indices = sorted_correct_indices[:acceptable_rank_threshold]
#                     acceptable_similarities = correct_similarities[acceptable_indices]
#                     # ★★★ 修正箇所ここまで ★★★
                    
#                     if len(acceptable_indices) > 0:
#                         # Step 4: 予測ベクトルと許容アイテムの類似度
#                         pred_to_acceptable = pred_similarities[acceptable_indices]
                        
#                         # Step 5: 各TopK%に対してWeighted計算
#                         for k in [1, 5, 10, 20]:
#                             # Standard TopK閾値
#                             standard_threshold = eval(f"top_{k}_percent_threshold")
                            
#                             # 予測類似度の順位に基づく閾値
#                             sorted_pred_sims = np.sort(pred_similarities)[::-1]
#                             topk_count = max(1, int(gallery_size * k / 100))
#                             if topk_count <= len(sorted_pred_sims):
#                                 similarity_threshold = sorted_pred_sims[topk_count - 1]
#                             else:
#                                 similarity_threshold = 0.0
                            
#                             # 成功したアイテムのマスク
#                             topk_success_mask = pred_to_acceptable >= similarity_threshold
                            
#                             # 成功したアイテムの重みを計算
#                             success_weights = acceptable_similarities[topk_success_mask]
#                             total_weights = acceptable_similarities
                            
#                             if np.sum(total_weights) > EPSILON:
#                                 weighted_score = np.sum(success_weights) / np.sum(total_weights)
#                             else:
#                                 weighted_score = 0.0
                            
#                             category_weighted_scores[k].append(weighted_score)
                        
#                         category_acceptance_counts.append(len(acceptable_indices))
                        
#                         # β=-1.0なら必ず成功
#                         if BETA_THRESHOLD <= -0.999:
#                             category_success_counts.append(len(acceptable_indices))
#                         else:
#                             category_success_counts.append(np.sum(pred_to_acceptable >= BETA_THRESHOLD))
#                     else:
#                         # 許容アイテムがない場合
#                         for k in [1, 5, 10, 20]:
#                             category_weighted_scores[k].append(0.0)
#                         category_acceptance_counts.append(0)
#                         category_success_counts.append(0)
            
#         if ranks:
#             all_ranks.extend(ranks)
#             ranks_np = np.array(ranks)
            
#             # 基本的なカテゴリ結果を作成
#             cat_result = {
#                 'count': len(ranks_np), 
#                 'gallery_size': gallery_size,
#                 'mrr': np.mean(1.0 / ranks_np), 
#                 'r_at_1': np.mean(ranks_np <= top_1_percent_threshold),
#                 'r_at_5': np.mean(ranks_np <= top_5_percent_threshold),
#                 'r_at_10': np.mean(ranks_np <= top_10_percent_threshold),
#                 'r_at_20': np.mean(ranks_np <= top_20_percent_threshold),
#                 'mnr': np.mean(ranks_np), 
#                 'mdr': np.median(ranks_np),
#                 'rsum': np.sum(ranks_np),
#                 'top_1_percent_threshold': top_1_percent_threshold,
#                 'top_5_percent_threshold': top_5_percent_threshold,
#                 'top_10_percent_threshold': top_10_percent_threshold,
#                 'top_20_percent_threshold': top_20_percent_threshold,
#             }
#             # ★★★ 修正箇所: セントロイドからの距離分散の代わりに、予測ベクトルの共分散行列の行列式を計算 ★★★
#             if cat_id in predictions_by_category:
#                 cat_pred_vectors = np.array(predictions_by_category[cat_id])
                
#                 # ★★★ 修正: 中心化されたベクトルを使用 ★★★
#                 if len(cat_pred_vectors) > 1:
#                     cat_mean = np.mean(cat_pred_vectors, axis=0)
#                     centered_vectors = cat_pred_vectors - cat_mean
#                 else:
#                     centered_vectors = cat_pred_vectors
                
#                 num_pred_dimensions = centered_vectors.shape[1] 

#                 if len(centered_vectors) > num_pred_dimensions: 
#                     try:
#                         cov_matrix = np.cov(centered_vectors, rowvar=False, bias=False)
#                         generalized_variance = np.linalg.det(cov_matrix)
#                         sign, log_generalized_variance = np.linalg.slogdet(cov_matrix)

#                         cat_result['generalized_variance'] = float(generalized_variance)
#                         cat_result['log_generalized_variance'] = float(log_generalized_variance)
#                     except np.linalg.LinAlgError as e:
#                         print(f"⚠️ Warning: Could not compute covariance determinant for category {cat_id}: {e}")
#                         cat_result['generalized_variance'] = 0.0
#                         cat_result['log_generalized_variance'] = -np.inf
#                 else:
#                     cat_result['generalized_variance'] = 0.0
#                     cat_result['log_generalized_variance'] = -np.inf
#             else:
#                 cat_result['generalized_variance'] = 0.0
#                 cat_result['log_generalized_variance'] = -np.inf
#             # ★★★ 修正箇所ここまで ★★★

#             # 重み付きメトリクスを追加
#             if use_weighted_topk and any(len(scores) > 0 for scores in category_weighted_scores.values()):
#                 # 各TopK%の重み付き正解率をカテゴリ結果に追加
#                 for k in [1, 5, 10, 20]:
#                     if len(category_weighted_scores[k]) > 0:
#                         cat_result[f'weighted_r_at_{k}'] = np.mean(category_weighted_scores[k])
#                         all_weighted_scores[k].extend(category_weighted_scores[k])
#                     else:
#                         cat_result[f'weighted_r_at_{k}'] = 0.0
                
#                 # その他の統計
#                 acceptance_counts_np = np.array(category_acceptance_counts)
#                 success_counts_np = np.array(category_success_counts)
                
#                 cat_result.update({
#                     'avg_acceptance_count': np.mean(acceptance_counts_np) if len(acceptance_counts_np) > 0 else 0.0,
#                     'avg_success_count': np.mean(success_counts_np) if len(success_counts_np) > 0 else 0.0,
#                     'acceptance_rate': np.mean(acceptance_counts_np) / gallery_size if gallery_size > 0 and len(acceptance_counts_np) > 0 else 0.0,
#                     'alpha_rank_threshold_percent': ALPHA_RANK_THRESHOLD_PERCENT,
#                     'beta_threshold': BETA_THRESHOLD,
#                 })
                
#                 all_acceptance_counts.extend(category_acceptance_counts)
#                 all_success_counts.extend(category_success_counts)
            
#             category_results[str(cat_id)] = cat_result
    
#     # 全体結果の計算
#     overall_results = {}
#     if all_ranks:
#         all_ranks_np = np.array(all_ranks)
#         avg_gallery_size = np.mean(all_gallery_sizes) if all_gallery_sizes else 0
        
#         overall_top_1_percent = max(1, int(avg_gallery_size * 0.01))
#         overall_top_5_percent = max(1, int(avg_gallery_size * 0.05))
#         overall_top_10_percent = max(1, int(avg_gallery_size * 0.10))
#         overall_top_20_percent = max(1, int(avg_gallery_size * 0.20))
        
#         overall_results = {
#             'total_queries': len(all_ranks_np), 
#             'successful_predictions': len(all_ranks_np),
#             'average_gallery_size': avg_gallery_size,
#             'mrr': np.mean(1.0 / all_ranks_np), 
#             'r_at_1': np.mean(all_ranks_np <= overall_top_1_percent),
#             'r_at_5': np.mean(all_ranks_np <= overall_top_5_percent),
#             'r_at_10': np.mean(all_ranks_np <= overall_top_10_percent),
#             'r_at_20': np.mean(all_ranks_np <= overall_top_20_percent),
#             'mnr': np.mean(all_ranks_np), 
#             'mdr': np.median(all_ranks_np),
#             'rsum': np.sum(all_ranks_np),
#             'overall_top_1_percent_threshold': overall_top_1_percent,
#             'overall_top_5_percent_threshold': overall_top_5_percent,
#             'overall_top_10_percent_threshold': overall_top_10_percent,
#             'overall_top_20_percent_threshold': overall_top_20_percent,
#         }
#         # ★★★ 修正箇所: 全体の一般化分散を計算して全体結果に追加 (中心化) ★★★
#         all_category_pred_vectors = []
#         all_category_labels = []
#         for cat_id in predictions_by_category:
#             if predictions_by_category[cat_id]: # カテゴリに予測ベクトルがあるか確認
#                 all_category_pred_vectors.extend(predictions_by_category[cat_id])
#                 all_category_labels.extend([cat_id] * len(predictions_by_category[cat_id])) 

        
#         if len(all_category_pred_vectors) > 1:
#             all_category_pred_vectors_np = np.array(all_category_pred_vectors)
#             all_category_labels_np = np.array(all_category_labels)

#         # # =======================================================================
#         # # ★★★ ここから可視化コードを追加 ★★★
#         # # =======================================================================
#         # try:
#         #     from sklearn.decomposition import PCA
#         #     import matplotlib.pyplot as plt
#         #     import seaborn as sns
            
#         #     print("\n[INFO] 🎨 Starting PCA and t-SNE visualization...")

#         #     # 1. PCA (2次元)
#         #     pca_model = PCA(n_components=2)
#         #     pca_embeddings = pca_model.fit_transform(all_category_pred_vectors_np)
            
#         #     plt.figure(figsize=(10, 8))
#         #     # カテゴリごとに色分けしてプロット
#         #     sns.scatterplot(
#         #         x=pca_embeddings[:, 0], 
#         #         y=pca_embeddings[:, 1], 
#         #         hue=all_category_labels_np,
#         #         palette=sns.color_palette("hsv", n_colors=len(np.unique(all_category_labels_np))),
#         #         legend="full",
#         #         s=5
#         #     )
#         #     plt.title('PCA of Predicted Embeddings (2D Projection)')
#         #     plt.xlabel(f'Principal Component 1 (Variance Explained: {pca_model.explained_variance_ratio_[0]*100:.2f}%)')
#         #     plt.ylabel(f'Principal Component 2 (Variance Explained: {pca_model.explained_variance_ratio_[1]*100:.2f}%)')
#         #     plt.grid(True)
#         #     # plt.show() # JUPYTER NOTEBOOKなどで実行する場合
#         #     pca_plot_path = os.path.join('experiments', 'IQON3000', 'B128_L2_H2_LR1e-04_Cycle0.0_Seed42', 'pca_embedding_space_custom.png')
#         #     plt.savefig(pca_plot_path)
#         #     print(f"[INFO] ✅ Custom PCA visualization saved to: {pca_plot_path}")
#         #     plt.close()

#             # 2. PCAの累積寄与率を確認
#         #     pca_full = PCA(n_components=None) # 全ての主成分を計算
#         #     pca_full.fit(all_category_pred_vectors_np)
#         #     explained_variance_ratio_cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            
#         #     # 90%の分散を説明するのに必要な次元数を取得
#         #     num_dims_for_90_percent_variance = np.argmax(explained_variance_ratio_cumsum >= 0.90) + 1
#         #     print(f"[INFO] 📊 Dimensions to explain 90% of variance: {num_dims_for_90_percent_variance} out of {num_dimensions}")
            
#         #     # 3. t-SNE (2次元)
#         #     # t-SNEは計算に時間がかかるため、サブサンプルで実行することも検討
#         #     if len(all_category_pred_vectors) > 5000:
#         #         print("[INFO] ⚠️ Large dataset for t-SNE, subsampling to 5000 points...")
#         #         indices = np.random.choice(len(all_category_pred_vectors), 5000, replace=False)
#         #         subset_embeddings = all_category_pred_vectors_np[indices]
#         #         subset_labels = all_category_labels_np[indices]
#         #     else:
#         #         subset_embeddings = all_category_pred_vectors_np
#         #         subset_labels = all_category_labels_np
            
#         #     from sklearn.manifold import TSNE
#         #     tsne_model = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
#         #     tsne_embeddings = tsne_model.fit_transform(subset_embeddings)
            
#         #     plt.figure(figsize=(10, 8))
#         #     sns.scatterplot(
#         #         x=tsne_embeddings[:, 0], 
#         #         y=tsne_embeddings[:, 1], 
#         #         hue=subset_labels,
#         #         palette=sns.color_palette("hsv", n_colors=len(np.unique(subset_labels))),
#         #         legend="full",
#         #         s=5
#         #     )
#         #     plt.title('t-SNE of Predicted Embeddings (2D Projection)')
#         #     plt.xlabel('t-SNE Dimension 1')
#         #     plt.ylabel('t-SNE Dimension 2')
#         #     plt.grid(True)
#         #     # plt.show() # ここフォルダ指定してしまってる。
#         #     tsne_plot_path = os.path.join('experiments', 'IQON3000', 'B128_L2_H2_LR1e-04_Cycle0.0_Seed42', 'tsne_embedding_space_custom.png')
#         #     plt.savefig(tsne_plot_path)
#         #     print(f"[INFO] ✅ Custom t-SNE visualization saved to: {tsne_plot_path}")
#         #     plt.close()

#         # except ImportError:
#         #     print("[INFO] ⚠️ Matplotlib or scikit-learn not found. Skipping visualization.")
#         # =======================================================================
#         # ★★★ ここまで可視化コードを追加 ★★★
#         # =======================================================================
            
#             # ★★★ 修正: 全体の平均を引いてから共分散行列を計算 ★★★
#             overall_mean = np.mean(all_category_pred_vectors_np, axis=0)
#             overall_centered_vectors = all_category_pred_vectors_np - overall_mean
            
#             num_overall_dimensions = overall_centered_vectors.shape[1]

#             if len(overall_centered_vectors) > num_overall_dimensions:
#                 try:
#                     overall_cov_matrix = np.cov(overall_centered_vectors, rowvar=False, bias=False)
#                     overall_generalized_variance = np.linalg.det(overall_cov_matrix)
#                     overall_sign, overall_log_generalized_variance = np.linalg.slogdet(overall_cov_matrix)

#                     overall_results['overall_generalized_variance'] = float(overall_generalized_variance)
#                     overall_results['overall_log_generalized_variance'] = float(overall_log_generalized_variance)
#                 except np.linalg.LinAlgError as e:
#                     print(f"⚠️ Warning: Could not compute overall covariance determinant: {e}")
#                     overall_results['overall_generalized_variance'] = 0.0
#                     overall_results['overall_log_generalized_variance'] = -np.inf
#             else:
#                 overall_results['overall_generalized_variance'] = 0.0
#                 overall_results['overall_log_generalized_variance'] = -np.inf
#         else:
#             overall_results['overall_generalized_variance'] = 0.0
#             overall_results['overall_log_generalized_variance'] = -np.inf
#         # ★★★ 修正箇所ここまで ★★★

#         # 重み付きメトリクスを全体結果に追加
#         if use_weighted_topk and any(len(scores) > 0 for scores in all_weighted_scores.values()):
#             # 全体の重み付きTopK%正解率
#             for k in [1, 5, 10, 20]:
#                 if len(all_weighted_scores[k]) > 0:
#                     overall_results[f'weighted_r_at_{k}'] = np.mean(all_weighted_scores[k])
#                 else:
#                     overall_results[f'weighted_r_at_{k}'] = 0.0
            
#             # デバッグ: 期待値との比較
#             print(f"\n[FINAL DEBUG] α_rank={ALPHA_RANK_THRESHOLD_PERCENT}%, β={BETA_THRESHOLD}:")
#             print(f"   Total processed queries: {total_processed}")
            
#             for k in [1, 5, 10, 20]:
#                 std_val = overall_results[f'r_at_{k}']
#                 weighted_val = overall_results.get(f'weighted_r_at_{k}', 0.0)
#                 print(f"   Top-{k}%: Standard={std_val:.4f}, Weighted={weighted_val:.4f}")
            
#             # その他の全体統計
#             if all_acceptance_counts and all_success_counts:
#                 all_acceptance_counts_np = np.array(all_acceptance_counts)
#                 all_success_counts_np = np.array(all_success_counts)
                
#                 overall_results.update({
#                     'avg_acceptance_count': np.mean(all_acceptance_counts_np),
#                     'avg_success_count': np.mean(all_success_counts_np),
#                     'overall_acceptance_rate': np.mean(all_acceptance_counts_np) / avg_gallery_size if avg_gallery_size > 0 else 0.0,
#                     'alpha_rank_threshold_percent': ALPHA_RANK_THRESHOLD_PERCENT,
#                     'beta_threshold': BETA_THRESHOLD,
#                 })
    
#     return {'dataset': dataset_type, 'overall': overall_results, 'categories': category_results}


def evaluate_model_comprehensive(model, test_items, gallery_by_category, dataset_type, use_weighted_topk=False, category_centers=None):
    """
    モデルの評価を包括的に行う関数。
    use_weighted_topk=Trueの場合、許容アイテムの範囲(ALPHA)と成功判定の範囲(K)を連動させる
    「ランキング整合性」を測定する方式で重み付きTopK精度を計算する。
    """
    print("[INFO] 🎯 Starting comprehensive evaluation (Dynamic ALPHA method)...")
    # config = get_dataset_config(dataset_type)
    # min_cat, max_cat = config['category_range']
    min_cat, max_cat = 0, 10000 # 仮の値
    EPSILON = 1e-8

    queries_by_category = defaultdict(list)
    for item in tqdm(test_items, desc="Organizing queries"):
        for target_cat, target_id in zip(item['target_categories'], item['target_item_ids']):
            target_cat_int = int(target_cat)
            if not (min_cat <= target_cat_int <= max_cat): continue
            target_id_str = str(target_id)
            if target_cat_int in gallery_by_category and target_id_str in gallery_by_category[target_cat_int]['ids']:
                query_data = {'query_features': item['query_features'], 'target_id': target_id_str}
                queries_by_category[target_cat_int].append(query_data)

    category_results = {}
    all_ranks = []
    all_gallery_sizes = []
    predictions_by_category = defaultdict(list)

    if use_weighted_topk:
        all_weighted_scores = {k: [] for k in [1, 5, 10, 20]}
    
    for cat_id, queries in tqdm(queries_by_category.items(), desc="Evaluating categories"):
        if not queries: 
            continue
        
        gallery = gallery_by_category[cat_id]
        gallery_size = len(gallery['ids'])
        all_gallery_sizes.append(gallery_size)
        
        # Standard Top-K%用の閾値
        top_1_percent_threshold = max(1, int(gallery_size * 0.01))
        top_5_percent_threshold = max(1, int(gallery_size * 0.05))
        top_10_percent_threshold = max(1, int(gallery_size * 0.10))
        top_20_percent_threshold = max(1, int(gallery_size * 0.20))
        
        query_features_batch = np.array([q['query_features'] for q in queries])
        predictions_batch = model({'query_features': tf.constant(query_features_batch)}, training=False).numpy()
        pred_vectors = predictions_batch[:, cat_id - 1, :]
        predictions_by_category[cat_id].extend(pred_vectors)
        
        pred_similarities_batch = np.dot(pred_vectors, gallery['features'].T)
        sorted_indices_batch = np.argsort(pred_similarities_batch, axis=1)[:, ::-1]

        if use_weighted_topk:
            category_weighted_scores = {k: [] for k in [1, 5, 10, 20]}
        
        ranks = []
        for i, query in enumerate(queries):
            sorted_gallery_ids = gallery['ids'][sorted_indices_batch[i]]
            pred_similarities = pred_similarities_batch[i]
            
            rank_list = np.where(sorted_gallery_ids == query['target_id'])[0]
            if len(rank_list) > 0: 
                true_rank = rank_list[0] + 1
                ranks.append(true_rank)
                
                if use_weighted_topk:
                    correct_idx = np.where(gallery['ids'] == query['target_id'])[0][0]
                    correct_feature = gallery['features'][correct_idx]
                    correct_similarities = np.dot(gallery['features'], correct_feature)
                    sorted_correct_indices = np.argsort(correct_similarities)[::-1]
                    sorted_pred_sims = np.sort(pred_similarities)[::-1]

                    for k in [1, 5, 10, 20]:
                        # 1. 許容アイテムを「k%」に合わせて動的に定義
                        acceptable_rank_threshold = max(1, int(gallery_size * k / 100))
                        acceptable_indices = sorted_correct_indices[:acceptable_rank_threshold]
                        
                        if len(acceptable_indices) == 0:
                            category_weighted_scores[k].append(0.0)
                            continue

                        # 2. 成功の基準（予測の上位k%）を定義
                        similarity_threshold = sorted_pred_sims[acceptable_rank_threshold - 1]
                        
                        # 3. 重み付きスコアを計算
                        acceptable_similarities = correct_similarities[acceptable_indices]
                        pred_to_acceptable = pred_similarities[acceptable_indices]
                        
                        topk_success_mask = pred_to_acceptable >= similarity_threshold
                        success_weights = acceptable_similarities[topk_success_mask]
                        total_weights = acceptable_similarities
                        
                        if np.sum(total_weights) > EPSILON:
                            weighted_score = np.sum(success_weights) / np.sum(total_weights)
                        else:
                            weighted_score = 0.0
                        
                        category_weighted_scores[k].append(weighted_score)

        if ranks:
            ranks_np = np.array(ranks)
            all_ranks.extend(ranks)
            
            cat_result = {
                'count': len(ranks_np), 
                'gallery_size': gallery_size,
                'mrr': np.mean(1.0 / ranks_np), 
                'r_at_1': np.mean(ranks_np <= top_1_percent_threshold),
                'r_at_5': np.mean(ranks_np <= top_5_percent_threshold),
                'r_at_10': np.mean(ranks_np <= top_10_percent_threshold),
                'r_at_20': np.mean(ranks_np <= top_20_percent_threshold),
                'mnr': np.mean(ranks_np), 
                'mdr': np.median(ranks_np)
            }
            
            if use_weighted_topk:
                for k in [1, 5, 10, 20]:
                    if len(category_weighted_scores[k]) > 0:
                        cat_result[f'weighted_r_at_{k}'] = np.mean(category_weighted_scores[k])
                        all_weighted_scores[k].extend(category_weighted_scores[k])
                    else:
                        cat_result[f'weighted_r_at_{k}'] = 0.0

            category_results[str(cat_id)] = cat_result
    
    overall_results = {}
    if all_ranks:
        all_ranks_np = np.array(all_ranks)
        avg_gallery_size = np.mean(all_gallery_sizes) if all_gallery_sizes else 0
        
        overall_top_1_percent = max(1, int(avg_gallery_size * 0.01))
        overall_top_5_percent = max(1, int(avg_gallery_size * 0.05))
        overall_top_10_percent = max(1, int(avg_gallery_size * 0.10))
        overall_top_20_percent = max(1, int(avg_gallery_size * 0.20))
        
        overall_results = {
            'total_queries': len(all_ranks_np), 
            'average_gallery_size': avg_gallery_size,
            'mrr': np.mean(1.0 / all_ranks_np), 
            'r_at_1': np.mean(all_ranks_np <= overall_top_1_percent),
            'r_at_5': np.mean(all_ranks_np <= overall_top_5_percent),
            'r_at_10': np.mean(all_ranks_np <= overall_top_10_percent),
            'r_at_20': np.mean(all_ranks_np <= overall_top_20_percent),
            'mnr': np.mean(all_ranks_np), 
            'mdr': np.median(all_ranks_np)
        }
        
        if use_weighted_topk:
            for k in [1, 5, 10, 20]:
                if len(all_weighted_scores[k]) > 0:
                    overall_results[f'weighted_r_at_{k}'] = np.mean(all_weighted_scores[k])
                else:
                    overall_results[f'weighted_r_at_{k}'] = 0.0
    
    return {'dataset': dataset_type, 'overall': overall_results, 'categories': category_results}

    
def evaluate_model(model, test_data, output_dir: str, data_dir: str, train_generator=None, use_weighted_topk=False, category_centers=None):
    """
    Main evaluation pipeline with optional weighted TopK metrics.
    """
    print(f"\n[INFO] 🚀 Starting model evaluation pipeline...")
    if use_weighted_topk:
        print(f"[INFO] ✨ Weighted TopK metrics enabled")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        test_items, gallery, dataset_type = collect_test_data(test_data)
        config = get_dataset_config(dataset_type)

        # ★ use_weighted_topk オプションを評価関数に渡す
        results = evaluate_model_comprehensive(model, test_items, gallery, dataset_type, use_weighted_topk=use_weighted_topk, category_centers=category_centers)
        
        if results:
            # ★ JSONの保存で循環参照エラーが発生するため、まず表示のみ実行
            try:
                print("\n================================================================================")
                print("🎯 EVALUATION RESULTS")
                print("================================================================================")
                
                overall_results = results['overall']
                category_results = results['categories']
                
                # Overall Performance の表示
                print("\n📊 Overall Performance:")
                for key, value in overall_results.items():
                    if isinstance(value, float):
                        print(f"   {key}: {value:.4f}")
                    else:
                        print(f"   {key}: {value}")
                
                # Standard TopK% Accuracy の表示
                print("\n📈 Standard TopK% Accuracy:")
                for k in model.k_values: # Assuming model.k_values is available
                    print(f"   Top-{k}%: {overall_results.get(f'r_at_{k}', 0.0):.4f} ({overall_results.get(f'r_at_{k}', 0.0)*100:.2f}%)")
                
                # Weighted TopK% Accuracy の表示
                if 'weighted_r_at_1' in overall_results:
                    print("\n✨ Weighted TopK% Accuracy:")
                    for k in model.k_values:
                        print(f"   Weighted Top-{k}%: {overall_results.get(f'weighted_r_at_{k}', 0.0):.4f} ({overall_results.get(f'weighted_r_at_{k}', 0.0)*100:.2f}%)")
                
                # カテゴリごとのパフォーマンス表示
                print("\n📂 Category-wise Performance:")
                print("--------------------------------------------------------------------------------")
                for cat_id_str, cat_metrics in category_results.items():
                    print(f"\n🏷️  Category {cat_id_str} (Category {cat_id_str}):")
                    print(f"    Queries: {cat_metrics['count']}, Gallery: {cat_metrics['gallery_size']}")
                    print(f"    MRR: {cat_metrics['mrr']:.4f}")
                    
                    # Standard
                    standard_metrics = " | ".join([f"Top-{k}%: {cat_metrics[f'r_at_{k}']:.3f}" for k in model.k_values])
                    print(f"    Standard: {standard_metrics}")
                    
                    # Weighted
                    if 'weighted_r_at_1' in cat_metrics:
                        weighted_metrics = " | ".join([f"W-Top-{k}%: {cat_metrics[f'weighted_r_at_{k}']:.3f}" for k in model.k_values])
                        print(f"    Weighted: {weighted_metrics}")
                        
                    # セントロイド分散のメトリクスを追加
                    if 'centroid_distance_variance' in cat_metrics:
                        print(f"    Centroid Variance: {cat_metrics['centroid_distance_variance']:.6f} | Avg Distance: {cat_metrics['avg_centroid_distance']:.6f}")
                        
                    # その他の詳細メトリクス
                    print(f"    Details: Acc-Count: {cat_metrics.get('avg_acceptance_count', 0.0):.1f} | Acc-Rate: {cat_metrics.get('acceptance_rate', 0.0):.3f}")
                    
                print("\n================================================================================")
                
            except Exception as display_error:
                print(f"[WARN] Display failed: {display_error}")
            
            # ★ 安全な形でCSVのみ保存
            try:
                print("[INFO] 💾 Saving CSV results...")
                save_csv_results_only(results, output_dir)
            except Exception as save_error:
                print(f"[WARN] CSV save failed: {save_error}")
            
            # ★ 可視化は後で実行
            try:
                generate_all_visualizations(model, results, test_items, gallery, config, output_dir, data_dir)
            except Exception as viz_error:
                print(f"[WARN] Visualization failed: {viz_error}")
        
        clear_memory()
        return results, test_items, gallery, dataset_type

    except Exception as e:
        print(f"[ERROR] ❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_csv_results_only(results, output_dir):
    """
    循環参照を避けてCSV形式のみで結果を保存
    """
    import pandas as pd
    
    csv_data = []
    
    # Overall results
    if 'overall' in results:
        overall = results['overall']
        row = {'category': 'overall'}
        # 数値のみを安全に抽出
        for key, value in overall.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                row[key] = float(value)
            elif isinstance(value, str):
                row[key] = value
        csv_data.append(row)
    
    # Category results
    if 'categories' in results:
        for cat_id, cat_results in results['categories'].items():
            row = {'category': cat_id}
            # 数値のみを安全に抽出
            for key, value in cat_results.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    row[key] = float(value)
                elif isinstance(value, str):
                    row[key] = value
            csv_data.append(row)
    
    # Save CSV
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, 'results_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"[INFO] ✅ CSV results saved to: {csv_path}")
        
        # 重み付きメトリクスがある場合は専用CSVも作成
        if any('weighted_' in col for col in df.columns):
            weighted_columns = ['category'] + [col for col in df.columns if 'weighted' in col or 'acceptance' in col or 'threshold' in col]
            if len(weighted_columns) > 1:
                weighted_df = df[weighted_columns]
                weighted_csv_path = os.path.join(output_dir, 'weighted_metrics_summary.csv')
                weighted_df.to_csv(weighted_csv_path, index=False)
                print(f"[INFO] ✅ Weighted metrics CSV saved to: {weighted_csv_path}")

