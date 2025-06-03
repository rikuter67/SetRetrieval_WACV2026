# data_generator.py - 条件付きネガティブサンプル版
import os
import gzip
import pickle
import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow.keras.utils import Sequence

def load_category_centers(center_path: str):
    """カテゴリ中心を読み込み"""
    if not os.path.exists(center_path):
        return None
    
    try:
        if center_path.endswith('.gz'):
            with gzip.open(center_path, 'rb') as f:
                centers = pickle.load(f)
        else:
            with open(center_path, 'rb') as f:
                centers = pickle.load(f)
        
        # 辞書形式の場合はNumPy配列に変換
        if isinstance(centers, dict):
            max_cat = max(centers.keys())
            feature_dim = len(list(centers.values())[0])
            center_array = np.zeros((max_cat, feature_dim), dtype=np.float32)
            for cat_id, vec in centers.items():
                if 1 <= cat_id <= max_cat:
                    center_array[cat_id-1] = np.array(vec, dtype=np.float32)
            return center_array
        else:
            return np.array(centers, dtype=np.float32)
            
    except Exception as e:
        print(f"Failed to load category centers: {e}")
        return None

class DataGenerator(Sequence):
    """条件付きネガティブサンプル生成データジェネレータ"""
    
    def __init__(self, split_path: str, batch_size: int = 32, shuffle: bool = True, 
                 seed: int = 42, center_path: str = None, neg_num: int = 10,
                 dataset_name: str = None, use_negatives: bool = True):
        
        self.split_path = split_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.neg_num = neg_num
        self.use_negatives = use_negatives  # CLNegが有効な場合のみTrue
        self.dataset_name = dataset_name or self._infer_dataset_name(split_path)
        self.rng = np.random.RandomState(seed)
        
        print(f"DataGenerator: use_negatives={use_negatives}, neg_num={neg_num if use_negatives else 0}")
        
        # データ読み込み
        self._load_data()
        
        # カテゴリ中心読み込み
        if center_path:
            self.cluster_centers = load_category_centers(center_path)
        else:
            self.cluster_centers = None
        
        # データセット固有の設定
        self._setup_dataset_config()
        
        # 条件付きネガティブプール構築
        if self.use_negatives:
            self._build_negative_pool()
            memory_mb = self._estimate_negative_pool_memory()
            print(f"Built negative pool: {len(self.negative_pool)} categories, ~{memory_mb:.1f}MB")
        else:
            self.negative_pool = {}
            print("Skipped negative pool construction (CLNeg disabled - major memory savings)")
        
        # インデックス初期化
        self.indexes = np.arange(len(self.scene_ids))
        if shuffle:
            self.rng.shuffle(self.indexes)
    
    def _estimate_negative_pool_memory(self):
        """ネガティブプールのメモリ使用量推定"""
        total_samples = sum(len(pool) for pool in self.negative_pool.values())
        memory_bytes = total_samples * self.feature_dim * 4  # float32 = 4 bytes
        return memory_bytes / (1024 * 1024)  # MB
    
    def _infer_dataset_name(self, path: str) -> str:
        """パスからデータセット名を推定"""
        if 'IQON3000' in path or 'iqon' in path.lower():
            return 'IQON3000'
        elif 'DeepFurniture' in path or 'furniture' in path.lower():
            return 'DeepFurniture'
        else:
            return 'Unknown'
    
    def _setup_dataset_config(self):
        """データセット固有の設定"""
        if self.dataset_name == 'IQON3000':
            self.expected_categories = 7
            self.category_range = (1, 7)
        elif self.dataset_name == 'DeepFurniture':
            self.expected_categories = 11
            self.category_range = (1, 11)
        else:
            unique_cats = np.unique(np.concatenate([
                self.query_categories.flatten(),
                self.positive_categories.flatten()
            ]))
            unique_cats = unique_cats[unique_cats > 0]
            self.expected_categories = len(unique_cats)
            self.category_range = (1, max(unique_cats)) if len(unique_cats) > 0 else (1, 10)
        
        print(f"Dataset: {self.dataset_name}, Expected categories: {self.expected_categories}, Range: {self.category_range}")
    
    def _load_data(self):
        """データ読み込み"""
        with open(self.split_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, tuple) and len(data) >= 6:
            self.query_features = np.array(data[0], dtype=np.float32)
            self.positive_features = np.array(data[1], dtype=np.float32)
            
            # シーンIDの安全な処理
            if len(data) > 2:
                scene_ids_raw = data[2]
                if isinstance(scene_ids_raw, (list, np.ndarray)):
                    scene_ids_list = []
                    for i, sid in enumerate(scene_ids_raw):
                        try:
                            if isinstance(sid, (str, bytes, np.str_, np.bytes_)):
                                scene_ids_list.append(abs(hash(str(sid))) % 1000000)
                            else:
                                scene_ids_list.append(int(sid))
                        except (ValueError, TypeError):
                            scene_ids_list.append(i)
                    self.scene_ids = np.array(scene_ids_list, dtype=np.int32)
                else:
                    self.scene_ids = np.arange(len(self.query_features), dtype=np.int32)
            else:
                self.scene_ids = np.arange(len(self.query_features), dtype=np.int32)
            
            self.query_categories = np.array(data[3], dtype=np.int32)
            self.positive_categories = np.array(data[4], dtype=np.int32)
            
            if len(data) > 5:
                set_sizes_raw = data[5]
                try:
                    self.set_sizes = np.array(set_sizes_raw, dtype=np.int32)
                except (ValueError, TypeError):
                    self.set_sizes = np.sum(self.query_categories > 0, axis=1)
            else:
                self.set_sizes = np.sum(self.query_categories > 0, axis=1)
            
            if len(data) > 6:
                self.query_ids = self._safe_convert_ids(data[6])
                self.positive_ids = self._safe_convert_ids(data[7] if len(data) > 7 else data[6])
            else:
                num_scenes, num_items = self.query_features.shape[:2]
                self.query_ids = np.arange(num_scenes * num_items).reshape(num_scenes, num_items)
                self.positive_ids = self.query_ids
        else:
            raise ValueError(f"Unsupported data format in {self.split_path}")
        
        self.max_item_num = self.query_features.shape[1]
        self.feature_dim = self.query_features.shape[2]
        
        print(f"Loaded data: {len(self.scene_ids)} scenes, {self.max_item_num} max items, {self.feature_dim}D features")
    
    def _safe_convert_ids(self, ids_data):
        """IDデータを安全に数値配列に変換"""
        if isinstance(ids_data, np.ndarray) and ids_data.dtype.kind in ['i', 'u', 'f']:
            return ids_data.astype(np.int32)
        
        ids_array = np.array(ids_data)
        if ids_array.size == 0:
            return np.array([], dtype=np.int32).reshape(0, self.max_item_num)
        
        original_shape = ids_array.shape
        flat_ids = ids_array.flatten()
        converted_ids = []
        
        for i, item_id in enumerate(flat_ids):
            try:
                if isinstance(item_id, (str, bytes, np.str_, np.bytes_)):
                    item_str = str(item_id)
                    if item_str == '' or item_str == '0':
                        converted_ids.append(0)
                    else:
                        converted_ids.append(abs(hash(item_str)) % 1000000)
                else:
                    converted_ids.append(int(item_id))
            except (ValueError, TypeError):
                converted_ids.append(i % 1000000)
        
        return np.array(converted_ids, dtype=np.int32).reshape(original_shape)
    
    def _build_negative_pool(self):
        """ネガティブサンプルプール構築（CLNeg有効時のみ）"""
        self.negative_pool = defaultdict(list)
        
        # 全特徴量とカテゴリを結合
        all_features = np.concatenate([
            self.query_features.reshape(-1, self.feature_dim),
            self.positive_features.reshape(-1, self.feature_dim)
        ], axis=0)
        
        all_categories = np.concatenate([
            self.query_categories.flatten(),
            self.positive_categories.flatten()
        ], axis=0)
        
        # カテゴリ別にプール構築
        for feat, cat in zip(all_features, all_categories):
            if cat > 0:  # パディング以外
                self.negative_pool[int(cat)].append(feat)
        
        # NumPy配列に変換
        for cat in self.negative_pool:
            self.negative_pool[cat] = np.array(self.negative_pool[cat])
    
    def _generate_negatives(self, query_batch, positive_batch, query_cat_batch, positive_cat_batch):
        """条件付きネガティブサンプル生成"""
        batch_size, max_items, feature_dim = query_batch.shape
        total_batch_size = batch_size * 2
        
        if not self.use_negatives:
            # CLNegが無効な場合は最小限のダミー配列を返す
            return np.zeros((total_batch_size, max_items, 1, feature_dim), dtype=np.float32)
        
        # CLNegが有効な場合は本格的なネガティブサンプル生成
        negative_samples = np.zeros(
            (total_batch_size, max_items, self.neg_num, feature_dim), 
            dtype=np.float32
        )
        
        for batch_idx in range(batch_size):
            # クエリのネガティブ
            for item_idx in range(max_items):
                cat = query_cat_batch[batch_idx, item_idx]
                if cat > 0 and cat in self.negative_pool:
                    pool = self.negative_pool[cat]
                    if len(pool) >= self.neg_num:
                        indices = self.rng.choice(len(pool), self.neg_num, replace=False)
                        negative_samples[batch_idx, item_idx] = pool[indices]
                    elif len(pool) > 0:
                        indices = self.rng.choice(len(pool), self.neg_num, replace=True)
                        negative_samples[batch_idx, item_idx] = pool[indices]
            
            # ポジティブのネガティブ
            for item_idx in range(max_items):
                cat = positive_cat_batch[batch_idx, item_idx]
                if cat > 0 and cat in self.negative_pool:
                    pool = self.negative_pool[cat]
                    if len(pool) >= self.neg_num:
                        indices = self.rng.choice(len(pool), self.neg_num, replace=False)
                        negative_samples[batch_size + batch_idx, item_idx] = pool[indices]
                    elif len(pool) > 0:
                        indices = self.rng.choice(len(pool), self.neg_num, replace=True)
                        negative_samples[batch_size + batch_idx, item_idx] = pool[indices]
        
        return negative_samples
    
    def __len__(self):
        return int(np.ceil(len(self.scene_ids) / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indexes)
    
    def __getitem__(self, idx):
        """条件付きバッチ取得"""
        batch_indices = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        current_batch_size = len(batch_indices)
        
        if current_batch_size == 0:
            return None
        
        # バッチデータ取得
        query_batch = self.query_features[batch_indices]
        positive_batch = self.positive_features[batch_indices]
        query_cat_batch = self.query_categories[batch_indices]
        positive_cat_batch = self.positive_categories[batch_indices]
        set_size_batch = self.set_sizes[batch_indices]
        scene_id_batch = self.scene_ids[batch_indices]
        
        # ID取得
        query_id_batch = self._safe_get_batch_ids(self.query_ids, batch_indices)
        positive_id_batch = self._safe_get_batch_ids(self.positive_ids, batch_indices)
        
        # 結合特徴量作成
        combined_features = np.concatenate([query_batch, positive_batch], axis=0)
        dummy_labels = np.zeros_like(combined_features)
        combined_set_sizes = np.concatenate([set_size_batch, set_size_batch])
        combined_scene_ids = np.concatenate([scene_id_batch, scene_id_batch])
        combined_query_ids = np.concatenate([query_id_batch, positive_id_batch])
        combined_positive_ids = np.concatenate([positive_id_batch, query_id_batch])
        
        # 条件付きネガティブサンプル生成
        negative_samples = self._generate_negatives(
            query_batch, positive_batch, query_cat_batch, positive_cat_batch
        )
        
        # TensorFlow形式で返却
        inputs = (
            tf.constant(combined_features, dtype=tf.float32),
            tf.constant(dummy_labels, dtype=tf.float32),
            tf.constant(combined_set_sizes, dtype=tf.float32),
            tf.constant(query_cat_batch, dtype=tf.int32),
            tf.constant(positive_cat_batch, dtype=tf.int32),
            tf.constant(combined_query_ids, dtype=tf.int32),
            tf.constant(combined_positive_ids, dtype=tf.int32),
            tf.constant(negative_samples, dtype=tf.float32)
        )
        
        return inputs, combined_scene_ids
    
    def _safe_get_batch_ids(self, ids_data, batch_indices):
        """バッチIDを安全に取得"""
        try:
            if isinstance(ids_data, np.ndarray):
                return ids_data[batch_indices]
            elif isinstance(ids_data, (list, tuple)):
                return np.array([ids_data[i] for i in batch_indices])
            else:
                return np.array(batch_indices).reshape(-1, 1)
        except (IndexError, ValueError):
            batch_size = len(batch_indices)
            if hasattr(self, 'max_item_num'):
                return np.arange(batch_size * self.max_item_num).reshape(batch_size, self.max_item_num)
            else:
                return np.array(batch_indices).reshape(-1, 1)
    
    def get_data_info(self):
        """データ情報取得"""
        return {
            'num_scenes': len(self.scene_ids),
            'max_item_num': self.max_item_num,
            'feature_dim': self.feature_dim,
            'batch_size': self.batch_size,
            'num_batches': len(self),
            'has_cluster_centers': self.cluster_centers is not None,
            'cluster_centers_shape': self.cluster_centers.shape if self.cluster_centers is not None else None,
            'use_negatives': self.use_negatives,
            'negative_pool_size': len(self.negative_pool) if self.use_negatives else 0,
            'estimated_memory_mb': self._estimate_negative_pool_memory() if self.use_negatives else 0
        }