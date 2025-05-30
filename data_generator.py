import os
import gzip
import pickle
import numpy as np
import tensorflow as tf
import time
import logging
from collections import defaultdict
from tensorflow.keras.utils import Sequence

def load_category_centers(center_path: str) -> (dict, int):
    """
    事前計算されたカテゴリ中心を読み込み
    
    Args:
        center_path: カテゴリ中心ファイルのパス
        
    Returns:
        (centers, count): カテゴリ中心辞書とカテゴリ数
    """
    if not os.path.exists(center_path):
        logging.warning(f"Category centers not found at {center_path}")
        return {}, 0
    
    try:
        if center_path.endswith('.gz'):
            with gzip.open(center_path, 'rb') as f:
                centers = pickle.load(f)
        else:
            with open(center_path, 'rb') as f:
                centers = pickle.load(f)
        
        count = len(centers) if isinstance(centers, dict) else centers.shape[0]
        logging.info(f"Loaded category centers from {center_path}: {count} categories")
        return centers, count
        
    except Exception as e:
        logging.error(f"Failed to load category centers from {center_path}: {e}")
        return {}, 0

class DataGenerator(Sequence):
    """
    統一データジェネレータ - DeepFurniture, IQON3000対応
    CPU使用率を抑制し、複数のデータ形式に対応
    """
    
    def __init__(
        self,
        split_path: str,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,    
        center_path: str = None,
        neg_pool_path: str = None,
        neg_num: int = 30,
        neg_threshold: float = 0.6,
        noise_matrix_path: str = None
    ):
        """
        Args:
            split_path: データファイルパス (.pkl)
            batch_size: バッチサイズ
            shuffle: データをシャッフルするか
            seed: ランダムシード
            center_path: カテゴリ中心ファイルパス
            neg_pool_path: ネガティブプールファイルパス
            neg_num: ネガティブサンプル数
            neg_threshold: ハードネガティブ閾値
            noise_matrix_path: ノイズ行列パス（未使用）
        """
        self.split_path = split_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.neg_num = neg_num
        self.neg_threshold = neg_threshold
        
        # パスの設定
        base_dir = os.path.dirname(split_path)
        if center_path is None:
            center_path = os.path.join(base_dir, 'category_centers.pkl.gz')
        if neg_pool_path is None:
            neg_pool_path = os.path.join(base_dir, 'neg_pool.pkl')
        
        self.center_path = center_path
        self.neg_pool_path = neg_pool_path
        
        # データ読み込み
        self._load_dataset()
        
        # カテゴリ中心読み込み
        self._load_cluster_centers()
        
        # ネガティブプール構築/読み込み
        self._setup_negative_pool()
        
        # インデックス初期化
        self.indexes = np.arange(len(self.scene_ids))
        if self.shuffle:
            self.rng.shuffle(self.indexes)
        
        logging.info(f"DataGenerator initialized: {len(self)} batches, "
                    f"max_item_num={self.max_item_num}, feature_dim={self.feature_dim}")

    def _load_dataset(self):
        """データセット読み込み - 複数形式対応"""
        logging.info(f"Loading dataset from {self.split_path}...")
        load_start = time.time()
        
        try:
            with open(self.split_path, 'rb') as f:
                data = pickle.load(f)
                
                # データ形式の自動検出
                if isinstance(data, tuple):
                    self._load_tuple_format(data)
                elif isinstance(data, list):
                    self._load_list_format(data)
                else:
                    raise ValueError(f"Unsupported data format: {type(data)}")
                    
        except Exception as e:
            raise ValueError(f"Failed to load dataset ({self.split_path}): {e}")
        
        # データ型最適化
        self._optimize_data_types()
        
        # 基本統計
        self._compute_basic_stats()
        
        load_time = time.time() - load_start
        logging.info(f"Dataset loaded in {load_time:.2f}s")

    def _load_tuple_format(self, data):
        """タプル形式データの読み込み (DeepFurniture形式)"""
        if len(data) >= 8:  # DeepFurniture完全形式
            (self.query_features, self.positive_features, 
             self.scene_ids, self.query_categories, 
             self.positive_categories, self.set_sizes, 
             self.query_ids, self.positive_ids) = data[:8]
        elif len(data) >= 6:  # 基本形式
            (self.query_features, self.positive_features, 
             self.query_categories, self.positive_categories, 
             self.query_ids, self.positive_ids) = data[:6]
            # 不足データを生成
            self.scene_ids = np.arange(len(self.query_features))
            self.set_sizes = np.array([np.sum(cat > 0) for cat in self.query_categories])
        else:
            raise ValueError(f"Tuple format requires at least 6 elements ({len(data)} found)")

    def _load_list_format(self, data):
        """リスト形式データの読み込み (IQON3000形式)"""
        if len(data) < 6:
            raise ValueError(f"List format requires at least 6 elements ({len(data)} found)")
        
        # numpy配列に変換
        features_data = [np.array(d) for d in data[:6]]
        (self.query_features, self.positive_features, 
         self.query_categories, self.positive_categories, 
         self.query_ids, self.positive_ids) = features_data
        
        # 補助データ生成
        self.scene_ids = np.arange(len(self.query_features))
        
        # set_sizesの計算
        if isinstance(self.query_categories[0], (list, np.ndarray)):
            self.set_sizes = np.array([np.sum(np.array(cat) > 0) for cat in self.query_categories])
        else:
            self.set_sizes = np.array([len(self.query_categories)] * len(self.query_features))

    def _optimize_data_types(self):
        """データ型最適化 - メモリ効率化"""
        # numpy配列に変換
        array_attrs = ['query_features', 'positive_features', 'query_categories', 
                       'positive_categories', 'query_ids', 'positive_ids', 
                       'scene_ids', 'set_sizes']
        
        for attr in array_attrs:
            if hasattr(self, attr) and not isinstance(getattr(self, attr), np.ndarray):
                setattr(self, attr, np.array(getattr(self, attr)))

        # float32に統一（メモリ効率化）
        if self.query_features.dtype != np.float32:
            self.query_features = self.query_features.astype(np.float32)
        if self.positive_features.dtype != np.float32:
            self.positive_features = self.positive_features.astype(np.float32)

    def _compute_basic_stats(self):
        """基本統計情報の計算"""
        self.max_item_num = self.query_features.shape[1] if len(self.query_features.shape) > 1 else 10
        self.feature_dim = self.query_features.shape[-1] if len(self.query_features.shape) > 1 else 512
        
        # カテゴリ統計
        unique_categories = set()
        for cats in [self.query_categories, self.positive_categories]:
            if isinstance(cats, np.ndarray):
                unique_categories.update(cats.flatten())
            else:
                for cat_array in cats:
                    unique_categories.update(np.array(cat_array).flatten())
        
        self.num_categories = len([c for c in unique_categories if c > 0])
        
        logging.info(f"Dataset stats: {len(self.scene_ids)} scenes, "
                    f"max_items={self.max_item_num}, feature_dim={self.feature_dim}, "
                    f"categories={self.num_categories}")

    def _load_cluster_centers(self):
        """カテゴリ中心の読み込み - エラーハンドリング強化"""
        try:
            centers_input, num_categories_loaded = load_category_centers(self.center_path)

            if isinstance(centers_input, dict):
                if centers_input:  # 辞書が空でない
                    # 最大カテゴリIDを取得して配列サイズを決定
                    max_cat_id = max(centers_input.keys()) if centers_input else 0
                    expected_num_cats = max(max_cat_id, self.num_categories, 11)  # 最低11カテゴリ
                    
                    # 特徴量次元を取得
                    first_vec = next(iter(centers_input.values()))
                    if hasattr(first_vec, 'shape') and len(first_vec.shape) > 0:
                        embedding_dim = first_vec.shape[0]
                    else:
                        embedding_dim = self.feature_dim
                    
                    # 配列を作成
                    center_array = np.zeros((expected_num_cats, embedding_dim), dtype=np.float32)
                    valid_cats = 0
                    
                    for cat_id, center_vec in centers_input.items():
                        if 1 <= cat_id <= expected_num_cats:
                            center_array[cat_id - 1] = center_vec
                            valid_cats += 1
                    
                    if valid_cats > 0:
                        self.cluster_centers = center_array
                        logging.info(f"Converted category centers from dict to array: {self.cluster_centers.shape}")
                    else:
                        logging.warning("No valid categories in centers dict, using dummy data")
                        self.cluster_centers = np.zeros((1, embedding_dim), dtype=np.float32)
                else:
                    logging.warning("Category centers dict is empty, using dummy data")
                    self.cluster_centers = np.zeros((1, self.feature_dim), dtype=np.float32)

            elif isinstance(centers_input, np.ndarray):
                if centers_input.size > 0 and centers_input.ndim == 2:
                    self.cluster_centers = centers_input.astype(np.float32)
                    logging.info(f"Loaded category centers array: {self.cluster_centers.shape}")
                else:
                    logging.warning(f"Invalid centers array shape: {centers_input.shape}, using dummy data")
                    self.cluster_centers = np.zeros((1, self.feature_dim), dtype=np.float32)
            
            else:
                logging.warning(f"Unexpected centers format: {type(centers_input)}, using dummy data")
                self.cluster_centers = np.zeros((1, self.feature_dim), dtype=np.float32)

        except Exception as e:
            logging.warning(f"Error loading category centers: {e}, using dummy data")
            self.cluster_centers = np.zeros((1, self.feature_dim), dtype=np.float32)

        # 2D配列であることを保証
        if self.cluster_centers.ndim == 1:
            self.cluster_centers = np.expand_dims(self.cluster_centers, axis=0)

    def _setup_negative_pool(self):
        """ネガティブプールの構築/読み込み"""
        try:
            if 'train' in self.split_path or not os.path.exists(self.neg_pool_path):
                logging.info("Building negative pool...")
                self._build_negative_pool()
            else:
                logging.info(f"Loading existing negative pool from {self.neg_pool_path}")
                with open(self.neg_pool_path, 'rb') as f:
                    self.cat_to_negatives = pickle.load(f)
        except Exception as e:
            logging.warning(f"Negative pool error: {e}, using empty pool")
            self.cat_to_negatives = {}

    def _build_negative_pool(self):
        """軽量版ネガティブプール構築"""
        logging.info("Building lightweight negative pool...")
        
        # メモリ効率のため、カテゴリあたりの最大サンプル数を制限
        max_samples_per_cat = 1000
        
        # 全特徴量とカテゴリを結合
        all_feats = np.concatenate([self.query_features, self.positive_features], axis=0)
        all_cats = np.concatenate([self.query_categories, self.positive_categories], axis=0)
        all_ids = np.concatenate([self.query_ids, self.positive_ids], axis=0)

        buffer = defaultdict(list)
        
        # フラット化して処理
        flat_feats = all_feats.reshape(-1, all_feats.shape[-1])
        flat_cats = all_cats.flatten()
        flat_ids = all_ids.flatten()
        
        for feat_vec, cat_val, id_val in zip(flat_feats, flat_cats, flat_ids):
            if cat_val > 0:
                cat_int = int(cat_val)
                if len(buffer[cat_int]) < max_samples_per_cat:
                    buffer[cat_int].append((feat_vec, id_val))

        # カテゴリ別に整理
        self.cat_to_negatives = {}
        for cat, items in buffer.items():
            if len(items) > 0:
                self.cat_to_negatives[cat] = {
                    'vectors': np.stack([fv for fv, _ in items]),
                    'ids': np.array([iid for _, iid in items])
                }
        
        # 保存
        try:
            with open(self.neg_pool_path, 'wb') as f:
                pickle.dump(self.cat_to_negatives, f)
            logging.info(f"Negative pool saved to {self.neg_pool_path}")
        except Exception as e:
            logging.warning(f"Could not save negative pool: {e}")

    def __len__(self) -> int:
        """バッチ数を返す"""
        return int(np.ceil(len(self.scene_ids) / self.batch_size))

    def on_epoch_end(self):
        """エポック終了時の処理"""
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def __getitem__(self, idx: int):
        """バッチデータを取得"""
        batch_indices = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        current_batch_size = len(batch_indices)
        
        if current_batch_size == 0:
            return None

        # バッチデータ収集
        query_feat_batch = self.query_features[batch_indices]
        positive_feat_batch = self.positive_features[batch_indices]
        query_cat_batch = self.query_categories[batch_indices]
        positive_cat_batch = self.positive_categories[batch_indices]
        query_id_batch = self.query_ids[batch_indices]
        positive_id_batch = self.positive_ids[batch_indices]
        set_size_batch = self.set_sizes[batch_indices]
        scene_id_batch = self.scene_ids[batch_indices]

        # クエリ+ポジティブを結合
        combined_features = np.concatenate([query_feat_batch, positive_feat_batch], axis=0)
        dummy_labels = np.zeros_like(combined_features)
        original_set_sizes = np.concatenate([set_size_batch, set_size_batch], axis=0)
        scene_ids_concat = np.concatenate([scene_id_batch, scene_id_batch], axis=0)

        query_ids_concat = np.concatenate([query_id_batch, positive_id_batch], axis=0)
        positive_ids_concat = np.concatenate([positive_id_batch, query_id_batch], axis=0)

        # ネガティブサンプリング
        negative_samples = self._create_negative_samples(
            combined_features, current_batch_size, 
            query_cat_batch, positive_cat_batch,
            query_ids_concat, positive_ids_concat
        )

        # TensorFlowテンソルに変換
        inputs = (
            tf.constant(combined_features, dtype=tf.float32),
            tf.constant(dummy_labels, dtype=tf.float32),
            tf.constant(original_set_sizes, dtype=tf.float32),
            tf.constant(query_cat_batch, dtype=tf.int32),
            tf.constant(positive_cat_batch, dtype=tf.int32),
            tf.constant(query_ids_concat, dtype=tf.int32),
            tf.constant(positive_ids_concat, dtype=tf.int32),
            tf.constant(negative_samples, dtype=tf.float32)
        )

        return inputs, scene_ids_concat

    def _create_negative_samples(self, combined_features, current_batch_size, 
                                query_cat_batch, positive_cat_batch,
                                query_ids_concat, positive_ids_concat):
        """ネガティブサンプル生成"""
        _, num_items, feature_dim = combined_features.shape
        negative_samples = np.zeros(
            (2 * current_batch_size, num_items, self.neg_num, feature_dim),
            dtype=np.float32
        )

        # アイテムごとにハードネガティブをサンプリング
        for combined_idx in range(2 * current_batch_size):
            is_query = combined_idx < current_batch_size
            set_idx = combined_idx if is_query else combined_idx - current_batch_size
            
            categories = query_cat_batch[set_idx] if is_query else positive_cat_batch[set_idx]
            item_ids = query_ids_concat[combined_idx] if is_query else positive_ids_concat[combined_idx]

            for item_idx, cat in enumerate(categories):
                if cat <= 0:
                    continue
                    
                feature_vector = combined_features[combined_idx, item_idx]
                exclude_id = int(item_ids[item_idx])
                
                negative_samples[combined_idx, item_idx] = (
                    self._sample_hard_negatives(feature_vector, int(cat), exclude_id)
                )

        return negative_samples

    def _sample_hard_negatives(self, query_vec, category: int, exclude_id: int) -> np.ndarray:
        """ハードネガティブサンプリング - 軽量版"""
        pool = self.cat_to_negatives.get(category, {
            'vectors': np.zeros((0, query_vec.shape[-1])), 
            'ids': np.array([])
        })
        
        vectors, ids = pool['vectors'], pool['ids']
        
        if vectors.size == 0:
            # プールが空の場合はゼロベクトルを返す
            return np.zeros((self.neg_num, query_vec.shape[-1]), dtype=np.float32)

        try:
            # 簡略化した類似度計算
            norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-7
            normalized = vectors / norms
            q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-7)
            similarities = normalized @ q_norm

            # 除外IDをマスク
            if len(ids) == len(similarities):
                exclude_mask = (ids == exclude_id)
                similarities[exclude_mask] = -1.0

            # ハードネガティブ候補を選択
            candidate_idxs = np.where(similarities >= self.neg_threshold)[0]
            if len(candidate_idxs) < self.neg_num:
                # 候補が少ない場合は類似度上位を選択
                candidate_idxs = np.argsort(similarities)[::-1][:self.neg_num]
            
            # サンプリング
            chosen = self.rng.choice(
                candidate_idxs, 
                size=self.neg_num, 
                replace=len(candidate_idxs) < self.neg_num
            )
            return vectors[chosen]
            
        except Exception as e:
            # エラー時はランダムサンプリング
            if len(vectors) >= self.neg_num:
                chosen = self.rng.choice(len(vectors), size=self.neg_num, replace=False)
            else:
                chosen = self.rng.choice(len(vectors), size=self.neg_num, replace=True)
            return vectors[chosen]

    def get_data_info(self):
        """データセット情報を返す"""
        return {
            'num_scenes': len(self.scene_ids),
            'max_item_num': self.max_item_num,
            'feature_dim': self.feature_dim,
            'num_categories': self.num_categories,
            'batch_size': self.batch_size,
            'num_batches': len(self),
            'cluster_centers_shape': self.cluster_centers.shape,
            'negative_pool_categories': len(self.cat_to_negatives)
        }