import os
import gzip
import pickle
import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow.keras.utils import Sequence

# ============================================================================
# データジェネレータ本体
# ============================================================================

class DataGenerator(Sequence):
    """
    セット検索タスク用の高機能データジェネレータ。

    主な機能:
    - 完全なアイテム集合を動的にクエリとターゲットに分割。
    - 集合の最小アイテム数を保証し、データの一貫性を担保。
    - TPaNeg用の候補ネガティブを事前計算キャッシュから高速に読み込み。
    - ホワイトニング変換をサポート。
    """
    def __init__(self,
                 split_path: str,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 seed: int = 42,
                 # --- TPaNeg / Hard Negative Mining ---
                 use_negatives: bool = False,
                 negative_cache_path: str = None,
                 candidate_neg_num: int = 50,
                 # --- Set Integrity & Splitting ---
                 random_split: bool = True,
                 min_query_items: int = 2,
                 min_target_items: int = 2,
                 # --- Optional Transformations ---
                 whitening_params: dict = None,
                 include_set_ids: bool = False):
        """
        Args:
            split_path (str): データ分割情報を含む .pkl ファイルのパス。
            batch_size (int): バッチサイズ。
            shuffle (bool): エポックごとにデータをシャッフルするかどうか。
            seed (int): 乱数シード。
            use_negatives (bool): TPaNeg用の候補ネガティブを使用するかどうか。
            negative_cache_path (str): 事前計算された候補ネガティブのキャッシュファイルパス。
            candidate_neg_num (int): 1つの正解アイテムに対して使用する候補ネガティブの最大数。
            random_split (bool): 集合をランダムな比率で分割するかどうか。
            min_query_items (int): 分割後のクエリ集合が持つべき最小アイテム数。
            min_target_items (int): 分割後のターゲット集合が持つべき最小アイテム数。
            whitening_params (dict): ホワイトニング変換用のパラメータ（mean, matrix）。
            include_set_ids (bool): テスト時にSetIDをバッチに含めるか。
        """
        # 基本パラメータ
        self.split_path = split_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # 集合分割に関する設定
        self.random_split = random_split
        self.min_query_items = max(2, min_query_items)
        self.min_target_items = max(2, min_target_items)
        print(f"🔒 Set Integrity Guarantee: Query >= {self.min_query_items}, Target >= {self.min_target_items}")

        # データ読み込みとフィルタリング
        self._load_and_prepare_data()

        # TPaNeg (候補ネガティブ) の設定
        self.use_negatives = use_negatives
        self.candidate_neg_num = candidate_neg_num
        self.negative_cache_ids = None # IDsを保持する
        self.item_feature_map = None # IDから特徴量を取得するためのマップ
        if self.use_negatives:
            self._load_negative_cache(negative_cache_path)

        # ホワイトニング設定
        self.whitening_params = whitening_params
        if self.whitening_params:
            self.whitening_mean = self.whitening_params['mean']
            self.whitening_matrix = self.whitening_params['matrix']
            print(f"✅ Whitening enabled for DataGenerator ({os.path.basename(split_path)}).")
            
        # テスト用設定
        self.include_SetIDs = include_set_ids

        # バッチ生成用のインデックスを初期化
        self.indexes = np.arange(len(self.scene_ids))
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def _load_and_prepare_data(self):
        """データの読み込み、統合、フィルタリングを行う"""
        print(f"Loading data from {self.split_path}...")
        if self.split_path.endswith('.gz'):
            opener = gzip.open
        else:
            opener = open
        with opener(self.split_path, 'rb') as f:
            data = pickle.load(f)

        # データ形式のチェック
        if not (isinstance(data, tuple) and len(data) >= 8):
            raise ValueError(f"Unsupported data format in {self.split_path}.")

        # pklから各要素を抽出
        q_feats, t_feats, scene_ids, q_cats, t_cats, _, q_ids, t_ids = data[:8]

        # データをシーン（完全な集合）ごとに統合
        raw_full_sets = []
        for i in range(len(q_feats)):
            q_valid_mask = q_cats[i] > 0
            t_valid_mask = t_cats[i] > 0

            # 結合して完全な集合を作成
            full_f = np.concatenate((q_feats[i][q_valid_mask], t_feats[i][t_valid_mask]), axis=0)
            full_c = np.concatenate((q_cats[i][q_valid_mask], t_cats[i][t_valid_mask]), axis=0)
            full_i = np.concatenate((q_ids[i][q_valid_mask], t_ids[i][t_valid_mask]), axis=0)

            raw_full_sets.append({
                'features': full_f.astype(np.float32),
                'categories': full_c.astype(np.int32),
                'ids': full_i.astype(np.int32)
            })
        
        # シーンIDも保持
        self.scene_ids = list(scene_ids)
        print(f"Loaded {len(raw_full_sets)} raw scenes.")

        # =======================================================
        # ★★★ 統合されたフィルタリングロジック ★★★
        # =======================================================
        min_required_items = self.min_query_items + self.min_target_items
        # ★ カテゴリ種類数の最小値 (要件通り4に設定)
        min_required_categories = 4
        
        original_count = len(raw_full_sets)
        
        self.full_sets = []
        valid_scene_ids = []
        
        filtered_items_count = 0
        
        for i, s in enumerate(raw_full_sets):
            
            # カテゴリが0（パディング）以外のアイテムのみを考慮
            valid_categories = s['categories'][s['categories'] > 0]
            
            num_unique_categories = len(np.unique(valid_categories))
            num_total_items = len(s['features'])
            
            # アイテム総数とユニークカテゴリ数の両方をチェック
            if (num_total_items >= min_required_items and 
                num_unique_categories >= min_required_categories):
                
                self.full_sets.append(s)
                valid_scene_ids.append(self.scene_ids[i])
            else:
                filtered_items_count += 1
        
        self.scene_ids = valid_scene_ids

        if filtered_items_count > 0:
            print(f"🚫 Filtered out {filtered_items_count} scenes. Reasons:")
            print(f"   - Minimum total items required: {min_required_items}")
            print(f"   - Minimum unique categories required: {min_required_categories}")
            
        print(f"✅ Working with {len(self.full_sets)} valid scenes.")

        # パディング用の次元情報を保存
        set_sizes = [len(s['features']) for s in self.full_sets]
        self.max_item_num = max(set_sizes) if set_sizes else 0
        self.feature_dim = self.full_sets[0]['features'].shape[1] if self.full_sets else 0
        print(f"Data stats: max_items={self.max_item_num}, feature_dim={self.feature_dim}")

    def _load_negative_cache(self, cache_path):
        """事前計算された候補ネガティブのキャッシュを読み込む"""
        if not cache_path or not os.path.exists(cache_path):
            print(f"⚠️ Negative cache not found at '{cache_path}'. TPaNeg is disabled.")
            self.use_negatives = False
            return
        
        try:
            print(f"Loading pre-computed negative cache from {cache_path}...")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # キャッシュデータの基本形式チェック
            if not isinstance(cache_data, dict):
                raise ValueError(f"Cache data is not a dictionary. Type: {type(cache_data)}")
            
            print(f"Available keys in cache: {list(cache_data.keys())}")
            
            # item_feature_mapの取得
            if 'item_feature_map' in cache_data:
                self.item_feature_map = cache_data['item_feature_map']
            else:
                raise ValueError("Missing 'item_feature_map' in cache")
            
            # hard_negatives_cacheの取得（複数の形式に対応）
            if 'hard_negatives_cache' in cache_data:
                # 新しい形式
                self.negative_cache_ids = cache_data['hard_negatives_cache']
                print("✅ Using standard cache format")
            elif 'negative_cache' in cache_data:
                # 古い形式（互換性対応）
                self.negative_cache_ids = cache_data['negative_cache']
                print("⚠️ Using legacy cache format")
            else:
                # その他のキー名の可能性を確認
                possible_keys = [k for k in cache_data.keys() if 'negative' in k.lower()]
                if possible_keys:
                    # 最初に見つかったnegativeを含むキーを使用
                    fallback_key = possible_keys[0]
                    self.negative_cache_ids = cache_data[fallback_key]
                    print(f"⚠️ Using fallback cache key: '{fallback_key}'")
                else:
                    raise ValueError(f"No valid negative cache key found. Available keys: {list(cache_data.keys())}")
            
            # データの型チェック
            if not isinstance(self.item_feature_map, dict):
                raise ValueError(f"item_feature_map is not a dictionary. Type: {type(self.item_feature_map)}")
            
            if not isinstance(self.negative_cache_ids, dict):
                raise ValueError(f"negative_cache_ids is not a dictionary. Type: {type(self.negative_cache_ids)}")
            
            # キーの型チェック（文字列であることを確認）
            if self.item_feature_map and not all(isinstance(k, str) for k in list(self.item_feature_map.keys())[:10]):
                print("⚠️ Converting item_feature_map keys to strings...")
                self.item_feature_map = {str(k): v for k, v in self.item_feature_map.items()}
            
            if self.negative_cache_ids and not all(isinstance(k, str) for k in list(self.negative_cache_ids.keys())[:10]):
                print("⚠️ Converting negative_cache_ids keys to strings...")
                self.negative_cache_ids = {str(k): v for k, v in self.negative_cache_ids.items()}
            
            print(f"✅ Negative cache loaded successfully.")
            print(f"   - Item feature map: {len(self.item_feature_map)} items")
            print(f"   - Negative cache: {len(self.negative_cache_ids)} cached relationships")
            
            # サンプルデータの確認（デバッグ用）
            if self.negative_cache_ids:
                sample_key = next(iter(self.negative_cache_ids.keys()))
                sample_value = self.negative_cache_ids[sample_key]
                print(f"   - Sample cache entry: {sample_key} -> {len(sample_value) if isinstance(sample_value, list) else type(sample_value)} negatives")
            
        except Exception as e:
            print(f"❌ Failed to load negative cache: {e}")
            print(f"   Cache file: {cache_path}")
            if 'cache_data' in locals():
                print(f"   Available keys: {list(cache_data.keys()) if isinstance(cache_data, dict) else 'Not a dict'}")
            print("   TPaNeg is disabled.")
            self.use_negatives = False
            self.negative_cache_ids = {}
            self.item_feature_map = {}
            # エラーを再スローしない（トレーニングを続行）
            return

    def _random_split_set(self, full_set):
        """集合を指定された最小数を保証しつつ、クエリとターゲットの差を1以下に抑えて分割する"""
        features, categories, ids = full_set['features'], full_set['categories'], full_set['ids']
        total_items = len(features)
        
        indices = np.arange(total_items)
        if self.random_split:
            self.rng.shuffle(indices)
        # random_split=Falseの場合は、indicesの順序はそのまま（決定論的分割）

        # ★★★ アイテム数の差を1以下に抑える制約（random_splitに関係なく適用）★★★
        # クエリとターゲットのアイテム数の差を1以下に抑えるようにquery_countを決定
        base_query_count = total_items // 2
        
        # 可能なquery_countの候補（差が1以下になる候補）
        possible_query_counts = []
        
        # +-1 の範囲で試す
        for qc_candidate in [base_query_count - 1, base_query_count, base_query_count + 1]:
            target_count = total_items - qc_candidate
            # 最小アイテム数制約を満たし、かつアイテム数の差が1以下か確認
            if (self.min_query_items <= qc_candidate <= total_items - self.min_target_items) and \
            (self.min_target_items <= target_count <= total_items - self.min_query_items) and \
            (abs(qc_candidate - target_count) <= 1):  # 差が1以下の制約
                possible_query_counts.append(qc_candidate)
        
        if not possible_query_counts:
            # フォールバック: 最小制約を満たす範囲で、最も均等に近い分割を選択
            min_q_fallback = self.min_query_items
            max_q_fallback = total_items - self.min_target_items
            if min_q_fallback > max_q_fallback:
                raise ValueError(f"Cannot split set of size {total_items} with min_query={self.min_query_items}, min_target={self.min_target_items}")
            
            # 最も均等に近い分割を選択（total_items // 2 に最も近いもの）
            best_query_count = min_q_fallback
            best_diff = abs((min_q_fallback) - (total_items - min_q_fallback))
            
            for qc in range(min_q_fallback, max_q_fallback + 1):
                tc = total_items - qc
                diff = abs(qc - tc)
                if diff < best_diff:
                    best_diff = diff
                    best_query_count = qc
            
            query_count = best_query_count
        else:
            if self.random_split:
                # ランダム分割の場合：可能な候補からランダムに選択
                query_count = self.rng.choice(possible_query_counts)
            else:
                # 決定論的分割の場合：最も均等に近い候補を選択
                # base_query_countに最も近い候補を選ぶ
                query_count = min(possible_query_counts, key=lambda x: abs(x - base_query_count))

        # ★★★ ここまで修正 ★★★

        query_indices = indices[:query_count]
        target_indices = indices[query_count:]

        query_set = {k: v[query_indices] for k, v in full_set.items()}
        target_set = {k: v[target_indices] for k, v in full_set.items()}

        # 最終チェック
        assert len(query_set['features']) >= self.min_query_items
        assert len(target_set['features']) >= self.min_target_items
        
        # 差が1以下であることを確認（デバッグ用）
        query_count_final = len(query_set['features'])
        target_count_final = len(target_set['features'])
        assert abs(query_count_final - target_count_final) <= 1, f"Item count difference too large: query={query_count_final}, target={target_count_final}"
        
        return query_set, target_set

    def __len__(self):
        """1エポックあたりのバッチ数を返す"""
        return int(np.ceil(len(self.full_sets) / self.batch_size))

    def on_epoch_end(self):
        """エポック終了時にインデックスをシャッフル"""
        if self.shuffle:
            self.rng.shuffle(self.indexes)
            
    def __getitem__(self, idx):
        """1バッチ分のデータを生成する"""
        batch_indices = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # バッチ用の配列を初期化
        batch = {
            'query_features': np.zeros((len(batch_indices), self.max_item_num, self.feature_dim), dtype=np.float32),
            'target_features': np.zeros((len(batch_indices), self.max_item_num, self.feature_dim), dtype=np.float32),
            'query_categories': np.zeros((len(batch_indices), self.max_item_num), dtype=np.int32),
            'target_categories': np.zeros((len(batch_indices), self.max_item_num), dtype=np.int32),
            'query_item_ids': np.zeros((len(batch_indices), self.max_item_num), dtype=np.int32),
            'target_item_ids': np.zeros((len(batch_indices), self.max_item_num), dtype=np.int32),
        }
        
        # TPaNeg用の候補ネガティブの初期化（use_negativesがTrueの場合のみ）
        if self.use_negatives:
            batch['candidate_negative_features'] = np.zeros((len(batch_indices), self.max_item_num, self.candidate_neg_num, self.feature_dim), dtype=np.float32)
            batch['candidate_negative_masks'] = np.zeros((len(batch_indices), self.max_item_num, self.candidate_neg_num), dtype=bool)
            batch['query_candidate_negative_features'] = np.zeros_like(batch['candidate_negative_features'])
            batch['query_candidate_negative_masks'] = np.zeros_like(batch['candidate_negative_masks'])

        if self.include_SetIDs:
            batch['set_ids'] = np.empty(len(batch_indices), dtype=object)

        # 各シーンについてデータを生成
        for i, scene_idx in enumerate(batch_indices):
            full_set = self.full_sets[scene_idx]
            query_set, target_set = self._random_split_set(full_set)

            # パディングしてバッチに格納
            q_len, t_len = len(query_set['ids']), len(target_set['ids'])
            batch['query_features'][i, :q_len] = query_set['features']
            batch['query_categories'][i, :q_len] = query_set['categories']
            batch['query_item_ids'][i, :q_len] = query_set['ids']
            
            batch['target_features'][i, :t_len] = target_set['features']
            batch['target_categories'][i, :t_len] = target_set['categories']
            batch['target_item_ids'][i, :t_len] = target_set['ids']

            if self.include_SetIDs:
                batch['set_ids'][i] = self.scene_ids[scene_idx]

            # TPaNeg用の候補ネガティブをキャッシュから取得し、バッチに埋め込む
            # このロジックはバッチを埋めるループの内側に移動
            if self.use_negatives: # <--- 外側のif self.use_negatives が True の場合のみ実行
                # ターゲット(Y)に対するネガティブ候補を取得
                for item_idx_in_target_set, item_id in enumerate(target_set['ids']):
                    hard_negative_pool_ids = self.negative_cache_ids.get(str(item_id), []) 
                    neg_ids = []
                    if hard_negative_pool_ids:
                        num_to_sample = min(len(hard_negative_pool_ids), self.candidate_neg_num)
                        sampled_indices = self.rng.choice(len(hard_negative_pool_ids), size=num_to_sample, replace=False)
                        neg_ids = [hard_negative_pool_ids[j] for j in sampled_indices]
                    
                    if neg_ids:
                        sampled_neg_feats = np.array([self.item_feature_map[str(nid)] for nid in neg_ids], dtype=np.float32)
                        current_num_negs = len(sampled_neg_feats)
                        batch['candidate_negative_features'][i, item_idx_in_target_set, :current_num_negs] = sampled_neg_feats
                        batch['candidate_negative_masks'][i, item_idx_in_target_set, :current_num_negs] = True
                
                # クエリ(X)に対するネガティブ候補を取得
                for item_idx_in_query_set, item_id in enumerate(query_set['ids']):
                    hard_negative_pool_ids = self.negative_cache_ids.get(str(item_id), []) 
                    neg_ids = []
                    if hard_negative_pool_ids:
                        num_to_sample = min(len(hard_negative_pool_ids), self.candidate_neg_num)
                        sampled_indices = self.rng.choice(len(hard_negative_pool_ids), size=num_to_sample, replace=False)
                        neg_ids = [hard_negative_pool_ids[j] for j in sampled_indices]
                    
                    if neg_ids:
                        sampled_neg_feats = np.array([self.item_feature_map[str(nid)] for nid in neg_ids], dtype=np.float32)
                        current_num_negs = len(sampled_neg_feats)
                        batch['query_candidate_negative_features'][i, item_idx_in_query_set, :current_num_negs] = sampled_neg_feats
                        batch['query_candidate_negative_masks'][i, item_idx_in_query_set, :current_num_negs] = True


        # ★★★ 修正されたホワイトニング処理 ★★★
        if self.whitening_params:
            # 3次元配列用のホワイトニング（query_features, target_features）
            for key in ['query_features', 'target_features']:
                if key in batch:
                    # (B, S, D) 形状の3次元配列
                    data = batch[key]
                    batch_size, seq_len, feature_dim = data.shape
                    
                    # 有効な（非ゼロ）要素のマスクを作成
                    valid_mask = np.sum(np.abs(data), axis=-1) > 0  # (B, S)
                    
                    if valid_mask.any():
                        # 有効な要素のみを平坦化してホワイトニング適用
                        valid_data = data[valid_mask]  # (N_valid, D)
                        whitened_data = np.dot(valid_data - self.whitening_mean, self.whitening_matrix)
                        
                        # 元の配列に書き戻し
                        data[valid_mask] = whitened_data
                        batch[key] = data
        
        # テスト用に set_ids を適切なString型に変換
        if self.include_SetIDs:
            batch['set_ids'] = np.array(batch['set_ids'], dtype=str)

        # 最終的にテンソルに変換して返す（NumPyのまま渡してもTFが変換してくれるが、明示的に行うのが安全）
        final_batch = {}
        for key, value in batch.items():
            # TF.data.Datasetはstring型のnumpy arrayをそのまま処理できない場合があるため、
            # set_idsはobject型、それ以外はTensorに変換
            if key == 'set_ids' and self.include_SetIDs:
                final_batch[key] = value # object dtype array
            else:
                final_batch[key] = tf.convert_to_tensor(value) 
        
        return final_batch