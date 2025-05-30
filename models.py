import os
import pdb
import sys
import psutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization

from util import inbatch_cat_rank
import pdb

tf.random.set_seed(42)

################################################################################
# 1. Utility: Safe L2 normalization
################################################################################
def safe_l2_normalize(x, axis=-1, eps=1e-7):
    """
    Safely performs L2 normalization along a given axis, avoiding division by zero.
    """
    norm_value = tf.norm(x, axis=axis, keepdims=True)
    safe_value = tf.maximum(norm_value, eps)
    return x / safe_value

################################################################################
# 2. MLP Layer with name parameter for parameter counting
################################################################################
class MLP(tf.keras.layers.Layer):
    """
    Multi-layer perceptron with configurable name for parameter counting
    """
    def __init__(self, hidden_dim=128, out_dim=64, is_softmax=0, is_mask=False, name='mixer', **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.is_softmax = is_softmax
        self.is_mask = is_mask
        self.layer_name = name
        
        # Dense layers with specified name
        self.linear = Dense(self.hidden_dim, activation="linear", name=name)
        self.linear2 = Dense(out_dim, activation="linear", name=name)
        
        if is_softmax:
            self.softmax = tf.keras.layers.Softmax()
    
    def call(self, inputs, training=False):
        x = self.linear(inputs)
        x = tf.nn.relu(x)
        x = self.linear2(x)
        
        if self.is_softmax:
            x = self.softmax(x)
        
        if self.is_mask and training:
            # Apply mask during training
            mask = tf.cast(tf.reduce_sum(inputs, axis=-1, keepdims=True) > 0, tf.float32)
            x = x * mask
            
        return x

################################################################################
# 3. Contrastive loss: Center Base版とOriginal版
################################################################################
def compute_combined_contrastive_loss(
    predicted_vectors: tf.Tensor,    # (B, N, D) or (B, C, D) - Cは11カテゴリ
    positive_vectors: tf.Tensor,     # (B, N, D)
    positive_categories: tf.Tensor,  # (B, N)  1-based IDs
    negative_gallery: tf.Tensor,     # (B, N, K, D)
    cluster_centers: tf.Tensor = None,      # (C, D) - Noneの場合はCenter Baseを使わない
    use_center_base: bool = True,
    alpha: float = 0.9
) -> tf.Tensor:
    """
    カテゴリ中心を差し引いた残差ベクトル同士で対比損失を計算（Center Base版）
    または通常の対比損失を計算（Original版）
    
    IQON3000データセットでは1-11のカテゴリIDを使用
    """
    if use_center_base and cluster_centers is not None:
        # Center Base Loss版
        # predicted_vectorsが(B, C, D)の形状の場合、カテゴリごとに処理
        if len(predicted_vectors.shape) == 3 and predicted_vectors.shape[1] >= 11:
            # (B, N)のカテゴリ→0-based index (1-11 -> 0-10)
            cat_idx = tf.clip_by_value(positive_categories - 1, 0, 10)
            
            # 各アイテムに対応する予測ベクトルを選択
            batch_size = tf.shape(predicted_vectors)[0]
            batch_indices = tf.range(batch_size)[:, tf.newaxis]
            cat_idx_expanded = cat_idx
            
            # (B, N, D) 形状で予測ベクトルを抽出
            pred_selected = tf.gather_nd(predicted_vectors, 
                                       tf.stack([tf.broadcast_to(batch_indices, tf.shape(cat_idx)),
                                               cat_idx], axis=-1))
            
            # (B, N, D) の各ベクトルに対応する中心ベクトルを取り出し
            centers = tf.gather(cluster_centers, cat_idx, batch_dims=0)

            # 残差を計算
            pred_res = pred_selected - centers        # (B,N,D)
            pos_res  = positive_vectors  - centers    # (B,N,D)
            centers_neg = tf.expand_dims(centers, 2)  # (B,N,1,D)
            neg_res  = negative_gallery - centers_neg # (B,N,K,D)

            # L2 正規化
            pred_norm = tf.nn.l2_normalize(pred_res, axis=-1)
            pos_norm  = tf.nn.l2_normalize(pos_res,  axis=-1)
            neg_norm  = tf.nn.l2_normalize(neg_res,  axis=-1)
        else:
            # (B, N, D) → (B, N, D) の場合
            cat_idx = tf.clip_by_value(positive_categories - 1, 0, tf.shape(cluster_centers)[0]-1)
            centers = tf.gather(cluster_centers, cat_idx, batch_dims=0)

            pred_res = predicted_vectors - centers
            pos_res  = positive_vectors  - centers
            centers_neg = tf.expand_dims(centers, 2)
            neg_res  = negative_gallery - centers_neg

            pred_norm = tf.nn.l2_normalize(pred_res, axis=-1)
            pos_norm  = tf.nn.l2_normalize(pos_res,  axis=-1)
            neg_norm  = tf.nn.l2_normalize(neg_res,  axis=-1)
    else:
        # Original版（Center Baseを使わない）
        if len(predicted_vectors.shape) == 3 and predicted_vectors.shape[1] >= 11:
            # カテゴリ予測から該当ベクトルを選択
            cat_idx = tf.clip_by_value(positive_categories - 1, 0, 10)
            batch_size = tf.shape(predicted_vectors)[0]
            batch_indices = tf.range(batch_size)[:, tf.newaxis]
            
            pred_selected = tf.gather_nd(predicted_vectors, 
                                       tf.stack([tf.broadcast_to(batch_indices, tf.shape(cat_idx)),
                                               cat_idx], axis=-1))
            pred_norm = tf.nn.l2_normalize(pred_selected, axis=-1)
        else:
            pred_norm = tf.nn.l2_normalize(predicted_vectors, axis=-1)
            
        pos_norm  = tf.nn.l2_normalize(positive_vectors,  axis=-1)
        neg_norm  = tf.nn.l2_normalize(negative_gallery,  axis=-1)

    # コサイン類似度
    sim_pos = tf.reduce_sum(pred_norm * pos_norm, axis=-1)          # (B,N)
    sim_neg = tf.einsum('bnd,bnkd->bnk', pred_norm, neg_norm)       # (B,N,K)

    # ハードネガティブだけ残す
    thresh = alpha * tf.expand_dims(sim_pos, -1)
    hard_mask = tf.cast(sim_neg - tf.expand_dims(sim_pos, -1) >= thresh, tf.float32)

    # ソフトマックス風損失
    exp_pos = tf.exp(sim_pos)
    exp_neg = tf.exp(sim_neg) * hard_mask
    neg_sum = tf.reduce_sum(exp_neg, axis=-1)
    denom   = tf.maximum(exp_pos + neg_sum, 1e-7)

    loss = - tf.math.log(exp_pos / denom)  # (B,N)
    return tf.reduce_mean(loss)

################################################################################
# 4. In-batch loss with center-based computation
################################################################################
def compute_inbatch_mixed_loss(
    category_predictions: tf.Tensor,   # (B, C, D) - Cは11カテゴリ
    ground_truth_vectors: tf.Tensor,   # (B, N, D)
    category_labels: tf.Tensor,        # (B, N) 1-based IDs (1-11)
    cluster_centers: tf.Tensor,        # (C, D) - 11カテゴリ分
    margin: float = 1.0,
    alpha: float = 0.5
) -> tf.Tensor:
    """
    カテゴリ中心差分での Cosine-margin + L2 混合損失
    IQON3000データセット用（1-11のカテゴリID）
    """
    batch_size = tf.shape(category_predictions)[0]

    def _one_sample(b):
        preds = category_predictions[b]      # (C,D) - 11カテゴリ
        gt    = ground_truth_vectors[b]      # (N,D)
        cats  = tf.clip_by_value(category_labels[b]-1, 0, 10)  # 1-11 -> 0-10

        # 有効アイテムだけ抽出（カテゴリID 1-11）
        valid = tf.where(tf.logical_and(category_labels[b] >= 1, category_labels[b] <= 11))[:,0]
        if tf.size(valid)==0:
            return tf.constant(0.0)

        cats_sel = tf.gather(cats, valid)       # (V,) - 0-based
        centers  = tf.gather(cluster_centers, cats_sel)  # (V,D)
        pred_sel = tf.gather(preds, cats_sel)    # (V,D)
        gt_sel   = tf.gather(gt,   valid)        # (V,D)

        # 残差→正規化
        pred_res = pred_sel - centers
        gt_res   = gt_sel   - centers
        p_norm = tf.nn.l2_normalize(pred_res, axis=-1)
        g_norm = tf.nn.l2_normalize(gt_res,   axis=-1)

        # Cosine-margin
        cos_pos = tf.reduce_sum(p_norm * g_norm, axis=-1)
        cos_neg = tf.reduce_sum(tf.reduce_mean(g_norm,axis=0)*p_norm, axis=-1)
        cos_loss = tf.nn.relu(margin + cos_neg - cos_pos)

        # L2 距離
        l2_loss = tf.norm(pred_res - gt_res, axis=-1)

        # 混合
        mix_loss = alpha * cos_loss + (1-alpha) * l2_loss
        return tf.reduce_mean(mix_loss)

    losses = tf.map_fn(_one_sample, tf.range(batch_size), dtype=tf.float32)
    return tf.reduce_mean(losses)

################################################################################
# 5. Enhanced PivotLayer with named components
################################################################################
class PivotLayer(tf.keras.layers.Layer):
    """
    A Transformer-inspired layer with named components for parameter counting
    """
    def __init__(self, dim, num_heads=2, ff_dim=512, dropout_rate=0.1, layer_name="attention", **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.layer_name = layer_name
        
        # Attention layers with proper naming
        self.self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=dim // num_heads,
            name=f"{layer_name}_self_attention"
        )
        self.cross_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=dim // num_heads,
            name=f"{layer_name}_cross_attention"
        )
        
        # Dropout and normalization layers
        self.dropout_self = Dropout(dropout_rate, name=f"{layer_name}_dropout_self")
        self.dropout_cross = Dropout(dropout_rate, name=f"{layer_name}_dropout_cross")
        self.dropout_ffn = Dropout(dropout_rate, name=f"{layer_name}_dropout_ffn")

        self.norm_self = LayerNormalization(epsilon=1e-6, name=f"{layer_name}_norm_self")
        self.norm_cross = LayerNormalization(epsilon=1e-6, name=f"{layer_name}_norm_cross")
        self.norm_ffn = LayerNormalization(epsilon=1e-6, name=f"{layer_name}_norm_ffn")

        # Feed-forward network with named components
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation=tf.nn.gelu, name=f"{layer_name}_ffn_1"),
            Dense(dim, name=f"{layer_name}_ffn_2")
        ], name=f"{layer_name}_ffn")

    def call(self, inputs, training=False):
        """
        inputs: (input_set, cluster_center)
          - input_set:      (B, N, dim)
          - cluster_center: (B, M, dim) - Mは11カテゴリ
        """
        input_set, cluster_center = inputs

        # 1. Self-attention on input_set
        input_norm = self.norm_self(input_set)
        attn_output = self.self_attn(query=input_norm, key=input_norm, value=input_norm)
        attn_output = self.dropout_self(attn_output, training=training)
        self_attended = input_set + attn_output

        # 2. Cross-attention (cluster_center is query, input_set is key/value)
        center_norm = self.norm_cross(cluster_center)
        cross_output = self.cross_attn(query=center_norm, key=self_attended, value=self_attended)
        cross_output = self.dropout_cross(cross_output, training=training)
        cross_attended = cluster_center + cross_output

        # 3. Feed-forward
        ffn_input = self.norm_ffn(cross_attended)
        ffn_out = self.ffn(ffn_input)
        ffn_out = self.dropout_ffn(ffn_out, training=training)
        
        return cross_attended + ffn_out

################################################################################
# 6. Enhanced SetRetrievalModel with parameter counting and GPU optimization
################################################################################
class SetRetrievalModel(tf.keras.Model):
    """
    Set retrieval model with enhanced GPU utilization and parameter counting
    IQON3000データセット用（1-11のカテゴリID）
    """
    def __init__(
        self, 
        dim=512, 
        num_layers=2, 
        num_heads=2, 
        ff_dim=512,
        cycle_lambda=0.2, 
        use_cycle_loss=False, 
        use_CLNeg_loss=False, 
        use_center_base=False,
        name="SetRetrievalModel",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.cycle_lambda = cycle_lambda
        self.use_cycle_loss = use_cycle_loss
        self.use_CLNeg_loss = use_CLNeg_loss
        self.use_center_base = use_center_base

        # Build pivot layers with proper naming
        self.pivot_layers = [
            PivotLayer(dim, num_heads, ff_dim, dropout_rate=0.1, layer_name=f"attention_layer_{i}") 
            for i in range(num_layers)
        ]
        
        # Final processing layers with proper naming
        self.norm_final = LayerNormalization(epsilon=1e-6, name="final_norm")
        self.final_dense = Dense(dim, kernel_initializer="he_normal", name="final_dense")

        # Add MLP layers for mixing operations
        self.MLP1 = MLP(dim, out_dim=dim, is_mask=True, name='mixer')
        self.MLP2 = MLP(dim, out_dim=ff_dim, is_softmax=False, name='mixer')
        self.MLP3 = MLP(dim, out_dim=1, name='mixer')

        # Buffers for intermediate states
        self._cluster_centers = None  
        self._predicted_11cats = None
        self._ground_truth_set = None
        self._catQ_buffer = None
        self._catP_buffer = None
        self._original_input = None

        # For test steps
        self.test_step_count = tf.Variable(0, trainable=False, dtype=tf.int64)

        # Average-rank metrics
        self.train_avg_rank = tf.keras.metrics.Mean(name="train_avg_rank")
        self.val_avg_rank = tf.keras.metrics.Mean(name="val_avg_rank")

    def count_parameters(self):
        """
        Count parameters by category (mixer, attention, and total)
        Returns dictionary with parameter counts
        """
        mixer_params = 0
        attention_params = 0
        other_params = 0
        counted_weights = set()
        
        # Count parameters from all trainable weights
        for weight in self.trainable_weights:
            weight_id = id(weight)  # Use object ID to avoid shape comparison issues
            if weight_id in counted_weights:
                continue
                
            weight_name = weight.name.lower()
            weight_params = tf.size(weight).numpy()
            
            # Categorize based on weight name
            if 'mixer' in weight_name:
                mixer_params += weight_params
            elif 'attention' in weight_name or 'attn' in weight_name:
                attention_params += weight_params
            else:
                other_params += weight_params
                
            counted_weights.add(weight_id)
        
        total_params = mixer_params + attention_params + other_params
        
        return {
            'mixer_params': mixer_params,
            'attention_params': attention_params,
            'other_params': other_params,
            'total_params': total_params
        }

    def display_parameter_summary(self):
        """
        Display a formatted summary of model parameters
        """
        try:
            param_counts = self.count_parameters()
            
            print("\n" + "="*60)
            print("MODEL PARAMETER SUMMARY")
            print("="*60)
            print(f"Mixer Parameters:     {param_counts['mixer_params']:,}")
            print(f"Attention Parameters: {param_counts['attention_params']:,}")
            print(f"Other Parameters:     {param_counts['other_params']:,}")
            print("-"*60)
            print(f"Total Parameters:     {param_counts['total_params']:,}")
            print("="*60)
            
            return param_counts
        except Exception as e:
            print(f"[WARN] Could not count parameters: {e}")
            # Fallback to simple total count
            try:
                total_params = sum([tf.size(w).numpy() for w in self.trainable_weights])
                print(f"[INFO] Total Parameters: {total_params:,}")
                return {'total_params': total_params, 'mixer_params': 0, 'attention_params': 0, 'other_params': total_params}
            except Exception as e2:
                print(f"[ERROR] Could not count any parameters: {e2}")
                return {'total_params': 0, 'mixer_params': 0, 'attention_params': 0, 'other_params': 0}

    def set_cluster_center(self, cluster_centers):
        """
        クラスタ中心を設定する関数を修正
        IQON3000データセット用（11カテゴリ）
        
        引数:
            cluster_centers: カテゴリIDとセンターの辞書またはnumpy配列
        """
        if cluster_centers is None:
            print("[WARN] None cluster centers provided, using dummy data")
            dummy_centers = np.zeros((11, self.dim), dtype=np.float32)
            self._cluster_centers = tf.constant(dummy_centers, dtype=tf.float32)
            return
        
        # 辞書型の場合
        if isinstance(cluster_centers, dict):
            if not cluster_centers:  # 空の辞書
                print("[WARN] Empty cluster centers dictionary, using dummy data")
                dummy_centers = np.zeros((11, self.dim), dtype=np.float32)
                self._cluster_centers = tf.constant(dummy_centers, dtype=tf.float32)
                return
                
            # 辞書からnumpy配列に変換（IQON3000は1-11のカテゴリID）
            center_array = np.zeros((11, next(iter(cluster_centers.values())).shape[0]), 
                                dtype=np.float32)
            
            for cat_id, center in cluster_centers.items():
                if 1 <= cat_id <= 11:  # 有効なカテゴリIDのみ
                    center_array[cat_id - 1] = center  # 0-basedインデックスに変換
        else:
            # NumPy配列の場合、サイズチェック
            if isinstance(cluster_centers, np.ndarray) and cluster_centers.size == 0:
                print("[WARN] Empty numpy array for cluster centers, using dummy data")
                dummy_centers = np.zeros((11, self.dim), dtype=np.float32)
                self._cluster_centers = tf.constant(dummy_centers, dtype=tf.float32)
                return
                
            # すでに配列形式の場合はそのまま使用
            center_array = np.array(cluster_centers, dtype=np.float32)
            
            # 11カテゴリ分でない場合の処理
            if center_array.shape[0] != 11:
                print(f"[WARN] Expected 11 categories, got {center_array.shape[0]}. Padding or truncating.")
                if center_array.shape[0] < 11:
                    # パディング
                    padded_array = np.zeros((11, center_array.shape[1]), dtype=np.float32)
                    padded_array[:center_array.shape[0]] = center_array
                    center_array = padded_array
                else:
                    # トリミング
                    center_array = center_array[:11]
        
        # Tensorに変換
        self._cluster_centers = tf.constant(center_array, dtype=tf.float32)
        print(f"[INFO] Set cluster centers with shape {center_array.shape} for 11 categories")

    def get_cluster_center(self):
        if self._cluster_centers is None:
            raise ValueError("Cluster center has not been set.")
        return self._cluster_centers

    def forward_pass(self, input_set, training=False):
        """
        CPU負荷を抑制したforward pass
        11カテゴリ分の出力を生成
        """
        batch_size = tf.shape(input_set)[0]
        center_const = self.get_cluster_center()  # (11, dim)
        center_const = tf.expand_dims(center_const, axis=0)  # (1, 11, dim)
        repeated_centers = tf.tile(center_const, [batch_size, 1, 1])  # (B, 11, dim)

        features_out = repeated_centers
        for layer in self.pivot_layers:
            features_out = layer((input_set, features_out), training=training)
        
        output = self.final_dense(features_out)  # (B, 11, dim)
        return output

    def call(self, inputs, training=False, direction="XY"):
        """
        Enhanced call method with better GPU utilization
        """
        (X_concat, _, _, catQ_batch, catP_batch, X_ids_batch, Y_ids_batch, neg_gallery) = inputs

        batch_2 = tf.shape(X_concat)[0]
        batch_1 = batch_2 // 2

        if direction == "XY":
            input_part = X_concat[:batch_1]
            target_part = X_concat[batch_1:]
            self._catQ_buffer = catQ_batch
            self._catP_buffer = catP_batch
            self._original_input = input_part
        elif direction == "YX":
            input_part = X_concat[batch_1:]
            target_part = X_concat[:batch_1]
            self._catQ_buffer = catQ_batch
            self._catP_buffer = catP_batch
            self._original_input = input_part
        else:
            raise ValueError("direction must be 'XY' or 'YX'")

        out = self.forward_pass(input_part, training=training)
        self._predicted_11cats = out
        self._ground_truth_set = target_part
        return out

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, data):
        """
        Optimized training step with improved GPU utilization
        """
        (batch_data, set_ids) = data
        (X_concat, y_init, x_size, catQ_batch, catP_batch, X_ids_batch, Y_ids_batch, neg_gallery) = batch_data
        batch_2 = tf.shape(X_concat)[0]
        batch_1 = batch_2 // 2

        with tf.GradientTape() as tape:
            # XY direction
            if self.use_CLNeg_loss:
                # Combined contrastive
                self(batch_data, training=True, direction="XY")
                loss_forward_query_to_target = compute_combined_contrastive_loss(
                    self._predicted_11cats, 
                    self._ground_truth_set, 
                    self._catP_buffer, 
                    neg_gallery[:batch_1],
                    self.get_cluster_center(),
                    self.use_center_base,
                    alpha=0.9
                )
                avg_rank_query_to_target = tf.py_function(
                    func=inbatch_cat_rank,
                    inp=[self._predicted_11cats, self._ground_truth_set, self._catP_buffer, self.test_step_count],
                    Tout=tf.float32
                )

                # YX direction
                self(batch_data, training=True, direction="YX")
                loss_forward_target_to_query = compute_combined_contrastive_loss(
                    self._predicted_11cats,
                    self._ground_truth_set,
                    self._catQ_buffer,
                    neg_gallery[batch_1:],
                    self.get_cluster_center(),
                    self.use_center_base,
                    alpha=0.9
                )
                avg_rank_target_to_query = tf.py_function(
                    func=inbatch_cat_rank,
                    inp=[self._predicted_11cats, self._ground_truth_set, self._catQ_buffer, self.test_step_count],
                    Tout=tf.float32
                )
            else:
                # Legacy in-batch
                self(batch_data, training=True, direction="XY")
                loss_forward_query_to_target = compute_inbatch_mixed_loss(
                    category_predictions=self._predicted_11cats,
                    ground_truth_vectors=self._ground_truth_set,
                    category_labels=self._catP_buffer,
                    cluster_centers=self.get_cluster_center(),
                    margin=1.0, alpha=0.5
                )
                avg_rank_query_to_target = tf.py_function(
                    func=inbatch_cat_rank,
                    inp=[self._predicted_11cats, self._ground_truth_set, self._catP_buffer, self.test_step_count],
                    Tout=tf.float32
                )

                self(batch_data, training=True, direction="YX")
                loss_forward_target_to_query = compute_inbatch_mixed_loss(
                    category_predictions=self._predicted_11cats,
                    ground_truth_vectors=self._ground_truth_set,
                    category_labels=self._catQ_buffer,
                    cluster_centers=self.get_cluster_center(),
                    margin=1.0, alpha=0.5
                )
                avg_rank_target_to_query = tf.py_function(
                    func=inbatch_cat_rank,
                    inp=[self._predicted_11cats, self._ground_truth_set, self._catQ_buffer, self.test_step_count],
                    Tout=tf.float32
                )

            mean_avg_rank = (avg_rank_query_to_target + avg_rank_target_to_query) / 2.0

            # Cycle consistency (optional)
            if self.use_cycle_loss:
                cycle_output_qt = self.forward_pass(self._predicted_11cats, training=False)
                cycle_loss_qt = compute_inbatch_mixed_loss(
                    cycle_output_qt,
                    self._original_input,
                    self._catQ_buffer,
                    cluster_centers=self.get_cluster_center(),
                    margin=1.0, alpha=0.05
                )
                cycle_output_tq = self.forward_pass(self._predicted_11cats, training=False)
                cycle_loss_tq = compute_inbatch_mixed_loss(
                    cycle_output_tq,
                    self._original_input,
                    self._catP_buffer,
                    cluster_centers=self.get_cluster_center(),
                    margin=1.0, alpha=0.05
                )
                cycle_loss_total = (cycle_loss_qt + cycle_loss_tq) / 2.0
            else:
                cycle_loss_total = 0.0

            total_loss = loss_forward_query_to_target + loss_forward_target_to_query + self.cycle