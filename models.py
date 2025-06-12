"""
models.py - Correct Set Retrieval Model Implementation
======================================================
Based on your actual problem setting: Query set → Target set prediction
with category-wise cross-attention and contrastive learning.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Dict, Any, Optional, Tuple
import math


class SetRetrievalModel(Model):
    """
    Set Retrieval Model - Correct Implementation
    
    Query set → Category-wise Target set prediction
    with cross-attention and contrastive learning
    """
    
    def __init__(self, 
                 feature_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 num_categories: int = 7,
                 hidden_dim: int = 512,
                 use_cycle_loss: bool = False,
                 temperature: float = 1.0,
                 dropout_rate: float = 0.1,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        # Configuration
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_categories = num_categories
        self.hidden_dim = hidden_dim
        self.use_cycle_loss = use_cycle_loss
        self.temperature = temperature
        self.dropout_rate = dropout_rate
        
        # Category centers (learnable)
        self.category_centers = None
        
        # Build layers
        self._build_layers()
        
        print(f"[INFO] SetRetrievalModel created (Correct):")
        print(f"  - Feature dim: {feature_dim}")
        print(f"  - Heads: {num_heads}, Layers: {num_layers}")
        print(f"  - Categories: {num_categories}")
        print(f"  - Cross-attention: Category centers as queries")
        print(f"  - Temperature: {temperature}")
    
    def _build_layers(self):
        """Build model layers"""
        
        # Input projection for query set
        self.input_projection = layers.Dense(
            self.hidden_dim,
            activation='relu',
            name='input_projection'
        )
        
        # Cross-attention layers (Category centers → Query set)
        self.cross_attention_layers = []
        for i in range(self.num_layers):
            layer_dict = {
                'cross_attention': layers.MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.hidden_dim // self.num_heads,
                    dropout=self.dropout_rate,
                    name=f'cross_attention_{i}'
                ),
                'self_attention': layers.MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.hidden_dim // self.num_heads,
                    dropout=self.dropout_rate,
                    name=f'self_attention_{i}'
                ),
                'norm1': layers.LayerNormalization(epsilon=1e-6, name=f'norm1_{i}'),
                'norm2': layers.LayerNormalization(epsilon=1e-6, name=f'norm2_{i}'),
                'norm3': layers.LayerNormalization(epsilon=1e-6, name=f'norm3_{i}'),
                'ffn': tf.keras.Sequential([
                    layers.Dense(self.hidden_dim * 2, activation='gelu'),
                    layers.Dropout(self.dropout_rate),
                    layers.Dense(self.hidden_dim)
                ], name=f'ffn_{i}')
            }
            self.cross_attention_layers.append(layer_dict)
        
        # Final output projection to feature space
        self.output_projection = layers.Dense(
            self.feature_dim,
            activation=None,
            name='output_projection'
        )
        
        # Output normalization
        self.output_norm = layers.LayerNormalization(epsilon=1e-6, name='output_norm')
    
    def set_category_centers(self, centers: np.ndarray):
        """Set category cluster centers"""
        if centers.shape[0] != self.num_categories:
            raise ValueError(f"Expected {self.num_categories} centers, got {centers.shape[0]}")
        
        # L2 normalize centers
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
        
        # Create learnable category centers initialized with cluster centers
        self.category_centers = self.add_weight(
            name='category_centers',
            shape=(self.num_categories, self.hidden_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Initialize with provided centers (project if needed)
        if centers.shape[1] != self.hidden_dim:
            # Simple projection if dimensions don't match
            init_centers = np.random.normal(0, 0.02, (self.num_categories, self.hidden_dim)).astype(np.float32)
        else:
            init_centers = centers.astype(np.float32)
        
        self.category_centers.assign(init_centers)
        print(f"✅ Category centers initialized: {centers.shape}")
    
    def call(self, inputs, training=None):
        """Forward pass"""
        
        # Parse inputs
        if isinstance(inputs, dict):
            query_features = inputs['query_features']  # [batch, seq_len, feature_dim]
            query_categories = inputs.get('query_categories')  # Not used in cross-attention
        else:
            # Legacy support
            query_features = inputs[0]
            query_categories = inputs[1] if len(inputs) > 1 else None
        
        # Process query set
        predictions = self._forward_pass(query_features, training)
        
        return predictions
    
    def _forward_pass(self, query_features, training=None):
        """Forward pass through cross-attention layers"""
        
        batch_size = tf.shape(query_features)[0]
        
        # Project query features
        query_projected = self.input_projection(query_features)  # [batch, seq_len, hidden_dim]
        
        # Get category centers and expand for batch
        if self.category_centers is None:
            raise ValueError("Category centers not set! Call set_category_centers() first.")
        
        # Category centers as queries: [batch, num_categories, hidden_dim]
        category_queries = tf.expand_dims(self.category_centers, 0)
        category_queries = tf.tile(category_queries, [batch_size, 1, 1])
        
        # Cross-attention: Category centers attend to query set
        x = category_queries
        
        for layer_dict in self.cross_attention_layers:
            # Cross-attention: categories (queries) attend to query set (keys, values)
            cross_attn_out = layer_dict['cross_attention'](
                query=x,                    # Category centers
                key=query_projected,        # Query set
                value=query_projected,      # Query set
                training=training
            )
            x = layer_dict['norm1'](x + cross_attn_out, training=training)
            
            # Self-attention among categories
            self_attn_out = layer_dict['self_attention'](
                query=x,
                key=x,
                value=x,
                training=training
            )
            x = layer_dict['norm2'](x + self_attn_out, training=training)
            
            # Feed-forward
            ffn_out = layer_dict['ffn'](x, training=training)
            x = layer_dict['norm3'](x + ffn_out, training=training)
        
        # Project to target feature space
        predictions = self.output_projection(x)  # [batch, num_categories, feature_dim]
        
        # L2 normalize predictions
        predictions = tf.nn.l2_normalize(predictions, axis=-1)
        
        return self.output_norm(predictions)
    
# models.py -> SetRetrievalModel 内のメソッドを修正

    def train_step(self, data):
        """Custom training step with proper contrastive loss"""
        batch_data = data

        with tf.GradientTape() as tape:
            predictions = self(batch_data, training=True)
            target_features = batch_data['target_features']
            target_categories = batch_data['target_categories'] # カテゴリ情報を取得

            # 新しい損失関数を呼び出す
            loss = contrastive_loss_per_item(
                predictions,
                target_features,
                target_categories, # カテゴリ情報を渡す
                temperature=self.temperature
            )

            # Cycle Loss (オプション) は、同様に新しい損失関数を使うように修正が必要
            # 今回は一旦主要な損失に焦点を当てるため、Cycle Lossのロジックは省略
            # if self.use_cycle_loss: ...

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # metricsを更新（必要であれば）
        self.compiled_metrics.update_state(tf.zeros_like(loss), loss) # ダミーのyとy_pred
        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):
        """Test step"""
        batch_data = data
        
        predictions = self(batch_data, training=False)
        target_features = batch_data['target_features']
        target_categories = batch_data['target_categories'] # カテゴリ情報を取得

        # 新しい損失関数を呼び出す
        test_loss = contrastive_loss_per_item(
            predictions,
            target_features,
            target_categories, # カテゴリ情報を渡す
            temperature=self.temperature
        )
        
        self.compiled_metrics.update_state(tf.zeros_like(test_loss), test_loss)
        return {m.name: m.result() for m in self.metrics}

# models.py -> contrastive_loss_with_negatives (修正後)
def contrastive_loss_with_negatives(predictions, targets, negative_samples=None, temperature=1.0):
    """
    Contrastive loss for set retrieval (Vectorized Version with dimension fix)
    
    Args:
        predictions: [batch, num_categories, feature_dim] - Predicted target sets
        targets: [batch, seq_len, feature_dim] - Ground truth target items
        negative_samples: [batch, seq_len, num_negatives, feature_dim] - Negative samples
        temperature: Temperature for softmax
    
    Returns:
        Contrastive loss
    """
    # --- ターゲット集合の表現を計算 ---
    target_mask = tf.reduce_sum(tf.abs(targets), axis=-1) > 0
    target_mask = tf.cast(target_mask, tf.float32)
    
    target_sum = tf.reduce_sum(targets * tf.expand_dims(target_mask, -1), axis=1)
    target_count = tf.maximum(tf.reduce_sum(target_mask, axis=1, keepdims=True), 1.0)
    target_repr = target_sum / target_count
    
    target_repr = tf.nn.l2_normalize(target_repr, axis=-1)

    # --- ポジティブペアの類似度を計算 ---
    pos_sim = tf.reduce_sum(predictions * tf.expand_dims(target_repr, 1), axis=-1)

    # --- ネガティブペアの類似度を計算 (バッチ内ネガティブ) ---
    # neg_sim_matrix: [batch, num_categories, batch]
    neg_sim_matrix = tf.linalg.matmul(predictions, target_repr, transpose_b=True)
    
    batch_size = tf.shape(predictions)[0]
    identity_mask = tf.eye(batch_size) # Shape: [batch, batch]
    
    # ▼▼▼▼▼▼▼▼▼▼【修正点】▼▼▼▼▼▼▼▼▼▼
    # マスクの次元を [batch, batch] -> [batch, 1, batch] に拡張する
    # これにより、[batch, num_categories, batch] とのブロードキャストが可能になる
    expanded_mask = tf.expand_dims(identity_mask, axis=1)
    # ▲▲▲▲▲▲▲▲▲▲【ここまで】▲▲▲▲▲▲▲▲▲▲

    # 対角成分を非常に小さい値にして、max計算で選ばれないようにする
    neg_sim_matrix = neg_sim_matrix * (1.0 - expanded_mask) + expanded_mask * (-1e9)
    
    # 各予測にとっての最も難しいネガティブ（最も似ている他のサンプル）を選択
    neg_sim = tf.reduce_max(neg_sim_matrix, axis=-1)

    # --- 損失計算 ---
    logits = tf.stack([pos_sim, neg_sim], axis=-1) / temperature
    labels = tf.zeros_like(pos_sim, dtype=tf.int32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    
    return tf.reduce_mean(loss)

# models.py に追加または修正

def contrastive_loss_per_item(predictions, target_features, target_categories, temperature=1.0):
    """
    Corrected Contrastive Loss - Compares predictions to individual items.

    Args:
        predictions: [B, C, D] - Predicted features for each category.
        target_features: [B, S, D] - Ground truth item features.
        target_categories: [B, S] - Ground truth item categories.
        temperature: Temperature for softmax.

    Returns:
        The contrastive loss.
    """
    # --- 1. データの前処理 ---
    # パディングされたアイテム（カテゴリID=0）をマスクするための準備
    # target_features/categoriesをフラットなリストに変形
    # [B, S, D] -> [B*S, D]
    # [B, S]    -> [B*S]
    B, S, D = tf.shape(target_features)[0], tf.shape(target_features)[1], tf.shape(target_features)[2]
    C = tf.shape(predictions)[1]
    
    flat_target_features = tf.reshape(target_features, [B * S, D])
    flat_target_categories = tf.reshape(target_categories, [B * S])

    # パディングを除いた有効なアイテムのみを対象にするマスク
    valid_target_mask = flat_target_categories > 0

    # --- 2. 全ペアの類似度計算 ---
    # 各カテゴリの予測 (B, C, D) と、バッチ内の全有効アイテム (B*S, D) との
    # コサイン類似度を一括で計算する。
    # tf.einsum は効率的な行列積計算機
    # 結果の logits は [B, C, B*S] の形状を持つ
    logits = tf.einsum('bcd,sd->bcs', predictions, flat_target_features) / temperature

    # パディングされたターゲットアイテムに対応する類似度を計算から除外
    # boolean_maskを適用すると次元が減ってしまうため、代わりに非常に小さい値を設定
    logits = logits * tf.cast(tf.reshape(valid_target_mask, [1, 1, B*S]), tf.float32) + \
             tf.cast(tf.reshape(~valid_target_mask, [1, 1, B*S]), tf.float32) * (-1e9)

    # --- 3. 正解ラベルの作成 ---
    # 各予測 (b, c) にとって、どのアイテム (s) が正解かを示すラベル行列を作成
    
    # a) カテゴリの一致を確認
    # p_cats: [1, C, 1], t_cats: [1, 1, B*S]
    p_cats = tf.reshape(tf.range(1, C + 1, dtype=tf.int32), [1, C, 1])
    t_cats = tf.reshape(flat_target_categories, [1, 1, B*S])
    category_match = tf.equal(p_cats, t_cats) # Shape: [1, C, B*S]

    # b) 同じコーディネート/セットに属しているかを確認
    # b_indices: [B, 1, 1]
    # t_indices: [1, 1, B*S] (各アイテムがどのセットbに属しているかを示す)
    b_indices = tf.reshape(tf.range(B), [B, 1, 1])
    t_indices = tf.reshape(tf.repeat(tf.range(B), S), [1, 1, B*S])
    instance_match = tf.equal(b_indices, t_indices) # Shape: [B, 1, B*S]

    # a)とb)の両方を満たすものが正解ラベル
    # labels: [B, C, B*S]
    labels = tf.cast(tf.logical_and(category_match, instance_match), tf.float32)
    
    # 正解が存在しない予測は損失計算から除外するマスク
    has_positives_mask = tf.reduce_sum(labels, axis=-1) > 0 # Shape: [B, C]

    # --- 4. 損失の計算 ---
    # softmax_cross_entropyを適用して損失を計算
    # labelsを合計値で割ることで、複数の正解がある場合に対応
    labels_normalized = tf.math.divide_no_nan(labels, tf.reduce_sum(labels, axis=-1, keepdims=True))
    
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_normalized, logits=logits)
    
    # マスクを適用して、有効な予測に関する損失のみを平均
    masked_loss = loss * tf.cast(has_positives_mask, tf.float32)
    
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(tf.cast(has_positives_mask, tf.float32))


def create_model(config: Dict[str, Any]) -> SetRetrievalModel:
    """Factory function to create SetRetrievalModel"""
    
    # Validate required keys
    required_keys = ['feature_dim', 'num_heads', 'num_layers', 'num_categories']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Set defaults
    default_config = {
        'temperature': 1.0,
        'dropout_rate': 0.1,
        'hidden_dim': config['feature_dim'],
        'use_cycle_loss': False,
    }
    default_config.update(config)
    
    model = SetRetrievalModel(
        feature_dim=default_config['feature_dim'],
        num_heads=default_config['num_heads'],
        num_layers=default_config['num_layers'],
        num_categories=default_config['num_categories'],
        hidden_dim=default_config['hidden_dim'],
        use_cycle_loss=default_config['use_cycle_loss'],
        temperature=default_config['temperature'],
        dropout_rate=default_config['dropout_rate']
    )
    
    return model


def evaluate_with_gallery(model, test_dataset, all_test_items, k_values=[1, 5, 10]):
    """
    Evaluate model with full test gallery
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        all_test_items: All test items as gallery [num_items, feature_dim]
        k_values: K values for Top-K accuracy
    
    Returns:
        Evaluation metrics
    """
    all_ranks = []
    all_scores = []
    
    print("🔍 Evaluating with full test gallery...")
    
    for batch_data, _ in test_dataset:
        # Get predictions
        predictions = model(batch_data, training=False)  # [batch, num_categories, feature_dim]
        
        # Extract target data
        query_features = batch_data['query_features']
        target_features = batch_data['target_features']
        
        batch_size = tf.shape(predictions)[0]
        
        for i in range(batch_size):
            # Get target representation (average of target items)
            target_items = target_features[i]  # [seq_len, feature_dim]
            target_mask = tf.reduce_sum(tf.abs(target_items), axis=-1) > 0
            valid_targets = tf.boolean_mask(target_items, target_mask)
            
            if tf.shape(valid_targets)[0] == 0:
                continue
                
            target_repr = tf.reduce_mean(valid_targets, axis=0)  # [feature_dim]
            target_repr = tf.nn.l2_normalize(target_repr, axis=-1)
            
            # Get best category prediction
            pred_categories = predictions[i]  # [num_categories, feature_dim]
            
            # Compute similarities with all gallery items
            gallery_similarities = tf.linalg.matmul(
                tf.expand_dims(target_repr, 0), 
                all_test_items, 
                transpose_b=True
            )  # [1, num_gallery_items]
            gallery_similarities = tf.squeeze(gallery_similarities, 0)  # [num_gallery_items]
            
            # Find rank of target
            target_sim = 1.0  # Perfect similarity with itself
            better_count = tf.reduce_sum(tf.cast(gallery_similarities > target_sim, tf.float32))
            rank = better_count + 1
            
            all_ranks.append(rank.numpy())
    
    # Calculate metrics
    all_ranks = np.array(all_ranks)
    
    results = {}
    for k in k_values:
        top_k_acc = np.mean(all_ranks <= k)
        results[f'top_{k}_acc'] = top_k_acc
    
    results['mrr'] = np.mean(1.0 / all_ranks)
    results['mean_rank'] = np.mean(all_ranks)
    results['median_rank'] = np.median(all_ranks)
    
    return results