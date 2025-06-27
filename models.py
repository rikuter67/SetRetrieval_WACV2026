"""
models.py - Fixed Set Retrieval Model Implementation with Correct TopK Metrics
============================================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Dict, Any, Optional, Tuple, List
import math
import pdb

class BatchTopKAccuracy(tf.keras.metrics.Metric):
    """動作する型安全なバッチ内TopK正解率メトリック"""
    
    def __init__(self, k=1, name='top1_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = tf.constant(float(k), dtype=tf.float32)
        self.total_correct = self.add_weight(name='total_correct', initializer='zeros', dtype=tf.float32)
        self.total_count = self.add_weight(name='total_count', initializer='zeros', dtype=tf.float32)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Return文なしの安全な実装
        """
        similarities = tf.cast(y_pred, tf.float32)
        batch_size = tf.cast(tf.shape(similarities)[0], tf.float32)
        
        def compute_metrics():
            """メトリック計算"""
            # 対角線要素
            diagonal = tf.linalg.diag_part(similarities)
            
            # 各行で対角線要素より大きい要素の数
            expanded_diag = tf.expand_dims(diagonal, axis=1)
            better_mask = similarities > expanded_diag
            
            # 対角線位置を除外
            batch_size_int = tf.cast(batch_size, tf.int32)
            eye_mask = tf.eye(batch_size_int, dtype=tf.bool)
            better_mask = tf.logical_and(better_mask, tf.logical_not(eye_mask))
            
            # ランク計算（すべてfloat32）
            num_better = tf.reduce_sum(tf.cast(better_mask, tf.float32), axis=1)
            ranks = num_better + tf.constant(1.0, dtype=tf.float32)
            
            # Top-K判定
            k_threshold = tf.minimum(self.k, batch_size - tf.constant(1.0, dtype=tf.float32))
            k_threshold = tf.maximum(k_threshold, tf.constant(1.0, dtype=tf.float32))
            
            correct = tf.reduce_sum(tf.cast(ranks <= k_threshold, tf.float32))
            
            return correct, batch_size
        
        def skip_metrics():
            """スキップ時"""
            return tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)
        
        # 条件分岐でメトリック計算
        correct_count, valid_count = tf.cond(
            tf.greater(batch_size, tf.constant(1.0, dtype=tf.float32)),
            compute_metrics,
            skip_metrics
        )
        
        # 統計更新
        self.total_correct.assign_add(correct_count)
        self.total_count.assign_add(valid_count)
    
    def result(self):
        accuracy = tf.math.divide_no_nan(self.total_correct, self.total_count)
        return accuracy * 100.0  # パーセント変換
    
    def reset_state(self):
        self.total_correct.assign(0.0)
        self.total_count.assign(0.0)

class SetRetrievalModel(Model):
    """
    Set Retrieval Model - Fixed Implementation with Proper TopK Metrics
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
                 k_values: List[int] = None,
                 cycle_lambda: float= 0.1,
                 **kwargs):

        # Remove k_values from kwargs to avoid conflicts
        kwargs.pop('k_values', None)        
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
        self.k_values = k_values if k_values is not None else [1, 5, 10, 20]
        self.cycle_lambda = cycle_lambda
        
        # Category centers (learnable)
        self.category_centers = None
        
        # Build layers
        self._build_layers()
        
        # Build TopK metrics - 修正版
        self._build_topk_metrics()
        
        print(f"[INFO] SetRetrievalModel created (Fixed TopK):")
        print(f"  - Feature dim: {feature_dim}")
        print(f"  - Heads: {num_heads}, Layers: {num_layers}")
        print(f"  - Categories: {num_categories}")
        print(f"  - Temperature: {temperature}")
        print(f"  - TopK values: {self.k_values}")
    
    def _build_layers(self):
        """Build model layers"""
        
        # Input projection
        self.input_projection = layers.Dense(
            self.hidden_dim,
            activation='relu',
            name='input_projection'
        )
        
        # Cross-attention layers
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
        
        # Output projection
        self.output_projection = layers.Dense(
            self.feature_dim,
            activation=None,
            name='output_projection'
        )
        
        # Output normalization
        self.output_norm = layers.LayerNormalization(epsilon=1e-6, name='output_norm')

    def _build_topk_metrics(self):
        self.topk_metrics = {}
        
        # Training metrics
        for k in sorted(self.k_values): 
            metric = BatchTopKAccuracy(k=k, name=f'top{k}_accuracy')
            self.topk_metrics[f'top{k}_accuracy'] = metric
        
        # Validation metrics - 名前とキーを一致させる
        for k in sorted(self.k_values): 
            val_metric = BatchTopKAccuracy(k=k, name=f'val_top{k}_accuracy')  # 名前にval_を含める
            self.topk_metrics[f'val_top{k}_accuracy'] = val_metric

    @property
    def metrics(self):
        """Return list of all metrics"""
        return list(self.topk_metrics.values())
    
    def set_category_centers(self, centers: np.ndarray):
        """Set category cluster centers"""
        if centers.shape[0] != self.num_categories:
            raise ValueError(f"Expected {self.num_categories} centers, got {centers.shape[0]}")
        
        # L2 normalize centers
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
        
        # Create learnable category centers
        self.category_centers = self.add_weight(
            name='category_centers',
            shape=(self.num_categories, self.hidden_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Initialize with provided centers
        if centers.shape[1] != self.hidden_dim:
            # Project if dimensions don't match
            init_centers = np.random.normal(0, 0.02, (self.num_categories, self.hidden_dim)).astype(np.float32)
        else:
            init_centers = centers.astype(np.float32)
        
        self.category_centers.assign(init_centers)
        print(f"✅ Category centers initialized: {centers.shape}")
    
    def call(self, inputs, training=None):
        """Forward pass"""
        
        # Parse inputs
        if isinstance(inputs, dict):
            query_features = inputs['query_features']
        else:
            query_features = inputs[0]
        
        # Forward pass
        predictions = self._forward_pass(query_features, training)
        
        return predictions
    
    def _forward_pass(self, query_features, training=None):
        """Forward pass through cross-attention layers"""
        
        batch_size = tf.shape(query_features)[0]
        
        # Project query features
        query_projected = self.input_projection(query_features)
        
        # Get category centers
        if self.category_centers is None:
            raise ValueError("Category centers not set! Call set_category_centers() first.")
        
        # Expand category centers for batch
        category_queries = tf.expand_dims(self.category_centers, 0)
        category_queries = tf.tile(category_queries, [batch_size, 1, 1])
        
        # Cross-attention layers
        x = category_queries
        
        for layer_dict in self.cross_attention_layers:
            # Cross-attention: categories attend to query set
            cross_attn_out = layer_dict['cross_attention'](
                query=x,
                key=query_projected,
                value=query_projected,
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
        
        predictions = self.output_projection(x)
        predictions = tf.nn.l2_normalize(predictions, axis=-1)
        
        return self.output_norm(predictions) 

    def _compute_set_similarities_fixed(self, predictions, target_features):
        """
        セット類似度計算の修正版 - カテゴリ情報を保持
        
        Args:
            predictions: [batch, num_categories, feature_dim]
            target_features: [batch, seq_len, feature_dim]
            
        Returns:
            similarities: [batch, batch] - Set similarity matrix
        """
        batch_size = tf.shape(predictions)[0]
        
        # ターゲット表現を計算（平均）
        target_mask = tf.reduce_sum(tf.abs(target_features), axis=-1) > 0
        target_mask_expanded = tf.expand_dims(target_mask, -1)
        
        masked_targets = target_features * tf.cast(target_mask_expanded, tf.float32)
        target_sum = tf.reduce_sum(masked_targets, axis=1)
        target_count = tf.maximum(tf.reduce_sum(tf.cast(target_mask, tf.float32), axis=1, keepdims=True), 1.0)
        target_repr = target_sum / target_count  # [batch, feature_dim]
        
        similarities = tf.zeros([batch_size, batch_size])
        
        for cat in range(self.num_categories):
            cat_pred = predictions[:, cat, :]  # [B, D]
            cat_similarities = tf.matmul(cat_pred, target_repr, transpose_b=True)
            similarities = tf.maximum(similarities, cat_similarities)
        
        return similarities
    
    def train_step(self, data):
        """Custom training step with fixed TopK metrics"""
        batch_data = data

        with tf.GradientTape() as tape:
            query_features = batch_data['query_features']
            query_categories = batch_data['query_categories']
            target_features = batch_data['target_features']
            target_categories = batch_data['target_categories']

            predictions_Y = self(batch_data, training=True) #Y'

            # Compute contrastive loss
            loss_query_to_target  = contrastive_loss_per_item(
                predictions_Y,
                target_features,
                target_categories,
                temperature=self.temperature
            )

            if self.use_cycle_loss:
                cycle_batch_data = {
                    'query_features': predictions_Y,
                    'query_categories': target_categories,
                    'target_features': query_features,
                    'target_categories': query_categories
                }
                predictions_X = self(cycle_batch_data, training=True) #X'

                loss_target_to_query = contrastive_loss_per_item(
                    predictions_X,
                    query_features,
                    query_categories,
                    temperature=self.temperature
                )

                total_loss = loss_query_to_target + self.cycle_lambda * loss_target_to_query

            else:
                loss_target_to_query = tf.constant(0.0)
                total_loss = loss_query_to_target

        # Apply gradients
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Compute set similarities for TopK metrics - 修正版を使用
        similarities = self._compute_set_similarities_fixed(predictions_Y, target_features)
        
        # Update TopK metrics
        dummy_y_true = tf.zeros_like(similarities)
        for k in sorted(self.k_values):
            metric_name = f'top{k}_accuracy'
            if metric_name in self.topk_metrics:
                self.topk_metrics[metric_name].update_state(dummy_y_true, similarities)
        
        # Compile results
        results = {
            'loss': total_loss,
            "X->Y' Loss": loss_query_to_target, 
            "Y'->X' Loss": loss_target_to_query                 
        }
        for k in sorted(self.k_values):
            metric_name = f'top{k}_accuracy'
            if metric_name in self.topk_metrics:
                results[metric_name] = self.topk_metrics[metric_name].result()
        
        return results



    def test_step(self, data):
        """Test step with true cycle consistency loss using predictions as input"""
        batch_data = data
        
        query_features = batch_data['query_features']
        target_features = batch_data['target_features']
        query_categories = batch_data['query_categories']
        target_categories = batch_data['target_categories']

        # Forward direction: Query → Target prediction
        predictions_Y = self(batch_data, training=False)
        loss_query_to_target = contrastive_loss_per_item(
            predictions_Y,
            target_features,
            target_categories,
            temperature=self.temperature
        )

        # Cycle consistency (if enabled)
        if self.use_cycle_loss:
            cycle_batch_data = {
                'query_features': predictions_Y,
                'query_categories': target_categories,  # train_stepと同じ
                'target_features': query_features,
                'target_categories': query_categories
            }
            predictions_X = self(cycle_batch_data, training=False) #X'

            # ✅ 正しい：cycle_lossではなく、loss_target_to_queryを計算
            loss_target_to_query = contrastive_loss_per_item(
                predictions_X,
                query_features,
                query_categories,
                temperature=self.temperature
            )

            total_loss = loss_query_to_target + self.cycle_lambda * loss_target_to_query
        else:
            loss_target_to_query = tf.constant(0.0)
            total_loss = loss_query_to_target
        
        similarities = self._compute_set_similarities_fixed(predictions_Y, target_features)
        
        dummy_y_true = tf.zeros_like(similarities)
        for k in sorted(self.k_values):
            val_metric_name = f'val_top{k}_accuracy'
            if val_metric_name in self.topk_metrics:
                self.topk_metrics[val_metric_name].update_state(dummy_y_true, similarities)
        
        results = {
            'loss': total_loss,
            "X->Y' Loss": loss_query_to_target, 
            "Y'->X' Loss": loss_target_to_query                 
        }
        for k in sorted(self.k_values):
            val_metric_name = f'val_top{k}_accuracy'
            if val_metric_name in self.topk_metrics:
                results[f'top{k}_accuracy'] = self.topk_metrics[val_metric_name].result()
        
        return results


def contrastive_loss_per_item(predictions, target_features, target_categories, temperature=1.0):
    """
    Contrastive Loss - Compares predictions to individual items
    
    Args:
        predictions: [B, C, D] - Predicted features for each category
        target_features: [B, S, D] - Ground truth item features
        target_categories: [B, S] - Ground truth item categories
        temperature: Temperature for softmax
    
    Returns:
        Contrastive loss
    """
    # Get dimensions
    B, S, D = tf.shape(target_features)[0], tf.shape(target_features)[1], tf.shape(target_features)[2]
    C = tf.shape(predictions)[1]
    
    # Flatten target data
    flat_target_features = tf.reshape(target_features, [B * S, D])
    flat_target_categories = tf.reshape(target_categories, [B * S])
    
    # Valid target mask (exclude padding)
    valid_target_mask = flat_target_categories > 0
    
    # Compute all similarities [B, C, B*S]
    logits = tf.einsum('bcd,sd->bcs', predictions, flat_target_features) / temperature
    
    # Mask invalid targets
    logits = logits * tf.cast(tf.reshape(valid_target_mask, [1, 1, B*S]), tf.float32) + \
             tf.cast(tf.reshape(~valid_target_mask, [1, 1, B*S]), tf.float32) * (-1e9)
    
    # Create positive labels
    # Category match
    p_cats = tf.reshape(tf.range(1, C + 1, dtype=tf.int32), [1, C, 1])
    t_cats = tf.reshape(flat_target_categories, [1, 1, B*S])
    category_match = tf.equal(p_cats, t_cats)
    
    # Instance match (same batch item)
    b_indices = tf.reshape(tf.range(B), [B, 1, 1])
    t_indices = tf.reshape(tf.repeat(tf.range(B), S), [1, 1, B*S])
    instance_match = tf.equal(b_indices, t_indices)
    
    # Positive labels
    labels = tf.cast(tf.logical_and(category_match, instance_match), tf.float32)
    
    # Mask for valid predictions (those with positive examples)
    has_positives_mask = tf.reduce_sum(labels, axis=-1) > 0
    
    # Normalize labels
    labels_normalized = tf.math.divide_no_nan(labels, tf.reduce_sum(labels, axis=-1, keepdims=True))
    
    # Compute loss
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_normalized, logits=logits)
    
    # Apply mask and compute mean
    masked_loss = loss * tf.cast(has_positives_mask, tf.float32)
    
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(tf.cast(has_positives_mask, tf.float32))



def debug_similarities(model, sample_batch):
    """
    デバッグ用：類似度行列を詳細に分析（Eager execution対応）
    
    Args:
        model: 訓練されたモデル
        sample_batch: サンプルバッチ
    
    Returns:
        分析結果の辞書
    """
    # Eager executionを確実に有効にする
    original_eager = tf.executing_eagerly()
    if not original_eager:
        tf.config.run_functions_eagerly(True)
    
    try:
        predictions = model(sample_batch, training=False)
        target_features = sample_batch['target_features']
        
        # 類似度行列を計算
        similarities = model._compute_set_similarities_fixed(predictions, target_features)
        
        # NumPy配列に変換して分析
        sim_np = similarities.numpy()
        pred_np = predictions.numpy()
        target_np = target_features.numpy()
        
        results = {
            'batch_size': sim_np.shape[0],
            'similarities_shape': sim_np.shape,
            'similarities_matrix': sim_np,
            'diagonal_values': np.diag(sim_np),
            'max_per_row': np.max(sim_np, axis=1),
            'argmax_per_row': np.argmax(sim_np, axis=1),
            'diagonal_ranks': [],
            'mean_similarity': np.mean(sim_np),
            'std_similarity': np.std(sim_np),
            'predictions_shape': pred_np.shape,
            'predictions_mean': np.mean(pred_np),
            'predictions_std': np.std(pred_np),
            'target_shape': target_np.shape,
            'target_mean': np.mean(target_np),
            'target_std': np.std(target_np),
        }
        
        # 各行で対角線要素のランクを計算
        for i in range(sim_np.shape[0]):
            row = sim_np[i]
            diagonal_val = row[i]
            rank = np.sum(row > diagonal_val) + 1
            results['diagonal_ranks'].append(rank)
        
        # 手動TopK計算
        mask = 1.0 - np.eye(sim_np.shape[0])
        masked_sim = sim_np * mask + np.eye(sim_np.shape[0]) * (-1e9)
        
        if sim_np.shape[0] > 1:
            top_indices = np.argsort(masked_sim, axis=1)[:, ::-1]
            correct_indices = np.arange(sim_np.shape[0])
            
            top1_acc = np.mean(top_indices[:, 0] == correct_indices) if top_indices.shape[1] > 0 else 0
            top5_acc = np.mean([correct_indices[i] in top_indices[i, :5] for i in range(len(correct_indices))]) if top_indices.shape[1] >= 5 else 0
            
            results['manual_top1_accuracy'] = top1_acc
            results['manual_top5_accuracy'] = top5_acc
            results['top_indices'] = top_indices
        else:
            results['manual_top1_accuracy'] = 0.0
            results['manual_top5_accuracy'] = 0.0
        
        return results
        
    except Exception as e:
        print(f"Error in debug_similarities: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}
    
    finally:
        # 元のeager execution設定を復元
        if not original_eager:
            tf.config.run_functions_eagerly(False)


# class DebugCallback(tf.keras.callbacks.Callback):
#     """デバッグ用コールバック"""
    
#     def __init__(self, sample_batch):
#         super().__init__()
#         self.sample_batch = sample_batch
#         self.debug_run = False
    
#     def on_epoch_end(self, epoch, logs=None):
#         if epoch == 0 and not self.debug_run:  # 最初のエポック後にのみ実行
#             print("\n" + "="*50)
#             print("🔍 DEBUGGING SIMILARITIES")
#             print("="*50)
            
#             debug_results = debug_similarities(self.model, self.sample_batch)
            
#             if 'error' in debug_results:
#                 print(f"❌ Debug failed: {debug_results['error']}")
#                 return
            
#             print(f"✅ Batch size: {debug_results['batch_size']}")
#             print(f"✅ Similarities shape: {debug_results['similarities_shape']}")
#             print(f"✅ Similarities matrix:")
#             print(debug_results['similarities_matrix'])
#             print(f"✅ Diagonal values: {debug_results['diagonal_values']}")
#             print(f"✅ Max per row: {debug_results['max_per_row']}")
#             print(f"✅ Argmax per row: {debug_results['argmax_per_row']}")
#             print(f"✅ Diagonal ranks: {debug_results['diagonal_ranks']}")
#             print(f"✅ Mean similarity: {debug_results['mean_similarity']:.4f}")
#             print(f"✅ Std similarity: {debug_results['std_similarity']:.4f}")
#             print(f"✅ Manual Top1 accuracy: {debug_results['manual_top1_accuracy']:.4f}")
#             print(f"✅ Manual Top5 accuracy: {debug_results['manual_top5_accuracy']:.4f}")
            
#             # 問題診断
#             if debug_results['batch_size'] <= 1:
#                 print("❌ PROBLEM: Batch size too small for TopK calculation!")
#             elif np.max(debug_results['diagonal_ranks']) > 1:
#                 print("❌ PROBLEM: Diagonal elements are not the maximum in their rows!")
#                 print("   This means predictions are not learning correctly.")
#             elif debug_results['std_similarity'] < 0.01:
#                 print("❌ PROBLEM: All similarities are too similar!")
#                 print("   This means feature representations are not distinctive.")
#             else:
#                 print("✅ Similarities look reasonable!")
            
#             print("="*50)
#             self.debug_run = True


# 追加の分析関数
def analyze_training_progress(model, train_batch, epoch):
    """
    学習の進捗を分析する関数
    
    Args:
        model: 学習中のモデル
        train_batch: 訓練バッチサンプル
        epoch: 現在のエポック
    
    Returns:
        分析結果
    """
    print(f"\n=== Epoch {epoch} Training Analysis ===")
    
    # 予測を取得
    predictions = model(train_batch, training=False)
    target_features = train_batch['target_features']
    
    # 特徴量の統計
    pred_np = predictions.numpy()
    target_np = target_features.numpy()
    
    print(f"Predictions stats:")
    print(f"  Mean: {np.mean(pred_np):.4f}, Std: {np.std(pred_np):.4f}")
    print(f"  Min: {np.min(pred_np):.4f}, Max: {np.max(pred_np):.4f}")
    
    print(f"Target stats:")
    print(f"  Mean: {np.mean(target_np):.4f}, Std: {np.std(target_np):.4f}")
    print(f"  Min: {np.min(target_np):.4f}, Max: {np.max(target_np):.4f}")
    
    # 類似度分析
    similarities = model._compute_set_similarities_fixed(predictions, target_features)
    sim_np = similarities.numpy()
    
    print(f"Similarities stats:")
    print(f"  Mean: {np.mean(sim_np):.4f}, Std: {np.std(sim_np):.4f}")
    print(f"  Diagonal mean: {np.mean(np.diag(sim_np)):.4f}")
    print(f"  Off-diagonal mean: {np.mean(sim_np - np.diag(np.diag(sim_np))):.4f}")
    
    # 各行での対角線要素のランク
    ranks = []
    for i in range(sim_np.shape[0]):
        rank = np.sum(sim_np[i] > sim_np[i, i]) + 1
        ranks.append(rank)
    
    print(f"Diagonal ranks: {ranks}")
    print(f"Average rank: {np.mean(ranks):.2f}")
    
    return {
        'predictions_std': np.std(pred_np),
        'similarities_std': np.std(sim_np),
        'average_rank': np.mean(ranks),
        'diagonal_mean': np.mean(np.diag(sim_np))
    }


class ProgressTrackingCallback(tf.keras.callbacks.Callback):
    """学習進捗を追跡するコールバック"""
    
    def __init__(self, sample_batch, track_every=10):
        super().__init__()
        self.sample_batch = sample_batch
        self.track_every = track_every
        self.progress_log = []
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.track_every == 0:
            progress = analyze_training_progress(self.model, self.sample_batch, epoch)
            self.progress_log.append({
                'epoch': epoch,
                'logs': logs,
                'analysis': progress
            })
            
            # 学習が停滞していないかチェック
            if len(self.progress_log) >= 2:
                prev_rank = self.progress_log[-2]['analysis']['average_rank']
                curr_rank = self.progress_log[-1]['analysis']['average_rank']
                
                if curr_rank > prev_rank * 0.9:  # ランクが改善していない
                    print(f"⚠️  Warning: Average rank not improving ({prev_rank:.2f} → {curr_rank:.2f})")
                else:
                    print(f"✅ Rank improving: {prev_rank:.2f} → {curr_rank:.2f}")


def evaluate_with_gallery(model, test_dataset, all_test_items, k_values=[1, 5, 10, 20]):
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
    
    print("🔍 Evaluating with full test gallery...")
    
    for batch_data, _ in test_dataset:
        # Get predictions
        predictions = model(batch_data, training=False)
        
        # Extract target data
        target_features = batch_data['target_features']
        
        batch_size = tf.shape(predictions)[0]
        
        for i in range(batch_size):
            # Get target representation
            target_items = target_features[i]
            target_mask = tf.reduce_sum(tf.abs(target_items), axis=-1) > 0
            valid_targets = tf.boolean_mask(target_items, target_mask)
            
            if tf.shape(valid_targets)[0] == 0:
                continue
                
            target_repr = tf.reduce_mean(valid_targets, axis=0)
            target_repr = tf.nn.l2_normalize(target_repr, axis=-1)
            
            # Compute similarities with gallery
            gallery_similarities = tf.linalg.matmul(
                tf.expand_dims(target_repr, 0), 
                all_test_items, 
                transpose_b=True
            )
            gallery_similarities = tf.squeeze(gallery_similarities, 0)
            
            # Find rank
            target_sim = 1.0  # Perfect similarity with itself
            better_count = tf.reduce_sum(tf.cast(gallery_similarities > target_sim, tf.float32))
            rank = better_count + 1
            
            all_ranks.append(rank.numpy())
    
    # Calculate metrics
    all_ranks = np.array(all_ranks)
    
    results = {}
    for k in sorted(self.k_values):
        top_k_acc = np.mean(all_ranks <= k)
        results[f'top_{k}_acc'] = top_k_acc
    
    results['mrr'] = np.mean(1.0 / all_ranks)
    results['mean_rank'] = np.mean(all_ranks)
    results['median_rank'] = np.median(all_ranks)
    
    return results