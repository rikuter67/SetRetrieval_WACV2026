# models.py - GPU Strategy Fixed Version
import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization

# Environment setup - remove CUDA_VISIBLE_DEVICES to avoid conflicts
tf.random.set_seed(42)
os.environ.update({
    'TF_CPP_MIN_LOG_LEVEL': '2',
    'PYTHONWARNINGS': 'ignore',
})

# Configure GPU strategy at module level
def setup_gpu_strategy():
    """Setup GPU strategy properly"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Use the first available GPU
            strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
            print(f"✅ GPU Strategy initialized: {gpus[0]}")
            return strategy, True
        except Exception as e:
            print(f"⚠️ GPU Strategy setup failed: {e}")
            return tf.distribute.get_strategy(), False
    else:
        print("❌ No GPU found, using CPU")
        return tf.distribute.get_strategy(), False

# Global strategy setup
STRATEGY, GPU_AVAILABLE = setup_gpu_strategy()

################################################################################
# 1. Top-K accuracy and ranking metrics
################################################################################
@tf.function(experimental_relax_shapes=True)
def compute_topk_metrics(predicted_vectors, ground_truth_vectors, category_labels, negative_samples=None, k_values=[1, 5, 10]):
    """
    Top-K精度とランキングメトリック計算
    """
    batch_size = tf.shape(predicted_vectors)[0]
    
    # Handle different input shapes
    if len(predicted_vectors.shape) == 3 and predicted_vectors.shape[1] >= 7:
        # (B, C, D) case - select appropriate category predictions
        max_cat_idx = tf.shape(predicted_vectors)[1] - 1
        cat_idx = tf.clip_by_value(category_labels - 1, 0, max_cat_idx)
        batch_indices = tf.range(batch_size)[:, tf.newaxis]
        pred_selected = tf.gather_nd(predicted_vectors, 
                                   tf.stack([tf.broadcast_to(batch_indices, tf.shape(cat_idx)),
                                           cat_idx], axis=-1))
    else:
        # (B, N, D) case
        pred_selected = predicted_vectors
    
    # Normalize vectors
    pred_norm = tf.nn.l2_normalize(pred_selected + 1e-8, axis=-1)
    gt_norm = tf.nn.l2_normalize(ground_truth_vectors + 1e-8, axis=-1)
    
    # Calculate target similarities
    target_similarities = tf.reduce_sum(pred_norm * gt_norm, axis=-1)  # (B, N)
    
    # Create gallery for ranking
    if negative_samples is not None and tf.reduce_sum(tf.abs(negative_samples)) > 0:
        # Use negative samples as gallery
        neg_norm = tf.nn.l2_normalize(negative_samples + 1e-8, axis=-1)  # (B, N, neg_num, D)
        gallery_similarities = tf.reduce_sum(
            tf.expand_dims(pred_norm, axis=2) * neg_norm, axis=-1
        )  # (B, N, neg_num)
        
        # Combine target and gallery similarities
        all_similarities = tf.concat([
            tf.expand_dims(target_similarities, axis=-1),  # (B, N, 1)
            gallery_similarities  # (B, N, neg_num)
        ], axis=-1)  # (B, N, neg_num+1)
    else:
        # Use batch items as gallery (exclude self)
        batch_similarities = tf.matmul(pred_norm, gt_norm, transpose_b=True)  # (B, N, N)
        
        # Create mask to exclude self-similarities
        diag_mask = tf.eye(tf.shape(batch_similarities)[-1], dtype=tf.bool)
        diag_mask = tf.expand_dims(diag_mask, 0)
        diag_mask = tf.tile(diag_mask, [batch_size, 1, 1])
        
        # Set self-similarities to very low value
        batch_similarities = tf.where(
            diag_mask, 
            tf.fill(tf.shape(batch_similarities), -999.0),
            batch_similarities
        )
        
        # Target similarities are the diagonal elements (before masking)
        target_sim_diag = tf.linalg.diag_part(tf.matmul(pred_norm, gt_norm, transpose_b=True))  # (B, N)
        
        # Combine target and batch similarities
        all_similarities = tf.concat([
            tf.expand_dims(target_sim_diag, axis=-1),  # (B, N, 1)
            batch_similarities  # (B, N, N)
        ], axis=-1)  # (B, N, N+1)
        
        # Update target similarities
        target_similarities = target_sim_diag
    
    # Calculate ranks
    target_sim_expanded = tf.expand_dims(target_similarities, axis=-1)
    higher_count = tf.reduce_sum(
        tf.cast(all_similarities > target_sim_expanded, tf.float32), 
        axis=-1
    )  # (B, N)
    
    ranks = higher_count + 1.0  # 1-based ranking
    
    # Calculate percentiles
    total_candidates = tf.cast(tf.shape(all_similarities)[-1], tf.float32)
    percentiles = (ranks / total_candidates) * 100.0
    
    # Calculate Top-K accuracies
    valid_mask = tf.cast(tf.reduce_sum(tf.abs(ground_truth_vectors), axis=-1) > 0, tf.float32)
    
    results = {}
    for k in k_values:
        top_k_hits = tf.cast(ranks <= float(k), tf.float32) * valid_mask
        valid_count = tf.maximum(tf.reduce_sum(valid_mask), 1.0)
        top_k_acc = tf.reduce_sum(top_k_hits) / valid_count
        results[f'top_{k}_acc'] = top_k_acc
    
    # Calculate mean percentile
    masked_percentiles = percentiles * valid_mask
    mean_percentile = tf.reduce_sum(masked_percentiles) / tf.maximum(tf.reduce_sum(valid_mask), 1.0)
    results['mean_percentile'] = mean_percentile
    
    return results

################################################################################
# 2. Loss functions
################################################################################
@tf.function(experimental_relax_shapes=True)
def compute_mixed_loss(
    category_predictions: tf.Tensor,   # (B, C, D)
    ground_truth_vectors: tf.Tensor,   # (B, N, D)
    category_labels: tf.Tensor,        # (B, N) 1-based IDs
    cluster_centers: tf.Tensor,        # (C, D)
    margin: float = 1.0,
    alpha: float = 0.5
) -> tf.Tensor:
    """
    Mixed loss with cosine similarity and L2 loss
    """
    batch_size = tf.shape(category_predictions)[0]
    num_items = tf.shape(ground_truth_vectors)[1]
    max_cat_idx = tf.shape(cluster_centers)[0] - 1
    
    # 0-based category indices with safety check
    cats_0based = tf.clip_by_value(category_labels - 1, 0, max_cat_idx)
    
    # Valid mask
    valid_mask = category_labels > 0
    
    # Batch indices
    batch_indices = tf.range(batch_size)[:, tf.newaxis]
    batch_indices = tf.broadcast_to(batch_indices, [batch_size, num_items])
    
    # Select predicted vectors
    gather_indices = tf.stack([batch_indices, cats_0based], axis=-1)
    pred_selected = tf.gather_nd(category_predictions, gather_indices)
    
    # Get category centers
    centers = tf.gather(cluster_centers, cats_0based)
    
    # Compute residuals
    pred_res = pred_selected - centers
    gt_res = ground_truth_vectors - centers
    
    # L2 normalize with stability
    pred_norm = tf.nn.l2_normalize(pred_res + 1e-8, axis=-1)
    gt_norm = tf.nn.l2_normalize(gt_res + 1e-8, axis=-1)
    
    # Cosine similarity
    cos_pos = tf.reduce_sum(pred_norm * gt_norm, axis=-1)
    
    # Negative similarity (batch mean)
    gt_mean = tf.reduce_mean(gt_norm, axis=1, keepdims=True)
    cos_neg = tf.reduce_sum(gt_mean * pred_norm, axis=-1)
    
    # Losses
    cos_loss = tf.nn.relu(margin + cos_neg - cos_pos)
    l2_loss = tf.norm(pred_res - gt_res, axis=-1)
    
    # Mix losses
    mix_loss = alpha * cos_loss + (1.0 - alpha) * l2_loss
    
    # Apply valid mask
    valid_mask_float = tf.cast(valid_mask, tf.float32)
    masked_loss = mix_loss * valid_mask_float
    
    # Calculate mean with proper normalization
    valid_counts = tf.maximum(tf.reduce_sum(valid_mask_float, axis=1), 1.0)
    batch_losses = tf.reduce_sum(masked_loss, axis=1) / valid_counts
    
    return tf.reduce_mean(batch_losses)

################################################################################
# 3. MLP and Pivot layers
################################################################################
class MLP(tf.keras.layers.Layer):
    """Multi-layer perceptron"""
    def __init__(self, hidden_dim=128, out_dim=64, dropout_rate=0.1, 
                 name='mlp', **kwargs):
        super().__init__(name=name, **kwargs)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        
        self.dense1 = Dense(hidden_dim, activation=None, name=f"{name}_dense1")
        self.dense2 = Dense(out_dim, activation=None, name=f"{name}_dense2")
        self.dropout = Dropout(dropout_rate, name=f"{name}_dropout")
        self.norm = LayerNormalization(name=f"{name}_norm")
    
    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.norm(x)
        x = tf.nn.gelu(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

class PivotLayer(tf.keras.layers.Layer):
    """Pivot layer with self and cross attention"""
    def __init__(self, dim, num_heads=8, ff_dim=512, dropout_rate=0.1, 
                 layer_name="pivot", **kwargs):
        super().__init__(name=layer_name, **kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # Multi-head attention layers
        self.self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=dim // num_heads,
            name=f"{layer_name}_self_attn"
        )
        self.cross_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=dim // num_heads,
            name=f"{layer_name}_cross_attn"
        )
        
        # Normalization layers
        self.norm1 = LayerNormalization(name=f"{layer_name}_norm1")
        self.norm2 = LayerNormalization(name=f"{layer_name}_norm2")
        self.norm3 = LayerNormalization(name=f"{layer_name}_norm3")
        
        # Dropout layers
        self.dropout1 = Dropout(dropout_rate, name=f"{layer_name}_dropout1")
        self.dropout2 = Dropout(dropout_rate, name=f"{layer_name}_dropout2")
        self.dropout3 = Dropout(dropout_rate, name=f"{layer_name}_dropout3")
        
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation=tf.nn.gelu, name=f"{layer_name}_ffn1"),
            Dropout(dropout_rate, name=f"{layer_name}_ffn_dropout"),
            Dense(dim, name=f"{layer_name}_ffn2")
        ], name=f"{layer_name}_ffn")
    
    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=False):
        input_set, cluster_center = inputs
        
        # Self-attention
        attn1_out = self.self_attn(
            query=input_set, key=input_set, value=input_set, training=training
        )
        attn1_out = self.dropout1(attn1_out, training=training)
        out1 = self.norm1(input_set + attn1_out)
        
        # Cross-attention
        attn2_out = self.cross_attn(
            query=cluster_center, key=out1, value=out1, training=training
        )
        attn2_out = self.dropout2(attn2_out, training=training)
        out2 = self.norm2(cluster_center + attn2_out)
        
        # Feed-forward
        ffn_out = self.ffn(out2, training=training)
        ffn_out = self.dropout3(ffn_out, training=training)
        final_out = self.norm3(out2 + ffn_out)
        
        return final_out

################################################################################
# 4. SetRetrievalModel with Strategy-aware training
################################################################################
class SetRetrievalModel(tf.keras.Model):
    """Set retrieval model with proper GPU strategy handling"""
    
    def __init__(
        self, 
        dim=512, 
        num_layers=6, 
        num_heads=8, 
        ff_dim=512,
        cycle_lambda=0.2, 
        use_cycle_loss=False, 
        use_CLNeg_loss=False, 
        use_center_base=False,
        num_categories=10,
        dropout_rate=0.1,
        name="SetRetrievalModel",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        # Model configuration
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.cycle_lambda = cycle_lambda
        self.use_cycle_loss = use_cycle_loss
        self.use_CLNeg_loss = use_CLNeg_loss
        self.use_center_base = use_center_base
        self.num_categories = num_categories
        self.dropout_rate = dropout_rate
        
        # Build layers
        self.pivot_layers = [
            PivotLayer(
                dim=dim, 
                num_heads=num_heads, 
                ff_dim=ff_dim, 
                dropout_rate=dropout_rate,
                layer_name=f"pivot_layer_{i}"
            ) for i in range(num_layers)
        ]
        
        # Final layers
        self.final_norm = LayerNormalization(name="final_norm")
        self.final_dense = Dense(dim, name="final_dense")
        
        # MLP layers for various operations
        self.feature_mlp = MLP(dim, dim, dropout_rate, "feature_mlp")
        self.output_mlp = MLP(dim, dim, dropout_rate, "output_mlp")
        
        # Cluster centers
        self._cluster_centers = None
        
        # Internal state
        self._predicted_outputs = None
        self._ground_truth_targets = None
        self._category_buffer = None
        
        # Top-K accuracy metrics
        self.train_top1_acc = tf.keras.metrics.Mean(name="train_top1_acc")
        self.train_top5_acc = tf.keras.metrics.Mean(name="train_top5_acc")
        self.train_percentile = tf.keras.metrics.Mean(name="train_percentile")
        self.val_top1_acc = tf.keras.metrics.Mean(name="val_top1_acc")
        self.val_top5_acc = tf.keras.metrics.Mean(name="val_top5_acc")
        self.val_percentile = tf.keras.metrics.Mean(name="val_percentile")
        
        # Initialize metrics properly
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize metrics with dummy values"""
        dummy_acc = 0.1  # Low accuracy initially
        dummy_percentile = 90.0  # High percentile initially (bad)
        
        for metric in [self.train_top1_acc, self.train_top5_acc, self.val_top1_acc, self.val_top5_acc]:
            metric.update_state(dummy_acc)
            metric.reset_state()
        
        for metric in [self.train_percentile, self.val_percentile]:
            metric.update_state(dummy_percentile)
            metric.reset_state()
    
    def count_parameters_detailed(self):
        """パラメータカウンティング"""
        param_breakdown = {
            'attention_params': 0,
            'ffn_params': 0, 
            'norm_params': 0,
            'dense_params': 0,
            'other_params': 0
        }
        
        total_params = 0
        
        for weight in self.trainable_weights:
            weight_name = weight.name.lower()
            weight_params = tf.size(weight).numpy()
            total_params += weight_params
            
            if any(keyword in weight_name for keyword in ['multi_head_attention', 'self_attn', 'cross_attn']):
                param_breakdown['attention_params'] += weight_params
            elif 'ffn' in weight_name:
                param_breakdown['ffn_params'] += weight_params
            elif any(keyword in weight_name for keyword in ['layer_normalization', 'norm']):
                param_breakdown['norm_params'] += weight_params
            elif 'dense' in weight_name:
                param_breakdown['dense_params'] += weight_params
            else:
                param_breakdown['other_params'] += weight_params
        
        return {'total_params': total_params, 'breakdown': param_breakdown}

    def display_parameter_summary(self):
        """パラメータサマリー表示"""
        try:
            param_info = self.count_parameters_detailed()
            breakdown = param_info['breakdown']
            total = param_info['total_params']
            
            print(f"\n{'='*60}")
            print("MODEL PARAMETER SUMMARY")
            print(f"{'='*60}")
            print(f"Config: {self.dim}D, {self.num_layers}L, {self.num_heads}H, {self.num_categories}C")
            print(f"Flags: CB={self.use_center_base}, Cycle={self.use_cycle_loss}, CLNeg={self.use_CLNeg_loss}")
            print(f"{'-'*60}")
            
            category_names = {
                'attention_params': 'Attention',
                'ffn_params': 'Feed-Forward', 
                'dense_params': 'Dense Layers',
                'norm_params': 'Normalization',
                'other_params': 'Other'
            }
            
            for key, count in breakdown.items():
                if count > 0:
                    percentage = (count / total) * 100
                    name = category_names.get(key, key.replace('_', ' ').title())
                    print(f"{name:<15}: {count:>8,} ({percentage:>5.1f}%)")
            
            print(f"{'-'*60}")
            print(f"{'Total':<15}: {total:>8,}")
            memory_mb = (total * 4) / (1024 * 1024)
            print(f"{'Memory Est.':<15}: {memory_mb:>8.1f} MB")
            print(f"{'='*60}")
            
            return param_info
            
        except Exception as e:
            print(f"[ERROR] Parameter summary failed: {e}")
            return {'total_params': 0, 'breakdown': {}}
    
    def set_cluster_center(self, cluster_centers):
        """Set cluster centers with validation"""
        if cluster_centers is None:
            print(f"[WARN] No cluster centers provided, using random initialization")
            self._cluster_centers = tf.Variable(
                tf.random.normal((self.num_categories, self.dim), stddev=0.1),
                trainable=False,
                name="cluster_centers"
            )
            return
        
        # Convert to proper tensor format
        if isinstance(cluster_centers, dict):
            center_array = np.zeros((self.num_categories, self.dim), dtype=np.float32)
            for cat_id, center in cluster_centers.items():
                if 1 <= cat_id <= self.num_categories:
                    center_array[cat_id - 1] = np.array(center, dtype=np.float32)
        else:
            center_array = np.array(cluster_centers, dtype=np.float32)
            if center_array.shape[0] != self.num_categories:
                print(f"[WARN] Cluster center count mismatch: expected {self.num_categories}, got {center_array.shape[0]}")
                new_array = np.zeros((self.num_categories, center_array.shape[1]), dtype=np.float32)
                copy_count = min(self.num_categories, center_array.shape[0])
                new_array[:copy_count] = center_array[:copy_count]
                center_array = new_array
        
        self._cluster_centers = tf.constant(center_array, dtype=tf.float32)
        print(f"[INFO] Set cluster centers: {center_array.shape}")
    
    def get_cluster_center(self):
        """Get cluster centers"""
        if self._cluster_centers is None:
            raise ValueError("Cluster centers not set. Call set_cluster_center() first.")
        return self._cluster_centers
    
    @tf.function(experimental_relax_shapes=True)
    def forward_pass(self, input_set, training=False):
        """Forward pass through the model"""
        batch_size = tf.shape(input_set)[0]
        
        # Get cluster centers and expand for batch
        centers = self.get_cluster_center()  # (num_categories, dim)
        centers = tf.expand_dims(centers, 0)  # (1, num_categories, dim)
        centers = tf.tile(centers, [batch_size, 1, 1])  # (B, num_categories, dim)
        
        # Process through pivot layers
        features = centers
        for layer in self.pivot_layers:
            features = layer((input_set, features), training=training)
        
        # Final processing
        features = self.final_norm(features)
        output = self.final_dense(features)
        
        return output  # (B, num_categories, dim)
    
    def call(self, inputs, training=False):
        """Model call method"""
        (X_concat, _, _, catQ_batch, catP_batch, _, _, _) = inputs
        
        batch_size = tf.shape(X_concat)[0] // 2
        input_features = X_concat[:batch_size]  # First half
        target_features = X_concat[batch_size:]  # Second half
        
        # Forward pass
        predictions = self.forward_pass(input_features, training=training)
        
        # Store for loss calculation
        self._predicted_outputs = predictions
        self._ground_truth_targets = target_features
        self._category_buffer = catP_batch
        
        return predictions
    
    @tf.function(experimental_relax_shapes=True)
    def train_step(self, data):
        """Training step without device forcing - let strategy handle it"""
        (batch_data, _) = data
        
        with tf.GradientTape() as tape:
            # XY direction
            predictions_xy = self(batch_data, training=True)
            pred_xy = self._predicted_outputs
            gt_xy = self._ground_truth_targets
            cat_xy = self._category_buffer
            
            # YX direction (swap inputs)
            (X_concat, _, _, catQ_batch, catP_batch, _, _, negative_samples) = batch_data
            batch_size = tf.shape(X_concat)[0] // 2
            
            X_concat_yx = tf.concat([X_concat[batch_size:], X_concat[:batch_size]], axis=0)
            batch_data_yx = (X_concat_yx, _, _, catP_batch, catQ_batch, _, _, negative_samples)
            
            predictions_yx = self(batch_data_yx, training=True)
            pred_yx = self._predicted_outputs
            gt_yx = self._ground_truth_targets
            cat_yx = self._category_buffer
            
            # Calculate losses
            if self.use_center_base and self._cluster_centers is not None:
                loss_xy = compute_mixed_loss(pred_xy, gt_xy, cat_xy, self.get_cluster_center(), margin=1.0, alpha=0.5)
                loss_yx = compute_mixed_loss(pred_yx, gt_yx, cat_yx, self.get_cluster_center(), margin=1.0, alpha=0.5)
            else:
                loss_xy = tf.reduce_mean(tf.keras.losses.mse(gt_xy, pred_xy))
                loss_yx = tf.reduce_mean(tf.keras.losses.mse(gt_yx, pred_yx))
            
            # Cycle loss
            cycle_loss = tf.constant(0.0)
            if self.use_cycle_loss:
                cycle_loss = tf.reduce_mean(tf.abs(pred_xy - pred_yx)) * self.cycle_lambda
            
            total_loss = loss_xy + loss_yx + cycle_loss
            
            # Calculate Top-K metrics
            neg_samples_xy = negative_samples[:batch_size] if self.use_CLNeg_loss else None
            neg_samples_yx = negative_samples[batch_size:] if self.use_CLNeg_loss else None
            
            metrics_xy = compute_topk_metrics(pred_xy, gt_xy, cat_xy, neg_samples_xy, [1, 5])
            metrics_yx = compute_topk_metrics(pred_yx, gt_yx, cat_yx, neg_samples_yx, [1, 5])
            
            # Average metrics
            mean_top1 = (metrics_xy['top_1_acc'] + metrics_yx['top_1_acc']) / 2.0
            mean_top5 = (metrics_xy['top_5_acc'] + metrics_yx['top_5_acc']) / 2.0
            mean_percentile = (metrics_xy['mean_percentile'] + metrics_yx['mean_percentile']) / 2.0
        
        # Apply gradients - let strategy handle device placement
        gradients = tape.gradient(total_loss, self.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics
        self.train_top1_acc.update_state(mean_top1)
        self.train_top5_acc.update_state(mean_top5)
        self.train_percentile.update_state(mean_percentile)
        
        # Clear internal state
        self._predicted_outputs = None
        self._ground_truth_targets = None
        self._category_buffer = None
        
        return {
            "loss": total_loss,
            "train_top1_acc": self.train_top1_acc.result(),
            "train_top5_acc": self.train_top5_acc.result(),
            "train_percentile": self.train_percentile.result(),
            "loss_xy": loss_xy,
            "loss_yx": loss_yx,
            "cycle_loss": cycle_loss
        }
    
    @tf.function(experimental_relax_shapes=True)
    def test_step(self, data):
        """Test step with Top-K metrics"""
        (batch_data, _) = data
        
        predictions = self(batch_data, training=False)
        
        # Get negative samples
        (_, _, _, _, _, _, _, negative_samples) = batch_data
        neg_samples = negative_samples if self.use_CLNeg_loss else None
        
        # Calculate metrics
        metrics = compute_topk_metrics(
            self._predicted_outputs,
            self._ground_truth_targets,
            self._category_buffer,
            neg_samples,
            [1, 5]
        )
        
        # Update metrics
        self.val_top1_acc.update_state(metrics['top_1_acc'])
        self.val_top5_acc.update_state(metrics['top_5_acc'])
        self.val_percentile.update_state(metrics['mean_percentile'])
        
        # Clear internal state
        self._predicted_outputs = None
        self._ground_truth_targets = None
        self._category_buffer = None
        
        return {
            "val_top1_acc": self.val_top1_acc.result(),
            "val_top5_acc": self.val_top5_acc.result(),
            "val_percentile": self.val_percentile.result()
        }
    
    def predict_batch(self, batch_data, training=False):
        """Predict a batch for evaluation purposes"""
        try:
            predictions = self(batch_data, training=training)
            return {
                'predictions': predictions,
                'pred_outputs': self._predicted_outputs,
                'gt_targets': self._ground_truth_targets,
                'categories': self._category_buffer
            }
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return None
    
    def build_model_completely(self, input_shape):
        """Build model completely with all layers initialized"""
        # Build with forward pass
        dummy_input = tf.zeros(input_shape)
        _ = self.forward_pass(dummy_input, training=False)
        
        # Build with full call method
        batch_size = input_shape[0]
        max_items = input_shape[1]
        feature_dim = input_shape[2]
        
        # Create dummy batch data
        X_concat = tf.zeros((batch_size * 2, max_items, feature_dim))
        catQ_batch = tf.ones((batch_size, max_items), dtype=tf.int32)
        catP_batch = tf.ones((batch_size, max_items), dtype=tf.int32)
        negative_samples = tf.zeros((batch_size * 2, max_items, 1, feature_dim))
        
        dummy_batch_data = (X_concat, None, None, catQ_batch, catP_batch, None, None, negative_samples)
        _ = self(dummy_batch_data, training=False)
        
        print("✅ Model built completely with all layers")
    
    def infer_single_set(self, input_features):
        """Single set inference"""
        if len(input_features.shape) == 2:
            input_features = tf.expand_dims(input_features, 0)
        
        output = self.forward_pass(input_features, training=False)
        return output[0]  # Remove batch dimension