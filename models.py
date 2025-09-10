import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import List
import pdb
# tf.config.run_functions_eagerly(True)  # debug flag

class PerCategoryItemTopKAccuracy(tf.keras.metrics.Metric):
    """
    Per-category Top-K% accuracy for set retrieval tasks.
    Evaluates what percentage of items rank in the top K% within their category,
    providing balanced evaluation across categories regardless of size.
    """
    
    def __init__(self, k_percent, num_categories, name='per_category_topk_acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.k_percent = float(k_percent)  
        self.num_categories = num_categories
        self.total_correct = self.add_weight(name='total_correct', initializer='zeros')
        self.total_count = self.add_weight(name='total_count', initializer='zeros')

    @tf.function(jit_compile=False)
    def update_state(self, predictions, target_features, target_categories, sample_weight=None):
        """
        Args:
            predictions: (batch_size, num_categories, feature_dim) - Model predictions per category
            target_features: (batch_size, items_per_set, feature_dim) - Ground truth features
            target_categories: (batch_size, items_per_set) - Ground truth category IDs (1-indexed)
        """
        batch_size, items_per_set, feature_dim = tf.unpack(tf.shape(target_features))
        
        # Flatten and filter valid items (non-zero category, non-zero features)
        target_flat = tf.reshape(target_features, [-1, feature_dim])
        categories_flat = tf.reshape(target_categories, [-1])
        batch_ids = tf.repeat(tf.range(batch_size), items_per_set)
        
        valid_mask = tf.logical_and(
            categories_flat > 0,
            tf.reduce_sum(tf.abs(target_flat), axis=-1) > 1e-6
        )
        
        valid_features = tf.boolean_mask(target_flat, valid_mask)
        valid_categories = tf.boolean_mask(categories_flat, valid_mask)  
        valid_batch_ids = tf.boolean_mask(batch_ids, valid_mask)
        
        if tf.size(valid_features) == 0:
            return
            
        # Get corresponding prediction vectors for each valid item
        pred_indices = tf.stack([valid_batch_ids, valid_categories - 1], axis=1)
        pred_vectors = tf.gather_nd(predictions, pred_indices)
        
        # Normalize for cosine similarity
        pred_norm = tf.nn.l2_normalize(pred_vectors, axis=-1)
        target_norm = tf.nn.l2_normalize(valid_features, axis=-1)
        
        total_correct, total_items = 0.0, 0.0
        
        # Evaluate each category independently
        for cat_id in range(1, self.num_categories + 1):
            cat_mask = tf.equal(valid_categories, cat_id)
            num_items = tf.reduce_sum(tf.cast(cat_mask, tf.int32))
            
            if num_items <= 1:
                continue
                
            # Extract category-specific predictions and targets
            cat_preds = tf.boolean_mask(pred_norm, cat_mask)
            cat_targets = tf.boolean_mask(target_norm, cat_mask)
            
            # Compute similarity matrix and ranks
            sim_matrix = tf.matmul(cat_preds, cat_targets, transpose_b=True)
            diag_sims = tf.linalg.diag_part(sim_matrix)
            
            # Count items with higher similarity (excluding self-comparison)
            higher_count = tf.reduce_sum(tf.cast(
                sim_matrix > tf.expand_dims(diag_sims, 1), tf.float32
            ), axis=1) - tf.cast(tf.range(num_items), tf.float32) * 0  # Remove self from count
            
            ranks = higher_count + 1.0
            
            # Count items within top K%
            k_threshold = tf.maximum(1.0, (self.k_percent / 100.0) * tf.cast(num_items, tf.float32))
            correct_in_cat = tf.reduce_sum(tf.cast(ranks <= k_threshold, tf.float32))
            
            total_correct += correct_in_cat
            total_items += tf.cast(num_items, tf.float32)
        
        self.total_correct.assign_add(total_correct)
        self.total_count.assign_add(total_items)

    def result(self):
        return tf.math.divide_no_nan(self.total_correct, self.total_count) * 100.0

    def reset_state(self):
        self.total_correct.assign(0.0)
        self.total_count.assign(0.0)

        
# ============================================================================
# ベースモデル定義
# ============================================================================
class CrossAttentionSetRetrieval(Model):
    """Set Retrieval Model の基底クラス（共有ロジック）"""
    
    def __init__(self, feature_dim: int = 512, num_heads: int = 8, num_layers: int = 6, num_categories: int = 16, hidden_dim: int = 512, 
                 use_cycle_loss: bool = False, temperature: float = 1.0, dropout_rate: float = 0.1, k_values: List[int] = None, cycle_lambda: float = 0.1, cluster_centering: bool = False,
                 use_tpaneg: bool = False, taneg_t_gamma_init: float = 0.5,  taneg_t_gamma_final: float = 0.8,  taneg_curriculum_epochs: int = 100,  paneg_epsilon: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        
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
        self.cluster_centering = cluster_centering
        self.use_tpaneg = use_tpaneg 

        self.taneg_t_gamma_init = taneg_t_gamma_init
        self.taneg_t_gamma_final = taneg_t_gamma_final
        self.taneg_curriculum_epochs = taneg_curriculum_epochs
        self.paneg_epsilon = paneg_epsilon

        self.category_centers = None
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.xy_loss_tracker = tf.keras.metrics.Mean(name="X->Y' Loss")
        self.yx_loss_tracker = tf.keras.metrics.Mean(name="Y'->X' Loss")
        
        self._build_layers()
        self._build_topk_metrics()
    
    def _build_layers(self):
        self.input_projection = layers.Dense(self.hidden_dim, activation='relu', name='input_projection')
        self.cross_attention_layers = []
        for i in range(self.num_layers):
            layer_dict = {
                'cross_attention': layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.hidden_dim // self.num_heads, dropout=self.dropout_rate, name=f'cross_attention_{i}'),
                'self_attention': layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.hidden_dim // self.num_heads, dropout=self.dropout_rate, name=f'self_attention_{i}'),
                'norm1': layers.LayerNormalization(epsilon=1e-6, name=f'norm1_{i}'),
                'norm2': layers.LayerNormalization(epsilon=1e-6, name=f'norm2_{i}'),
                'norm3': layers.LayerNormalization(epsilon=1e-6, name=f'norm3_{i}'),
                'ffn': tf.keras.Sequential([layers.Dense(self.hidden_dim * 2, activation='gelu'), layers.Dropout(self.dropout_rate), layers.Dense(self.hidden_dim)], name=f'ffn_{i}')
            }
            self.cross_attention_layers.append(layer_dict)
        self.output_projection = layers.Dense(self.feature_dim, activation=None, name='output_projection')
        self.output_norm = layers.LayerNormalization(epsilon=1e-6, name='output_norm')

    def _build_topk_metrics(self):
        self.topk_metrics = {}
        for k in sorted(self.k_values): 
            self.topk_metrics[f'top{k}_accuracy'] = PerCategoryItemTopKAccuracy(k, self.num_categories, name=f'top{k}_accuracy')
            self.topk_metrics[f'val_top{k}_accuracy'] = PerCategoryItemTopKAccuracy(k, self.num_categories, name=f'val_top{k}_accuracy')

    @property
    def metrics(self):
        base_metrics = [self.loss_tracker, self.xy_loss_tracker, self.yx_loss_tracker]
        return base_metrics + list(self.topk_metrics.values())
        
    def set_category_centers(self, centers: np.ndarray, whitening_params=None):
        if centers.shape[0] != self.num_categories:
            raise ValueError(f"Expected {self.num_categories} centers, got {centers.shape[0]}")
        if whitening_params is not None:
            print("🔄 Applying whitening transformation to category centers...")
            centers = np.dot(centers - whitening_params['mean'], whitening_params['matrix'])
            print(f"✅ Category centers whitened: {centers.shape}")
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
        self.category_centers = self.add_weight(name='category_centers', shape=(self.num_categories, self.hidden_dim), initializer='glorot_uniform', trainable=True)
        if centers.shape[1] != self.hidden_dim:
            init_centers = np.random.normal(0, 0.02, (self.num_categories, self.hidden_dim)).astype(np.float32)
        else:
            init_centers = centers.astype(np.float32)
        self.category_centers.assign(init_centers)
        print(f"✅ Category centers initialized: {centers.shape}")

    def call(self, inputs, training=None):
        query_features = inputs['query_features']
        query_projected = self.input_projection(query_features)
        if self.category_centers is None:
            raise ValueError("Category centers not set! Call set_category_centers() first.")
        batch_size = tf.shape(query_features)[0]
        x = tf.tile(tf.expand_dims(self.category_centers, 0), [batch_size, 1, 1])
        for layer_dict in self.cross_attention_layers:
            cross_attn_out = layer_dict['cross_attention'](query=x, key=query_projected, value=query_projected, training=training)
            x = layer_dict['norm1'](x + cross_attn_out, training=training)
            self_attn_out = layer_dict['self_attention'](query=x, key=x, value=x, training=training)
            x = layer_dict['norm2'](x + self_attn_out, training=training)
            ffn_out = layer_dict['ffn'](x, training=training)
            x = layer_dict['norm3'](x + ffn_out, training=training)
        predictions = self.output_projection(x)

        # NOTE: self.output_norm is not used here. This is a potential bug but not the cause of the stagnation.
        predictions = self.output_norm(predictions) 

        if self.cluster_centering:
            cluster_centers_normalized = tf.nn.l2_normalize(self.category_centers, axis=-1)
            predictions = predictions - tf.expand_dims(cluster_centers_normalized, 0)
            predictions = tf.nn.l2_normalize(predictions, axis=-1)
        return predictions

    def infer_single_set(self, query_features):
        if len(query_features.shape) == 2:
            query_features = tf.expand_dims(query_features, 0)
        predictions = self({'query_features': query_features}, training=False)
        return tf.squeeze(predictions, axis=0)


class SetRetrieval(CrossAttentionSetRetrieval):
    def __init__(self, *args, **kwargs):
        if 'use_clcatneg' in kwargs:
            kwargs['use_tpaneg'] = kwargs.pop('use_clcatneg')
        super().__init__(*args, **kwargs)
        print("🚀 TPaNeg (Dynamic Hard Negative) Model initialized")

        self.validation_loss_tracker = tf.keras.metrics.Mean(name="loss") 
        self.reconstruct_x_loss_tracker = tf.keras.metrics.Mean(name="L_RecX")
        self.reconstruct_y_loss_tracker = tf.keras.metrics.Mean(name="L_RecY")

        self.initial_taneg_t_gamma_curr = self.taneg_t_gamma_init
        self.final_taneg_t_gamma_curr = self.taneg_t_gamma_final
        self.taneg_curriculum_epochs_val = self.taneg_curriculum_epochs
        self.current_epoch = 0 
        
        print(f"📚 Curriculum Learning (TaNeg T_gamma): {self.initial_taneg_t_gamma_curr:.1f} → {self.final_taneg_t_gamma_curr:.1f} over {self.taneg_curriculum_epochs_val} epochs")
        print(f"📚 PaNeg Epsilon (fixed ε): {self.paneg_epsilon:.1f}")

    def get_current_taneg_t_gamma(self):
        if self.current_epoch >= self.taneg_curriculum_epochs_val:
            return self.final_taneg_t_gamma_curr
        
        progress = self.current_epoch / self.taneg_curriculum_epochs_val
        current_t_gamma = self.initial_taneg_t_gamma_curr + (self.final_taneg_t_gamma_curr - self.initial_taneg_t_gamma_curr) * progress
        return current_t_gamma

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch
        current_t_gamma = self.get_current_taneg_t_gamma()
        if epoch % 10 == 0:
            print(f"📚 Epoch {epoch}: TaNeg T_gamma = {current_t_gamma:.3f}")

    @property
    def metrics(self):
        common_metrics = [self.loss_tracker, self.xy_loss_tracker, self.yx_loss_tracker]
        all_topk_metrics = list(self.topk_metrics.values())
        val_main_loss_metric = [self.validation_loss_tracker]
        reconstruction_metrics = [self.reconstruct_x_loss_tracker, self.reconstruct_y_loss_tracker]

        return common_metrics + reconstruction_metrics + all_topk_metrics + val_main_loss_metric

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            pred_Y = self({'query_features': data['query_features']}, training=True)
            pred_X = self({'query_features': data['target_features']}, training=True)

            if self.use_tpaneg:
                current_taneg_t_gamma = self.get_current_taneg_t_gamma()
                
                loss_X_to_Y = self.tpaneg_loss(pred_Y, data['target_features'], data['target_categories'], data['candidate_negative_features'], data['candidate_negative_masks'], current_taneg_t_gamma)
                loss_Y_to_X = self.tpaneg_loss(pred_X, data['query_features'], data['query_categories'], data['query_candidate_negative_features'], data['query_candidate_negative_masks'], current_taneg_t_gamma)

                total_loss = loss_X_to_Y + loss_Y_to_X

                if self.use_cycle_loss:
                    # Cycle loss is not part of the current debugging, so this branch is not used.
                    reconstructed_X = self({'query_features': pred_Y}, training=True)
                    reconstructed_Y = self({'query_features': pred_X}, training=True)
                    
                    cycle_loss_X = self.tpaneg_loss(reconstructed_X, data['query_features'], data['query_categories'], data['query_candidate_negative_features'], data['query_candidate_negative_masks'], current_taneg_t_gamma)
                    cycle_loss_Y = self.tpaneg_loss(reconstructed_Y, data['target_features'], data['target_categories'], data['candidate_negative_features'], data['candidate_negative_masks'], current_taneg_t_gamma)
                    total_loss += self.cycle_lambda * (cycle_loss_X + cycle_loss_Y)

            else: 
                loss_X_to_Y = self.in_batch_loss(pred_Y, data['target_features'], data['target_categories'] )
                loss_Y_to_X = self.in_batch_loss(pred_X, data['query_features'], data['query_categories'])

                total_loss = loss_X_to_Y + loss_Y_to_X

                if self.use_cycle_loss:
                    # Cycle loss is not part of the current debugging, so this branch is not used.
                    reconstructed_X = self({'query_features': pred_Y}, training=True)
                    reconstructed_Y = self({'query_features': pred_X}, training=True)
                    
                    cycle_loss_X = self.in_batch_loss(reconstructed_X, data['query_features'], data['query_categories'])
                    cycle_loss_Y = self.in_batch_loss(reconstructed_Y, data['target_features'], data['target_categories'])
                    total_loss += self.cycle_lambda * (cycle_loss_X + cycle_loss_Y)

        gradients = tape.gradient(total_loss, self.trainable_variables)


        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(total_loss)
        self.xy_loss_tracker.update_state(loss_X_to_Y)
        self.yx_loss_tracker.update_state(loss_Y_to_X)
        if self.use_cycle_loss:
            self.reconstruct_x_loss_tracker.update_state(cycle_loss_X)
            self.reconstruct_y_loss_tracker.update_state(cycle_loss_Y)

        for k in self.k_values:
            metric_name = f'top{k}_accuracy'
            if metric_name in self.topk_metrics:
                 self.topk_metrics[metric_name].update_state(pred_Y, data['target_features'], data['target_categories'])
        
        return {m.name: m.result() for m in self.metrics if not m.name.startswith('val_')}


    @tf.function
    def test_step(self, data):
        pred_Y = self({'query_features': data['query_features']}, training=False)
        
        current_taneg_t_gamma = self.get_current_taneg_t_gamma()
        
        if self.use_tpaneg and 'candidate_negative_features' in data:
            val_loss = self.tpaneg_loss(pred_Y, data['target_features'], data['target_categories'], data['candidate_negative_features'], data['candidate_negative_masks'], current_taneg_t_gamma)
        else:
            val_loss = self.in_batch_loss(pred_Y, data['target_features'], data['target_categories'])
        
        self.validation_loss_tracker.update_state(val_loss)

        for k in self.k_values:
            metric_name = f'val_top{k}_accuracy'
            if metric_name in self.topk_metrics:
                self.topk_metrics[metric_name].update_state(
                    pred_Y, data['target_features'], data['target_categories'])

        results = {"loss": self.validation_loss_tracker.result()}
        for k in self.k_values:
            metric_name = f'val_top{k}_accuracy'
            if metric_name in self.topk_metrics:
                results[metric_name] = self.topk_metrics[metric_name].result()
        return results

    def _get_predictions_for_items(self, predictions, categories):
        shape = tf.shape(categories)
        B, S = shape[0], shape[1]
        cat_indices = tf.expand_dims(tf.maximum(categories - 1, 0), axis=-1)
        batch_indices = tf.tile(tf.reshape(tf.range(B, dtype=tf.int32), [B, 1, 1]), [1, S, 1])
        indices = tf.concat([batch_indices, cat_indices], axis=-1)
        return tf.gather_nd(predictions, indices)
    
    @tf.function
    def in_batch_loss(self, predictions, target_features, target_categories):
        """
        In-batch Negative Samplingに基づくカテゴリ別コントラスティブ学習損失。
        同じセット内の同じカテゴリのアイテムはネガティブとして扱わない。
        """
        B, S, D = tf.shape(target_features)[0], tf.shape(target_features)[1], tf.shape(target_features)[2]

        target_feats_flat = tf.reshape(target_features, [-1, D]) # (B*S, D)
        target_cats_flat = tf.reshape(target_categories, [-1])   # (B*S)
        
        batch_indices = tf.range(B, dtype=tf.int32)
        item_indices = tf.range(S, dtype=tf.int32)
        grid_b, grid_i = tf.meshgrid(batch_indices, item_indices, indexing='ij')
        original_flat_batch_indices = tf.reshape(grid_b, [-1]) # 各アイテムの元のバッチインデックス (B*S,)
        original_flat_item_indices = tf.reshape(grid_i, [-1])  # 各アイテムの元のセット内インデックス (B*S,)
        
        is_valid_item_mask = tf.logical_and(target_cats_flat > 0, tf.reduce_sum(tf.abs(target_feats_flat), axis=-1) > 1e-6)
        
        preds_indices_for_gather = tf.stack([original_flat_batch_indices, tf.maximum(0, target_cats_flat - 1)], axis=1) # (B*S, 2)
        preds_for_items_flat = tf.gather_nd(predictions, preds_indices_for_gather) # (B*S, D)
        
        preds = tf.where(tf.math.is_finite(preds_for_items_flat), preds_for_items_flat, 0.0)
        targets = tf.where(tf.math.is_finite(target_feats_flat), target_feats_flat, 0.0)
        
        preds_norm, _ = tf.linalg.normalize(preds + 1e-8, axis=-1)
        targets_norm, _ = tf.linalg.normalize(targets + 1e-8, axis=-1)
        
        # 2. カテゴリ別の損失計算
        total_loss = 0.0
        total_items_contributing = 0.0 
        
        for cat_id in range(1, self.num_categories + 1):
            cat_id_tensor = tf.constant(cat_id, dtype=tf.int32)
            cat_specific_mask = tf.logical_and(tf.equal(target_cats_flat, cat_id_tensor),is_valid_item_mask)
            indices_in_cat = tf.where(cat_specific_mask)[:, 0]
            cat_count = tf.shape(indices_in_cat)[0]
            
            def process_category():
                cat_preds = tf.gather(preds_norm, indices_in_cat)    # (N_cat, D)
                cat_targets = tf.gather(targets_norm, indices_in_cat) # (N_cat, D)
                
                cat_original_batch_indices = tf.gather(original_flat_batch_indices, indices_in_cat) # (N_cat,)
                
                # カテゴリ内類似度行列 (N_cat x N_cat)
                sim_matrix = tf.matmul(cat_preds, cat_targets, transpose_b=True)
                same_batch_mask = tf.equal(tf.expand_dims(cat_original_batch_indices, 1), tf.expand_dims(cat_original_batch_indices, 0))
                
                # ポジティブペア (対角成分) を取得
                identity_mask = tf.eye(cat_count, dtype=tf.bool) # (N_cat, N_cat)
                
                neg_base_mask = tf.logical_not(identity_mask)
                neg_mask = tf.logical_and(neg_base_mask, tf.logical_not(same_batch_mask))
    
                mask_to_exclude_from_negatives = tf.logical_and(same_batch_mask, neg_base_mask)
                masked_logits = tf.where(mask_to_exclude_from_negatives, -1e9, sim_matrix / self.temperature)
                
                # 正解ラベル（対角成分が正解）
                cat_size = tf.shape(cat_preds)[0]
                labels = tf.range(cat_size, dtype=tf.int32)
                
                # InfoNCE損失
                cat_loss = tf.nn.sparse_softmax_cross_entropy_with_logits( labels=labels, logits=masked_logits)
                
                return tf.reduce_sum(cat_loss), tf.cast(cat_size, tf.float32)
            
            def skip_category():
                return 0.0, 0.0
        
            cat_loss_sum, cat_items_count = tf.cond(cat_count > 1, process_category, skip_category)
            
            total_loss += cat_loss_sum
            total_items_contributing += cat_items_count
        
        # 3. 最終損失
        def compute_final():
            return total_loss / total_items_contributing
        
        def return_zero():
            return tf.constant(0.0, dtype=tf.float32)
        
        final_loss = tf.cond(total_items_contributing > 0.0, compute_final, return_zero)
        final_loss += 0.0 * tf.reduce_sum(predictions)
        
        return final_loss


    @tf.function
    def tpaneg_loss(self, predictions, target_features, target_categories, candidate_neg_feats, candidate_neg_masks, current_taneg_t_gamma):
        B, S, D = tf.shape(target_features)[0], tf.shape(target_features)[1], tf.shape(target_features)[2]
        N_cand = tf.shape(candidate_neg_feats)[2] # N_cand は候補ネガティブの数

        # 1. データのフラット化と前処理
        target_feats_flat = tf.reshape(target_features, [-1, D])       # (B*S, D)
        target_cats_flat = tf.reshape(target_categories, [-1])         # (B*S)
        
        cand_neg_feats_flat = tf.reshape(candidate_neg_feats, [-1, N_cand, D])
        cand_neg_masks_flat = tf.reshape(candidate_neg_masks, [-1, N_cand])

        batch_indices = tf.range(B, dtype=tf.int32)
        item_indices = tf.range(S, dtype=tf.int32)
        grid_b, grid_i = tf.meshgrid(batch_indices, item_indices, indexing='ij')
        original_flat_batch_indices = tf.reshape(grid_b, [-1]) 

        is_valid_item_mask = tf.logical_and( target_cats_flat > 0, tf.reduce_sum(tf.abs(target_feats_flat), axis=-1) > 1e-6)
        
        preds_indices_for_gather = tf.stack([original_flat_batch_indices, tf.maximum(0, target_cats_flat - 1)], axis=1)
        preds_for_items_flat = tf.gather_nd(predictions, preds_indices_for_gather) # (B*S, D)
        
        preds = tf.where(tf.math.is_finite(preds_for_items_flat), preds_for_items_flat, 0.0)
        targets = tf.where(tf.math.is_finite(target_feats_flat), target_feats_flat, 0.0)
        cand_negs = tf.where(tf.math.is_finite(cand_neg_feats_flat), cand_neg_feats_flat, 0.0)
        
        preds_norm, _ = tf.linalg.normalize(preds + 1e-8, axis=-1)
        targets_norm, _ = tf.linalg.normalize(targets + 1e-8, axis=-1)
        
        mask_expanded = tf.cast(tf.expand_dims(cand_neg_masks_flat, axis=-1), tf.float32) # (B*S, N_cand, 1)
        safe_neg_feats = cand_negs * mask_expanded + (1.0 - mask_expanded) * 1e-8 # マスク外は小さな値
        cand_negs_norm, _ = tf.linalg.normalize(safe_neg_feats, axis=-1)
        cand_negs_norm = tf.where(tf.math.is_finite(cand_negs_norm), cand_negs_norm, 0.0)

        # 2. 類似度計算
        sim_target_neg = tf.einsum('id,ind->in', targets_norm, cand_negs_norm)
        sim_pred_pos = tf.einsum('id,id->i', preds_norm, targets_norm)
        sim_pred_neg = tf.einsum('id,ind->in', preds_norm, cand_negs_norm)
        
        # 3. TPaNegマスクの適用
        paneg_epsilon = tf.constant(self.paneg_epsilon, dtype=tf.float32)
        taneg_mask = sim_target_neg >= current_taneg_t_gamma以上
        paneg_mask = sim_pred_neg >= (tf.expand_dims(sim_pred_pos, 1) - paneg_epsilon)
        
        base_cand_neg_mask = cand_neg_masks_flat # DataGeneratorから来るマスク
        final_neg_mask = tf.logical_and(tf.logical_and(base_cand_neg_mask, taneg_mask), paneg_mask)

        # 4. InfoNCE損失計算
        temperature = tf.constant(self.temperature, dtype=tf.float32)
        neg_logits = tf.where(final_neg_mask, sim_pred_neg / temperature, -1e9) # (B*S, N_cand)
        pos_logits = tf.expand_dims(sim_pred_pos / temperature, 1) # (B*S, 1)
        all_logits = tf.concat([pos_logits, neg_logits], axis=1) # (B*S, 1 + N_cand)
        labels = tf.zeros_like(target_cats_flat, dtype=tf.int32) # (B*S,) 全て0
        per_item_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=all_logits) # (B*S,)
        
        # 5. 有効なアイテムのフィルタリング
        items_to_consider = tf.logical_and(is_valid_item_mask, tf.reduce_any(final_neg_mask, axis=1))
        masked_loss = per_item_loss * tf.cast(items_to_consider, dtype=tf.float32)
        num_items_for_loss = tf.reduce_sum(tf.cast(items_to_consider, dtype=tf.float32))
        final_loss = tf.math.divide_no_nan(tf.reduce_sum(masked_loss), num_items_for_loss)
        effective_negatives_per_item = tf.reduce_sum(tf.cast(final_neg_mask, tf.float32), axis=1)
        avg_effective_negatives = tf.reduce_mean(effective_negatives_per_item)

        return final_loss + (0.0 * tf.reduce_sum(predictions))



    def _get_taneg_candidates(self, item_id, category, similarity_threshold):
        """
        TaNeg: Target-aware curriculum hard Negative mining
        
        論文のEquation 7実装: N_cy(T_γ) = {l ∈ G_cy : l ≠ y, sim(l, y) ≥ T_γ}
        """
        if not hasattr(self, 'negative_pool') or category not in self.negative_pool:
            return []
        
        category_pool = self.negative_pool[category]  # 同カテゴリの全アイテム
        
        if len(category_pool) <= 1:
            return []
        
        target_features = self._get_item_features(item_id, category)
        if target_features is None:
            return []

        similarities = np.dot(category_pool, target_features)
        hard_negative_mask = similarities >= similarity_threshold
        hard_negatives = category_pool[hard_negative_mask]
        
        if len(hard_negatives) > self.candidate_neg_num:
            indices = np.random.choice(len(hard_negatives), size=self.candidate_neg_num, replace=False)
            hard_negatives = hard_negatives[indices]
        return hard_negatives.tolist()

    def _apply_curriculum_threshold_schedule(self, epoch, total_epochs):
        """
        カリキュラム学習：エポックに応じて類似度閾値を段階的に上昇   0.2 → 0.4 (IQON3000), 0.5 → 0.8 (DeepFurniture)
        """
        if self.dataset_name == 'IQON3000':
            start_threshold = 0.2
            end_threshold = 0.4
        else:  # DeepFurniture
            start_threshold = 0.5
            end_threshold = 0.8
        
        progress = min(epoch / total_epochs, 1.0)
        current_threshold = start_threshold + (end_threshold - start_threshold) * progress
        
        return current_threshold
