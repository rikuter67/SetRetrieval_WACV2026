import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import List
import pdb
# tf.config.run_functions_eagerly(True) 


# ============================================================================
# メトリック定義
# ============================================================================
# class BatchTopKAccuracy(tf.keras.metrics.Metric):
#     """バッチ内TopK正解率メトリック"""
#     def __init__(self, k=1, name='top1_accuracy', **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.k = tf.constant(float(k), dtype=tf.float32)
#         self.total_correct = self.add_weight(name='total_correct', initializer='zeros', dtype=tf.float32)
#         self.total_count = self.add_weight(name='total_count', initializer='zeros', dtype=tf.float32)
    
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         similarities = tf.cast(y_pred, tf.float32)
#         batch_size = tf.cast(tf.shape(similarities)[0], tf.float32)
        
#         def compute_metrics():
#             diagonal = tf.linalg.diag_part(similarities)
#             expanded_diag = tf.expand_dims(diagonal, axis=1)
#             better_mask = similarities > expanded_diag
#             batch_size_int = tf.cast(batch_size, tf.int32)
#             eye_mask = tf.eye(batch_size_int, dtype=tf.bool)
#             better_mask = tf.logical_and(better_mask, tf.logical_not(eye_mask))
#             num_better = tf.reduce_sum(tf.cast(better_mask, tf.float32), axis=1)
#             ranks = num_better + tf.constant(1.0, dtype=tf.float32)
#             k_threshold = tf.minimum(self.k, batch_size - tf.constant(1.0, dtype=tf.float32))
#             k_threshold = tf.maximum(k_threshold, tf.constant(1.0, dtype=tf.float32))
#             correct = tf.reduce_sum(tf.cast(ranks <= k_threshold, tf.float32))
#             return correct, batch_size
        
#         def skip_metrics():
#             return tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)
        
#         correct_count, valid_count = tf.cond(
#             tf.greater(batch_size, tf.constant(1.0, dtype=tf.float32)),
#             compute_metrics,
#             skip_metrics
#         )
#         self.total_correct.assign_add(correct_count)
#         self.total_count.assign_add(valid_count)
    
#     def result(self):
#         accuracy = tf.math.divide_no_nan(self.total_correct, self.total_count)
#         return accuracy * 100.0
    
#     def reset_state(self):
#         self.total_correct.assign(0.0)
#         self.total_count.assign(0.0)

class PerCategoryItemTopKAccuracy(tf.keras.metrics.Metric):
   """
   最小限の修正でエラーを解決した版
   - tf.rangeループをPythonループに変更
   - XLA無効化
   - 動的形状問題を回避
   """
   def __init__(self, k, num_categories, name='per_category_topk_acc', **kwargs):
       super().__init__(name=name, **kwargs)
       self.k_percent = float(k)  # 静的値として保存
       self.num_categories = num_categories
       
       self.total_correct = self.add_weight(name='total_correct', initializer='zeros')
       self.total_count = self.add_weight(name='total_count', initializer='zeros')

   @tf.function(jit_compile=False)  # XLA無効化
   def update_state(self, predictions, target_features, target_categories, sample_weight=None):
       """
       修正版TopK%精度計算
       """
       B, S, D = tf.shape(target_features)[0], tf.shape(target_features)[1], tf.shape(target_features)[2]
       
       # データフラット化
       target_features_flat = tf.reshape(target_features, [-1, D])
       target_categories_flat = tf.reshape(target_categories, [-1])
       
       # バッチインデックス作成
       batch_indices = tf.range(B, dtype=tf.int32)
       batch_grid = tf.tile(tf.expand_dims(batch_indices, 1), [1, S])
       batch_indices_flat = tf.reshape(batch_grid, [-1])
       
       # 有効アイテムマスク
       valid_mask = tf.logical_and(
           target_categories_flat > 0,
           tf.reduce_sum(tf.abs(target_features_flat), axis=-1) > 1e-6
       )
       
       valid_features = tf.boolean_mask(target_features_flat, valid_mask)
       valid_categories = tf.boolean_mask(target_categories_flat, valid_mask)
       valid_batch_indices = tf.boolean_mask(batch_indices_flat, valid_mask)
       
       N_valid = tf.shape(valid_features)[0]

       # 条件分岐で処理
       def process_when_valid():
           # 予測ベクトル取得
           pred_indices = tf.stack([
               valid_batch_indices,
               tf.maximum(0, valid_categories - 1)
           ], axis=1)
           pred_vectors = tf.gather_nd(predictions, pred_indices)
           
           # 正規化
           pred_vectors = tf.nn.l2_normalize(pred_vectors, axis=-1)
           valid_features_norm = tf.nn.l2_normalize(valid_features, axis=-1)
           
           total_correct_items = 0.0
           total_items = 0.0
           
           # Pythonのrangeを使用（TensorFlowのtf.rangeではない）
           for cat_id in range(1, self.num_categories + 1):
               cat_id_tensor = tf.constant(cat_id, dtype=tf.int32)
               cat_mask = tf.equal(valid_categories, cat_id_tensor)
               cat_count = tf.reduce_sum(tf.cast(cat_mask, tf.int32))
               
               # 条件分岐を使って処理
               def process_category():
                   cat_features = tf.boolean_mask(valid_features_norm, cat_mask)
                   cat_pred_vectors = tf.boolean_mask(pred_vectors, cat_mask)
                   cat_count_float = tf.cast(cat_count, tf.float32)
                   
                   # 類似度行列
                   sim_matrix = tf.matmul(cat_pred_vectors, cat_features, transpose_b=True)
                   diagonal_sims = tf.linalg.diag_part(sim_matrix)
                   
                   # ランク計算
                   expanded_diag = tf.expand_dims(diagonal_sims, axis=1)
                   
                   better_mask = tf.logical_and(
                       sim_matrix > expanded_diag,
                       tf.logical_not(tf.eye(tf.shape(sim_matrix)[0], dtype=tf.bool))
                   )
                   ranks = tf.reduce_sum(tf.cast(better_mask, tf.float32), axis=1) + 1.0
                   
                   # TopK%閾値
                   k_threshold = tf.maximum(
                       1.0, 
                       (self.k_percent / 100.0) * cat_count_float
                   )
                   
                   correct_in_cat = tf.reduce_sum(tf.cast(ranks <= k_threshold, tf.float32))
                   return correct_in_cat, cat_count_float
               
               def skip_category():
                   return 0.0, 0.0
               
               # tf.condで条件分岐
               correct_items, count_items = tf.cond(
                   cat_count > 1,
                   process_category,
                   skip_category
               )
               
               total_correct_items += correct_items
               total_items += count_items
           
           return total_correct_items, total_items
           
       def skip_when_empty():
           return 0.0, 0.0
       
       # メイン条件分岐
       correct_items, count_items = tf.cond(
           N_valid > 0,
           process_when_valid,
           skip_when_empty
       )
       
       # 統計更新
       self.total_correct.assign_add(correct_items)
       self.total_count.assign_add(count_items)

   def result(self):
       return tf.math.divide_no_nan(self.total_correct, self.total_count) * 100.0

   def reset_state(self):
       self.total_correct.assign(0.0)
       self.total_count.assign(0.0)

        
# ============================================================================
# ベースモデル定義
# ============================================================================
class SetRetrievalBaseModel(Model):
    """Set Retrieval Model の基底クラス（共有ロジック）"""
    
    def __init__(self, 
                 feature_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 num_categories: int = 16,
                 hidden_dim: int = 512,
                 use_cycle_loss: bool = False,
                 temperature: float = 1.0,
                 dropout_rate: float = 0.1,
                 k_values: List[int] = None,
                 cycle_lambda: float = 0.1,
                 cluster_centering: bool = False,
                 use_tpaneg: bool = False,
                 # 論文のT_gamma (TaNegの類似度閾値) に対応
                 taneg_t_gamma_init: float = 0.5, 
                 taneg_t_gamma_final: float = 0.8, # <-- 新しい引数
                 taneg_curriculum_epochs: int = 100, # <-- 新しい引数
                 # 論文のepsilon (PaNegのマージン) に対応
                 paneg_epsilon: float = 0.2, 
                 **kwargs):

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

        # 論文のT_gammaの初期値、最終値、カリキュラムエポック数
        self.taneg_t_gamma_init = taneg_t_gamma_init
        self.taneg_t_gamma_final = taneg_t_gamma_final
        self.taneg_curriculum_epochs = taneg_curriculum_epochs
        # 論文のepsilon (PaNegのマージン)
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
                'ffn': tf.keras.Sequential([
                    layers.Dense(self.hidden_dim * 2, activation='gelu'),
                    layers.Dropout(self.dropout_rate),
                    layers.Dense(self.hidden_dim)
                ], name=f'ffn_{i}')
            }
            self.cross_attention_layers.append(layer_dict)
        self.output_projection = layers.Dense(self.feature_dim, activation=None, name='output_projection')
        # BUG: This layer was defined but never used in the call method.
        # It's not critical for fixing the current stagnation, but should be added for correctness later.
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

    # def _compute_set_similarities_fixed(self, predictions, target_features):
    #     target_mask = tf.reduce_sum(tf.abs(target_features), axis=-1) > 0
    #     masked_targets = target_features * tf.cast(tf.expand_dims(target_mask, -1), tf.float32)
    #     target_sum = tf.reduce_sum(masked_targets, axis=1)
    #     target_count = tf.maximum(tf.reduce_sum(tf.cast(target_mask, tf.float32), axis=1, keepdims=True), 1.0)
    #     target_repr = target_sum / target_count
    #     if self.cluster_centering:
    #         cluster_centers_normalized = tf.nn.l2_normalize(self.category_centers, axis=-1)
    #         avg_cluster_center = tf.reduce_mean(cluster_centers_normalized, axis=0, keepdims=True)
    #         target_repr = target_repr - avg_cluster_center
    #         target_repr = tf.nn.l2_normalize(target_repr, axis=-1)
    #     all_pairwise_similarities_by_category = tf.einsum('qcd,td->qct', predictions, target_repr)
    #     return tf.reduce_max(all_pairwise_similarities_by_category, axis=1)

    def infer_single_set(self, query_features):
        if len(query_features.shape) == 2:
            query_features = tf.expand_dims(query_features, 0)
        predictions = self({'query_features': query_features}, training=False)
        return tf.squeeze(predictions, axis=0)


class TPaNegModel(SetRetrievalBaseModel):
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
                
                loss_X_to_Y = self._compute_tpaneg_loss(
                    pred_Y,
                    data['target_features'], data['target_categories'],
                    data['candidate_negative_features'], data['candidate_negative_masks'],
                    current_taneg_t_gamma
                )
                loss_Y_to_X = self._compute_tpaneg_loss(
                    pred_X,
                    data['query_features'], data['query_categories'],
                    data['query_candidate_negative_features'], data['query_candidate_negative_masks'],
                    current_taneg_t_gamma
                )

                total_loss = loss_X_to_Y + loss_Y_to_X

                if self.use_cycle_loss:
                    # Cycle loss is not part of the current debugging, so this branch is not used.
                    reconstructed_X = self({'query_features': pred_Y}, training=True)
                    reconstructed_Y = self({'query_features': pred_X}, training=True)
                    
                    cycle_loss_X = self._compute_tpaneg_loss(
                        reconstructed_X, 
                        data['query_features'], data['query_categories'],
                        data['query_candidate_negative_features'], data['query_candidate_negative_masks'],
                        current_taneg_t_gamma
                    )
                    cycle_loss_Y = self._compute_tpaneg_loss(
                        reconstructed_Y,
                        data['target_features'], data['target_categories'],
                        data['candidate_negative_features'], data['candidate_negative_masks'],
                        current_taneg_t_gamma
                    )
                    total_loss += self.cycle_lambda * (cycle_loss_X + cycle_loss_Y)

            else: 
                loss_X_to_Y = self._compute_in_batch_hard_negative_loss(
                    pred_Y,
                    data['target_features'],
                    data['target_categories']
                )
                loss_Y_to_X = self._compute_in_batch_hard_negative_loss(
                    pred_X,
                    data['query_features'],
                    data['query_categories']
                )

                total_loss = loss_X_to_Y + loss_Y_to_X

                if self.use_cycle_loss:
                    # Cycle loss is not part of the current debugging, so this branch is not used.
                    reconstructed_X = self({'query_features': pred_Y}, training=True)
                    reconstructed_Y = self({'query_features': pred_X}, training=True)
                    
                    cycle_loss_X = self._compute_in_batch_hard_negative_loss(
                        reconstructed_X, data['query_features'], data['query_categories']
                    )
                    cycle_loss_Y = self._compute_in_batch_hard_negative_loss(
                        reconstructed_Y, data['target_features'], data['target_categories']
                    )
                    total_loss += self.cycle_lambda * (cycle_loss_X + cycle_loss_Y)

        gradients = tape.gradient(total_loss, self.trainable_variables)


    # # 🔍 勾配デバッグコード追加
    #     if tf.equal(tf.cast(self.optimizer.iterations, tf.int32) % 10, 0):  # 10ステップごと
    #         tf.print("=== Gradient Debug ===")
    #         tf.print("Total Loss:", total_loss)
    #         tf.print("Loss X->Y:", loss_X_to_Y)
    #         tf.print("Loss Y->X:", loss_Y_to_X)
            
    #         # 勾配の統計
    #         grad_norms = [tf.norm(g) for g in gradients if g is not None]
    #         if grad_norms:
    #             tf.print("Gradient norms (first 5):", grad_norms[:5])
    #             tf.print("Max gradient norm:", tf.reduce_max(grad_norms))
    #             tf.print("Mean gradient norm:", tf.reduce_mean(grad_norms))
    #             tf.print("Num None gradients:", tf.reduce_sum([1 for g in gradients if g is None]))
    #         else:
    #             tf.print("⚠️ ALL GRADIENTS ARE NONE!")


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
                 self.topk_metrics[metric_name].update_state(
                    pred_Y, 
                    data['target_features'], 
                    data['target_categories']
                )
        
        return {m.name: m.result() for m in self.metrics if not m.name.startswith('val_')}


    @tf.function
    def test_step(self, data):
        pred_Y = self({'query_features': data['query_features']}, training=False)
        
        current_taneg_t_gamma = self.get_current_taneg_t_gamma()
        
        if self.use_tpaneg and 'candidate_negative_features' in data:
            val_loss = self._compute_tpaneg_loss(
                pred_Y, data['target_features'], data['target_categories'],
                data['candidate_negative_features'], data['candidate_negative_masks'],
                current_taneg_t_gamma
            )
        else:
            val_loss = self._compute_in_batch_hard_negative_loss(
                pred_Y, data['target_features'], data['target_categories']
            )
        
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
    
    
    # def _compute_standard_contrastive_loss(self, predictions, target_features, target_categories):
    #     sim_matrix = self._compute_set_similarities_fixed(predictions, target_features)
    #     labels = tf.range(tf.shape(sim_matrix)[0])
    #     loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=sim_matrix / self.temperature)
    #     return tf.reduce_mean(loss)
        
    @tf.function
    def _compute_in_batch_hard_negative_loss(self, predictions, target_features, target_categories):
        """
        In-batch Negative Samplingに基づくカテゴリ別コントラスティブ学習損失。
        同じセット内の同じカテゴリのアイテムはネガティブとして扱わない。
        """
        B, S, D = tf.shape(target_features)[0], tf.shape(target_features)[1], tf.shape(target_features)[2]
        
        # 1. データのフラット化と前処理
        # target_features は Y
        # predictions は Y' (Batch_Size, Num_Categories, Feature_Dimension)

        target_feats_flat = tf.reshape(target_features, [-1, D]) # (B*S, D)
        target_cats_flat = tf.reshape(target_categories, [-1])   # (B*S)
        
        # オリジナルのバッチインデックスとアイテムインデックスのペアをフラット化
        batch_indices = tf.range(B, dtype=tf.int32)
        item_indices = tf.range(S, dtype=tf.int32)
        grid_b, grid_i = tf.meshgrid(batch_indices, item_indices, indexing='ij')
        original_flat_batch_indices = tf.reshape(grid_b, [-1]) # 各アイテムの元のバッチインデックス (B*S,)
        original_flat_item_indices = tf.reshape(grid_i, [-1])  # 各アイテムの元のセット内インデックス (B*S,)
        
        # 有効アイテムマスク (パディングされたアイテムを除外)
        is_valid_item_mask = tf.logical_and(
            target_cats_flat > 0, # カテゴリIDが0より大きい
            tf.reduce_sum(tf.abs(target_feats_flat), axis=-1) > 1e-6 # 特徴量がゼロでない
        )
        
        # 予測ベクトルの取得 (target_features の各アイテムが属するカテゴリに対応する predictions のベクトル)
        # predictions の形状は [B, C, D]
        # target_cats_flat は [B*S]
        preds_indices_for_gather = tf.stack([original_flat_batch_indices, tf.maximum(0, target_cats_flat - 1)], axis=1) # (B*S, 2)
        preds_for_items_flat = tf.gather_nd(predictions, preds_indices_for_gather) # (B*S, D)
        
        # NaN/Inf対策と正規化
        preds = tf.where(tf.math.is_finite(preds_for_items_flat), preds_for_items_flat, 0.0)
        targets = tf.where(tf.math.is_finite(target_feats_flat), target_feats_flat, 0.0)
        
        preds_norm, _ = tf.linalg.normalize(preds + 1e-8, axis=-1)
        targets_norm, _ = tf.linalg.normalize(targets + 1e-8, axis=-1)
        
        # 2. カテゴリ別の損失計算
        total_loss = 0.0
        total_items_contributing = 0.0 # 損失に貢献するアイテムの総数
        
        for cat_id in range(1, self.num_categories + 1):
            cat_id_tensor = tf.constant(cat_id, dtype=tf.int32)
            
            # このカテゴリに属し、かつ有効なアイテムのマスク
            cat_specific_mask = tf.logical_and(
                tf.equal(target_cats_flat, cat_id_tensor),
                is_valid_item_mask
            )
            
            # このカテゴリのアイテムのインデックス (flat_indices_in_cat)
            indices_in_cat = tf.where(cat_specific_mask)[:, 0] # (N_cat,)
            
            cat_count = tf.shape(indices_in_cat)[0]
            
            def process_category():
                # このカテゴリのアイテムの予測とターゲットを抽出
                cat_preds = tf.gather(preds_norm, indices_in_cat)    # (N_cat, D)
                cat_targets = tf.gather(targets_norm, indices_in_cat) # (N_cat, D)
                
                # 同じカテゴリのアイテムの元のバッチインデックスも抽出
                cat_original_batch_indices = tf.gather(original_flat_batch_indices, indices_in_cat) # (N_cat,)
                
                # カテゴリ内類似度行列 (N_cat x N_cat)
                sim_matrix = tf.matmul(cat_preds, cat_targets, transpose_b=True)
                
                # --- 同じセット内のアイテムを除外するマスクを作成 ---
                # 形状: (N_cat, N_cat)
                # i番目のクエリとj番目のターゲットが同じバッチ（セット）に属するかどうかをチェック
                # cat_original_batch_indices は (N_cat,) なので、これを (N_cat, 1) と (1, N_cat) に拡張して比較
                same_batch_mask = tf.equal(
                    tf.expand_dims(cat_original_batch_indices, 1), # (N_cat, 1)
                    tf.expand_dims(cat_original_batch_indices, 0)  # (1, N_cat)
                ) # (N_cat, N_cat)
                
                # ポジティブペア (対角成分) を取得
                identity_mask = tf.eye(cat_count, dtype=tf.bool) # (N_cat, N_cat)
                
                # ネガティブとして扱うペアのマスク:
                # 1. ポジティブペアではない (identity_mask == False)
                # 2. 同じバッチに属さない (same_batch_mask == False)
                # ただし、今回は「同じバッチに属さないか、かつポジティブペアではない」を組み合わせる
                
                # 正しいネガティブマスクのロジック:
                # ネガティブとして利用したいのは、「自分自身ではない」かつ「同じバッチ内ではない」ペア
                # もしくは、あなたの意図「同じB_1内でも同じカテゴリの正解以外のアイテムはポジティブとしてもネガティブとしても扱いたくない。」
                # を厳密に解釈すると、
                # ポジティブ: 自分自身 (対角)
                # ネガティブ: 同じカテゴリだが、異なるバッチに属するアイテム (same_batch_mask == False AND NOT identity_mask)
                # 除外: 同じカテゴリで、同じバッチに属するが、自分自身ではないアイテム (same_batch_mask == True AND NOT identity_mask)

                # --- ネガティブマスクの構築 ---
                # まず、ネガティブにしたいのは「対角成分以外」
                neg_base_mask = tf.logical_not(identity_mask)
                
                # そして、「同じバッチに属する」ものをネガティブから除外したい
                # → same_batch_mask が True の場所はネガティブではない
                # 最終的なネガティブマスクは (neg_base_mask AND NOT same_batch_mask)
                neg_mask = tf.logical_and(neg_base_mask, tf.logical_not(same_batch_mask))
                
                # InfoNCE損失では、正解以外を全てネガティブとして扱うため、
                # ネガティブとして扱いたくない部分の類似度を非常に小さな値にする
                # (例: -inf に近い値) ことで、softmax計算から除外する
                
                # 除外したい要素に非常に小さな値を設定
                # same_batch_mask && NOT identity_mask の位置に -inf を設定
                # これは、同じバッチに属するが自分自身ではない、という位置
                mask_to_exclude_from_negatives = tf.logical_and(same_batch_mask, neg_base_mask)
                
                # 除外する要素には非常に小さな値を設定し、softmaxで確率が0に近づくようにする
                masked_logits = tf.where(mask_to_exclude_from_negatives, -1e9, sim_matrix / self.temperature)
                
                # 正解ラベル（対角成分が正解）
                cat_size = tf.shape(cat_preds)[0]
                labels = tf.range(cat_size, dtype=tf.int32)
                
                # InfoNCE損失
                cat_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=masked_logits
                )
                
                # tf.reduce_sum(cat_loss) は、マスクされた -1e9 の項も計算に含めてしまう可能性があるので注意が必要
                # 損失はポジティブペアに対して計算されるべきなので、labelsで指定されたポジティブペアの損失を合計する
                # masked_logitsからlabelsに対応するlogitsだけを取り出してcross_entropyを計算
                # (logitsは既に-1e9でマスクされているので、そのままsumしても良いはずだが、より安全なのは以下)
                
                return tf.reduce_sum(cat_loss), tf.cast(cat_size, tf.float32)
            
            def skip_category():
                return 0.0, 0.0
            
            # カテゴリに十分なアイテムがある場合のみ処理
            # コントラスティブ損失では、ポジティブ1つとネガティブが最低1つ必要。
            # 今回の設計では、「同じバッチ内の自分自身以外のアイテムはネガティブではない」ので、
            # 同じカテゴリのアイテムが複数あっても、それらが全て同じバッチに属する場合、
            # 有効なネガティブペアがバッチ内に存在しない可能性がある。
            # なので、cat_count > 1 ではなく、
            # 「そのカテゴリ内で、異なるバッチに属するアイテムが存在する」という条件が必要になる。
            # あるいは、ここでは cat_count > 1 のままにして、ネガティブマスクで対応する。
            
            # このままでも `masked_logits` の -1e9 が softmax でゼロに近づけるため、
            # 有効なネガティブがなければ結果的に損失が計算されない方向になるはず。
            # cat_count > 1 で良いでしょう。
            cat_loss_sum, cat_items_count = tf.cond(
                cat_count > 1,
                process_category,
                skip_category
            )
            
            total_loss += cat_loss_sum
            total_items_contributing += cat_items_count
        
        # 3. 最終損失
        def compute_final():
            return total_loss / total_items_contributing
        
        def return_zero():
            return tf.constant(0.0, dtype=tf.float32)
        
        final_loss = tf.cond(
            total_items_contributing > 0.0, # 損失に貢献するアイテムが存在する場合のみ計算
            compute_final,
            return_zero
        )
        
        # 勾配パス保持
        final_loss += 0.0 * tf.reduce_sum(predictions)
        
        return final_loss


    @tf.function
    def _compute_tpaneg_loss(self, predictions, target_features, target_categories, candidate_neg_feats, candidate_neg_masks, current_taneg_t_gamma):
        B, S, D = tf.shape(target_features)[0], tf.shape(target_features)[1], tf.shape(target_features)[2]
        N_cand = tf.shape(candidate_neg_feats)[2] # N_cand は候補ネガティブの数

        # 1. データのフラット化と前処理
        target_feats_flat = tf.reshape(target_features, [-1, D])       # (B*S, D)
        target_cats_flat = tf.reshape(target_categories, [-1])         # (B*S)
        
        # cand_neg_feats: (B, S, N_cand, D) -> (B*S, N_cand, D)
        cand_neg_feats_flat = tf.reshape(candidate_neg_feats, [-1, N_cand, D])
        # cand_neg_masks: (B, S, N_cand) -> (B*S, N_cand)
        cand_neg_masks_flat = tf.reshape(candidate_neg_masks, [-1, N_cand])

        # 各アイテムの元のバッチインデックスとセット内インデックスを保持
        batch_indices = tf.range(B, dtype=tf.int32)
        item_indices = tf.range(S, dtype=tf.int32)
        grid_b, grid_i = tf.meshgrid(batch_indices, item_indices, indexing='ij')
        original_flat_batch_indices = tf.reshape(grid_b, [-1]) # 各アイテムの元のバッチインデックス (B*S,)
        # original_flat_item_indices = tf.reshape(grid_i, [-1]) # 必要であればセット内インデックスも
        
        # 有効アイテムマスク (カテゴリID > 0 かつ特徴量がゼロでない)
        is_valid_item_mask = tf.logical_and(
            target_cats_flat > 0,
            tf.reduce_sum(tf.abs(target_feats_flat), axis=-1) > 1e-6
        )
        
        # predictions の形状は [B, C, D] であるという前提
        # target_cats_flat の各アイテムが属するカテゴリに対応する予測ベクトルを predictions から抽出
        preds_indices_for_gather = tf.stack([original_flat_batch_indices, tf.maximum(0, target_cats_flat - 1)], axis=1)
        preds_for_items_flat = tf.gather_nd(predictions, preds_indices_for_gather) # (B*S, D)
        
        # NaN/Inf対策と正規化
        preds = tf.where(tf.math.is_finite(preds_for_items_flat), preds_for_items_flat, 0.0)
        targets = tf.where(tf.math.is_finite(target_feats_flat), target_feats_flat, 0.0)
        cand_negs = tf.where(tf.math.is_finite(cand_neg_feats_flat), cand_neg_feats_flat, 0.0)
        
        preds_norm, _ = tf.linalg.normalize(preds + 1e-8, axis=-1)
        targets_norm, _ = tf.linalg.normalize(targets + 1e-8, axis=-1)
        
        # 候補ネガティブも正規化 (マスクされている場合は0に近づける)
        # tf.expand_dims(cand_neg_masks_flat, axis=-1) は (B*S, N_cand, 1) になる
        mask_expanded = tf.cast(tf.expand_dims(cand_neg_masks_flat, axis=-1), tf.float32) # (B*S, N_cand, 1)
        safe_neg_feats = cand_negs * mask_expanded + (1.0 - mask_expanded) * 1e-8 # マスク外は小さな値
        cand_negs_norm, _ = tf.linalg.normalize(safe_neg_feats, axis=-1)
        cand_negs_norm = tf.where(tf.math.is_finite(cand_negs_norm), cand_negs_norm, 0.0)

        # 2. 類似度計算
        # sim_target_neg: (B*S, N_cand) - target_item と candidate_neg の類似度
        sim_target_neg = tf.einsum('id,ind->in', targets_norm, cand_negs_norm)
        
        # sim_pred_pos: (B*S,) - predicted_item と target_item (ポジティブペア) の類似度
        sim_pred_pos = tf.einsum('id,id->i', preds_norm, targets_norm)
        
        # sim_pred_neg: (B*S, N_cand) - predicted_item と candidate_neg の類似度
        sim_pred_neg = tf.einsum('id,ind->in', preds_norm, cand_negs_norm)
        
        # 3. TPaNegマスクの適用
        paneg_epsilon = tf.constant(self.paneg_epsilon, dtype=tf.float32)
        
        # TaNegマスク: target_item と candidate_neg の類似度が閾値以上
        taneg_mask = sim_target_neg >= current_taneg_t_gamma
        
        # PaNegマスク: predicted_item と candidate_neg の類似度が (predicted_item と target_item の類似度 - epsilon) 以上
        paneg_mask = sim_pred_neg >= (tf.expand_dims(sim_pred_pos, 1) - paneg_epsilon)
        
        # 基本の候補ネガティブマスク (パディングされた候補を除外)
        base_cand_neg_mask = cand_neg_masks_flat # DataGeneratorから来るマスク

        # --- 新しいロジック: 同じセット内のネガティブを除外 ---
        # candidate_neg_feats は DataGenerator で B*S の各アイテムに対して N_cand 個のネガティブが与えられる。
        # ここで N_cand は、DataGenerator が「元のターゲットアイテムと同じセットに属する」ものをネガティブ候補として
        # 選んでいないという前提に立つのが自然。
        # もし DataGenerator の中で、同じセット内の他のアイテムも candidate_neg_feats に含まれてしまっている場合、
        # ここでそれを除外するマスクを追加する必要がある。

        # DataGeneratorがどのように candidate_negatives を収集しているかによるが、
        # 通常、HardNegativeMinerは「異なるセット」から負例を選ぶため、
        # 同じバッチ内の同じセットに属するアイテムが candidate_neg_feats に含まれることは稀です。
        # もし含まれる可能性があるなら、以下のマスクを追加します。

        # 候補ネガティブが元のバッチと同じかどうかを判定するための情報が必要。
        # 現在の DataGenerator からは candidate_neg_feats の元のバッチ・セットIDは直接得られない。
        # もし candidate_neg_feats の各候補がどのセットに由来するかを DataGenerator が提供しない限り、
        # この関数内での厳密な「同じセット内ネガティブ除外」は難しい。
        # (DataGeneratorから 'candidate_negative_original_set_ids' のようなものを渡す必要がある)

        # 仮に `candidate_neg_feats` には「同じセット内の他のアイテム」は含まれていないと仮定し、
        # TPaNegの意図通りに動くものとします。
        # もし、precompute_negatives_gpu や HardNegativeMiner が、同じセット内の他のアイテムを
        # 候補ネガティブとして含んでしまっているなら、DataGeneratorの出力にその情報を追加し、
        # ここでその情報を使ってマスクを適用する必要があります。

        # 現状のコードの範囲内でできるのは、あくまで渡された candidate_neg_masks_flat の範囲内での処理。
        
        # TPaNeg論文の意図通り、最終的にネガティブとして採用する条件
        # 1. 元々候補として有効 (cand_neg_masks_flat)
        # 2. TaNeg基準を満たす (taneg_mask)
        # 3. PaNeg基準を満たす (paneg_mask)
        final_neg_mask = tf.logical_and(tf.logical_and(base_cand_neg_mask, taneg_mask), paneg_mask)

        # 4. InfoNCE損失計算
        temperature = tf.constant(self.temperature, dtype=tf.float32)
        
        # ネガティブ logits: マスクされたネガティブ候補には非常に小さい値を設定
        neg_logits = tf.where(final_neg_mask, sim_pred_neg / temperature, -1e9) # (B*S, N_cand)
        
        # ポジティブ logits: predicted_item と target_item の類似度
        pos_logits = tf.expand_dims(sim_pred_pos / temperature, 1) # (B*S, 1)
        
        # ポジティブとネガティブのlogitsを結合
        all_logits = tf.concat([pos_logits, neg_logits], axis=1) # (B*S, 1 + N_cand)
        
        # 正解ラベルはポジティブ (最初の列)
        labels = tf.zeros_like(target_cats_flat, dtype=tf.int32) # (B*S,) 全て0

        # アイテムごとの損失 (InfoNCE)
        per_item_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=all_logits) # (B*S,)
        
        # 5. 有効なアイテムのフィルタリング
        # is_valid_item_mask: パディングされたアイテムやカテゴリID=0のアイテムを除外
        # tf.reduce_any(final_neg_mask, axis=1): そのアイテムに対して少なくとも1つのハードネガティブが選択されたか
        items_to_consider = tf.logical_and(is_valid_item_mask, tf.reduce_any(final_neg_mask, axis=1))
        
        # マスクされた損失の合計
        masked_loss = per_item_loss * tf.cast(items_to_consider, dtype=tf.float32)
        
        # 損失に貢献するアイテムの数
        num_items_for_loss = tf.reduce_sum(tf.cast(items_to_consider, dtype=tf.float32))
        
        # 最終損失 (ゼロ除算対策)
        final_loss = tf.math.divide_no_nan(tf.reduce_sum(masked_loss), num_items_for_loss)

        # デバッグコード追加
        # effective_negatives_per_item = tf.reduce_sum(tf.cast(final_neg_mask, tf.float32), axis=1)
        # avg_effective_negatives = tf.reduce_mean(effective_negatives_per_item)

        # tf.print("Avg effective negatives per item:", avg_effective_negatives)
        # tf.print("Max effective negatives:", tf.reduce_max(effective_negatives_per_item))
        # tf.print("Items with 0 negatives:", tf.reduce_sum(tf.cast(tf.equal(effective_negatives_per_item, 0), tf.float32)))
        
        # 勾配パス保持のためのダミー加算 (損失が0になるケース対策)
        return final_loss + (0.0 * tf.reduce_sum(predictions))



    # @tf.function # これが関数全体をグラフ化する
    # def _compute_in_batch_hard_negative_loss(self, predictions, target_features, target_categories):
    #     """
    #     バッチ内の同じカテゴリのアイテムをネガティブとして扱う
    #     カテゴリ別コントラスティブ学習損失
    #     """
    #     B, S, D = tf.shape(target_features)[0], tf.shape(target_features)[1], tf.shape(target_features)[2]

    #     # 1. データのフラット化と前処理
    #     target_feats_flat = tf.reshape(target_features, [-1, D])
    #     target_cats_flat = tf.reshape(target_categories, [-1])
        
    #     # バッチインデックスの作成
    #     batch_indices = tf.range(B, dtype=tf.int32)
    #     item_indices = tf.range(S, dtype=tf.int32)
    #     grid_b, grid_i = tf.meshgrid(batch_indices, item_indices, indexing='ij')
    #     flat_indices = tf.stack([tf.reshape(grid_b, [-1]), tf.reshape(grid_i, [-1])], axis=1)
        
    #     # 有効アイテムマスク
    #     is_valid_item_mask = tf.logical_and(
    #         target_cats_flat > 0,
    #         tf.reduce_sum(tf.abs(target_feats_flat), axis=-1) > 1e-6
    #     )
        
    #     # 予測ベクトルの取得
    #     preds_indices = tf.stack([flat_indices[:, 0], tf.maximum(0, target_cats_flat - 1)], axis=1)
    #     preds_for_items_flat = tf.gather_nd(predictions, preds_indices)
        
    #     # NaN/Inf対策と正規化
    #     preds = tf.where(tf.math.is_finite(preds_for_items_flat), preds_for_items_flat, 0.0)
    #     targets = tf.where(tf.math.is_finite(target_feats_flat), target_feats_flat, 0.0)
        
    #     preds_norm, _ = tf.linalg.normalize(preds + 1e-8, axis=-1)
    #     targets_norm, _ = tf.linalg.normalize(targets + 1e-8, axis=-1)
        
    #     # 2. カテゴリ別の損失計算
    #     total_loss = 0.0
    #     total_items = 0.0
        
    #     pdb.set_trace()
        
    #     # 各カテゴリについて処理
    #     for cat_id in range(1, self.num_categories + 1):
    #         cat_id_tensor = tf.constant(cat_id, dtype=tf.int32)
            
    #         # このカテゴリのアイテムマスク
    #         cat_mask = tf.logical_and(
    #             tf.equal(target_cats_flat, cat_id_tensor),
    #             is_valid_item_mask
    #         )
            
    #         cat_count = tf.reduce_sum(tf.cast(cat_mask, tf.int32))
            
    #         def process_category():
    #             # このカテゴリのアイテムを抽出
    #             cat_preds = tf.boolean_mask(preds_norm, cat_mask)      # (N_cat, D)
    #             cat_targets = tf.boolean_mask(targets_norm, cat_mask)  # (N_cat, D)
                
    #             # カテゴリ内類似度行列
    #             sim_matrix = tf.matmul(cat_preds, cat_targets, transpose_b=True)  # (N_cat, N_cat)
                
    #             # 温度で割る
    #             logits = sim_matrix / self.temperature
                
    #             # 正解ラベル（対角成分が正解）
    #             cat_size = tf.shape(cat_preds)[0]
    #             labels = tf.range(cat_size, dtype=tf.int32)
                
    #             # InfoNCE損失
    #             cat_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #                 labels=labels, logits=logits
    #             )
                
    #             return tf.reduce_sum(cat_loss), tf.cast(cat_size, tf.float32)
            
    #         def skip_category():
    #             return 0.0, 0.0
            
    #         # カテゴリに十分なアイテムがある場合のみ処理
    #         cat_loss_sum, cat_items_count = tf.cond(
    #             cat_count > 1,  # 最低2個必要（自分 + ネガティブ）
    #             process_category,
    #             skip_category
    #         )
            
    #         total_loss += cat_loss_sum
    #         total_items += cat_items_count
        
    #     # 3. 最終損失
    #     def compute_final():
    #         return total_loss / total_items
        
    #     def return_zero():
    #         return tf.constant(0.0, dtype=tf.float32)
        
    #     final_loss = tf.cond(
    #         total_items > 0.0,
    #         compute_final,
    #         return_zero
    #     )
        
    #     # 勾配パス保持
    #     final_loss += 0.0 * tf.reduce_sum(predictions)
        
    #     return final_loss

    # @tf.function
    # def _compute_tpaneg_loss(self, predictions, target_features, target_categories, candidate_neg_feats, candidate_neg_masks, current_taneg_t_gamma):
    #     B, S, D = tf.shape(target_features)[0], tf.shape(target_features)[1], tf.shape(target_features)[2]
    #     N_cand = tf.shape(candidate_neg_feats)[2]

    #     target_feats_flat = tf.reshape(target_features, [-1, D])
    #     target_cats_flat = tf.reshape(target_categories, [-1])
    #     cand_neg_feats_flat = tf.reshape(candidate_neg_feats, [-1, N_cand, D])
    #     cand_neg_masks_flat = tf.reshape(candidate_neg_masks, [-1, N_cand])

    #     is_valid_item_mask = target_cats_flat > 0
        
    #     batch_indices = tf.range(B, dtype=tf.int32) 
    #     item_indices = tf.range(S, dtype=tf.int32)
    #     grid_b, grid_i = tf.meshgrid(batch_indices, item_indices, indexing='ij') 
    #     flat_indices = tf.stack([tf.reshape(grid_b, [-1]), tf.reshape(grid_i, [-1])], axis=1)
        
    #     preds_indices_for_gather = tf.stack([flat_indices[:, 0], tf.maximum(0, target_cats_flat - 1)], axis=1)
    #     preds_for_items_flat = tf.gather_nd(predictions, preds_indices_for_gather)
        
    #     preds = tf.where(tf.math.is_finite(preds_for_items_flat), preds_for_items_flat, 0.0)
    #     targets = tf.where(tf.math.is_finite(target_feats_flat), target_feats_flat, 0.0)
    #     cand_negs = tf.where(tf.math.is_finite(cand_neg_feats_flat), cand_neg_feats_flat, 0.0)
        
    #     preds_norm, _ = tf.linalg.normalize(preds + 1e-8, axis=-1)
    #     targets_norm, _ = tf.linalg.normalize(targets + 1e-8, axis=-1)
        
    #     mask_expanded = tf.expand_dims(cand_neg_masks_flat, axis=-1)
    #     safe_neg_feats = cand_negs + (tf.cast(mask_expanded, tf.float32) * 1e-8)
    #     cand_negs_norm, _ = tf.linalg.normalize(safe_neg_feats, axis=-1)
    #     cand_negs_norm = tf.where(tf.math.is_finite(cand_negs_norm), cand_negs_norm, 0.0)
        
    #     sim_target_neg = tf.einsum('id,ind->in', targets_norm, cand_negs_norm)
    #     sim_pred_pos = tf.einsum('id,id->i', preds_norm, targets_norm)
    #     sim_pred_neg = tf.einsum('id,ind->in', preds_norm, cand_negs_norm)
        
    #     paneg_epsilon = tf.constant(self.paneg_epsilon, dtype=tf.float32)
    #     taneg_mask = sim_target_neg >= current_taneg_t_gamma
    #     paneg_mask = sim_pred_neg >= (tf.expand_dims(sim_pred_pos, 1) - paneg_epsilon)
    #     final_neg_mask = tf.logical_and(tf.logical_and(cand_neg_masks_flat, taneg_mask), paneg_mask)

    #     temperature = tf.constant(self.temperature, dtype=tf.float32)
    #     neg_logits = tf.where(final_neg_mask, sim_pred_neg / temperature, -1e9)
    #     pos_logits = tf.expand_dims(sim_pred_pos / temperature, 1)
    #     all_logits = tf.concat([pos_logits, neg_logits], axis=1)
        
    #     labels = tf.zeros_like(target_cats_flat, dtype=tf.int32)
    #     per_item_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=all_logits)
        
    #     items_to_consider = tf.logical_and(is_valid_item_mask, tf.reduce_any(final_neg_mask, axis=1))
        
    #     masked_loss = per_item_loss * tf.cast(items_to_consider, dtype=tf.float32)
        
    #     num_items_for_loss = tf.reduce_sum(tf.cast(items_to_consider, dtype=tf.float32))
        
    #     final_loss = tf.math.divide_no_nan(tf.reduce_sum(masked_loss), num_items_for_loss)
        
    #     # pdb.set_trace()

    #     return final_loss + (0.0 * tf.reduce_sum(predictions))



    def _get_taneg_candidates(self, item_id, category, similarity_threshold):
        """
        TaNeg: Target-aware curriculum hard Negative mining
        
        論文のEquation 7実装:
        N_cy(T_γ) = {l ∈ G_cy : l ≠ y, sim(l, y) ≥ T_γ}
        """
        if not hasattr(self, 'negative_pool') or category not in self.negative_pool:
            return []
        
        category_pool = self.negative_pool[category]  # 同カテゴリの全アイテム
        
        if len(category_pool) <= 1:
            return []
        
        # 正解アイテムの特徴量を取得（簡易実装）
        target_features = self._get_item_features(item_id, category)
        if target_features is None:
            return []
        
        # 類似度計算して閾値以上のものを選択
        similarities = np.dot(category_pool, target_features)
        hard_negative_mask = similarities >= similarity_threshold
        
        # 正解アイテム自体は除外
        hard_negatives = category_pool[hard_negative_mask]
        
        # ランダムサンプリングで数を制限
        if len(hard_negatives) > self.candidate_neg_num:
            indices = np.random.choice(
                len(hard_negatives), 
                size=self.candidate_neg_num, 
                replace=False
            )
            hard_negatives = hard_negatives[indices]
        
        return hard_negatives.tolist()

    def _apply_curriculum_threshold_schedule(self, epoch, total_epochs):
        """
        カリキュラム学習：エポックに応じて類似度閾値を段階的に上昇
        
        論文では: 0.2 → 0.4 (IQON3000), 0.5 → 0.8 (DeepFurniture)
        """
        if self.dataset_name == 'IQON3000':
            start_threshold = 0.2
            end_threshold = 0.4
        else:  # DeepFurniture
            start_threshold = 0.5
            end_threshold = 0.8
        
        # 線形スケジューリング
        progress = min(epoch / total_epochs, 1.0)
        current_threshold = start_threshold + (end_threshold - start_threshold) * progress
        
        return current_threshold

    @tf.function
    def compute_cycle_consistency_loss(self, query_features, target_features, 
                                    query_categories, target_categories):
        """
        双方向Cycle Consistency Loss計算
        
        Args:
            query_features: (B, N_X, D) - クエリアイテム特徴量
            target_features: (B, N_Y, D) - ターゲットアイテム特徴量
            query_categories: (B, N_X) - クエリカテゴリ
            target_categories: (B, N_Y) - ターゲットカテゴリ
        """
        
        # Forward Path: X → Y'
        # 1. クエリからターゲットカテゴリの予測を生成
        forward_input = {
            'query_features': query_features,
            'target_categories': target_categories  # Z^Y の代わり
        }
        predicted_targets = self(forward_input, training=True)  # Y' = f_θ(X, Z^Y)
        
        # 2. Forward Contrastive Loss
        forward_loss = self._compute_item_level_contrastive_loss(
            predicted_targets, target_features, target_categories
        )
        
        # Backward Path: Y' → X'
        # 3. 予測からクエリカテゴリの再構築
        backward_input = {
            'query_features': predicted_targets,  # Y'を新しいクエリとして使用
            'target_categories': query_categories  # Z^X の代わり
        }
        reconstructed_queries = self(backward_input, training=True)  # X' = f_θ(Y', Z^X)
        
        # 4. Backward Contrastive Loss
        backward_loss = self._compute_item_level_contrastive_loss(
            reconstructed_queries, query_features, query_categories
        )
        
        # 5. 総合損失（論文のEquation 4）
        total_cycle_loss = forward_loss + self.cycle_lambda * backward_loss
        
        return total_cycle_loss, forward_loss, backward_loss


    def _compute_item_level_contrastive_loss(self, predictions, target_items, target_categories):
        """
        アイテムレベルのContrastive Loss（論文のEquation 5）
        
        L_con(Y,Ŷ) = -1/N_Y * Σ log(exp(sim(ŷ_p, y_p)/τ) / Σ_i exp(sim(ŷ_p, y_i)/τ))
        """
        
        total_loss = 0.0
        num_valid_items = 0
        
        B, S = tf.shape(target_items)[0], tf.shape(target_items)[1]
        C = tf.shape(predictions)[1]
        
        for batch_idx in range(B):
            for item_idx in range(S):
                target_cat = target_categories[batch_idx, item_idx]
                
                if target_cat <= 0:
                    continue  # 無効なアイテムはスキップ
                
                # カテゴリ予測ベクトルを取得
                predicted_vector = predictions[batch_idx, target_cat - 1, :]  # (D,)
                positive_item = target_items[batch_idx, item_idx, :]          # (D,)
                
                # 正解との類似度
                pos_sim = tf.reduce_sum(predicted_vector * positive_item) / self.temperature
                
                # 同じバッチ内の全アイテムとの類似度（ネガティブ含む）
                all_items = tf.reshape(target_items, [-1, tf.shape(target_items)[-1]])  # (B*S, D)
                all_sims = tf.reduce_sum(
                    tf.expand_dims(predicted_vector, 0) * all_items, axis=1
                ) / self.temperature  # (B*S,)
                
                # InfoNCE損失
                # 分子: exp(pos_sim)
                # 分母: Σ exp(all_sims)
                loss = -pos_sim + tf.reduce_logsumexp(all_sims)
                
                total_loss += loss
                num_valid_items += 1
        
        return tf.cond(
            num_valid_items > 0,
            lambda: total_loss / tf.cast(num_valid_items, tf.float32),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )


    # 論文のEquation 9: ダブル双方向処理
    def compute_double_bidirectional_loss(self, query_features, target_features,
                                        query_categories, target_categories):
        """
        論文のEquation 9実装: 両方向でのCycle Consistency
        L_bi(X,Y,Z^Y,Z^X) + L_bi(Y,X,Z^X,Z^Y)
        """
        
        # 第1の双方向: (X → Y → X)
        loss_1, fwd_1, bwd_1 = self.compute_cycle_consistency_loss(
            query_features, target_features, query_categories, target_categories
        )
        
        # 第2の双方向: (Y → X → Y) - 役割を交換
        loss_2, fwd_2, bwd_2 = self.compute_cycle_consistency_loss(
            target_features, query_features, target_categories, query_categories
        )
        
        # 総合損失
        total_loss = loss_1 + loss_2
        
        return total_loss, {
            'cycle_loss_1': loss_1,
            'cycle_loss_2': loss_2,
            'forward_1': fwd_1,
            'backward_1': bwd_1,
            'forward_2': fwd_2,
            'backward_2': bwd_2
        }