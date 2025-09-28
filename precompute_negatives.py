import os
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from collections import defaultdict
import argparse

# tf.config.run_functions_eagerly(True) # デバッグ時以外はコメントアウト

def precompute_negatives_gpu(dataset_path: str, output_path: str, 
                             similarity_threshold: float, candidate_neg_num: int):
    """
    GPUを使ってTPaNeg用のハードネガティブを事前計算し、キャッシュとして保存します。
    """
    
    # 1. 訓練データを読み込む
    train_path = os.path.join(dataset_path, 'train.pkl')
    print(f"Loading training data from: {train_path}")
    with open(train_path, 'rb') as f:
        data = pickle.load(f)
    q_feats, t_feats, _, q_cats, t_cats, _, q_ids, t_ids = data[:8]

    # 2. 全アイテムの情報をTensorFlow Tensorとして準備
    print("Preparing data structures for TensorFlow on GPU...")
    item_feature_map_np = {}
    item_to_category_map_np = {}
    
    def collect_items(feats, cats, ids):
        for i in range(len(feats)):
            for j, item_id in enumerate(ids[i]):
                if cats[i][j] > 0:
                    item_id_str = str(item_id)
                    if item_id_str not in item_feature_map_np:
                        item_feature_map_np[item_id_str] = feats[i][j].astype(np.float32)
                        item_to_category_map_np[item_id_str] = cats[i][j]

    collect_items(q_feats, q_cats, q_ids)
    collect_items(t_feats, t_cats, t_ids)
    
    unique_item_ids_list = list(item_feature_map_np.keys())
    unique_item_features_list = [item_feature_map_np[item_id] for item_id in unique_item_ids_list]
    unique_item_categories_list = [item_to_category_map_np[item_id] for item_id in unique_item_ids_list]

    all_item_features_tf = tf.constant(np.array(unique_item_features_list), dtype=tf.float32)
    all_item_categories_tf = tf.constant(np.array(unique_item_categories_list), dtype=tf.int32)
    all_item_ids_str_tf = tf.constant(unique_item_ids_list, dtype=tf.string)
    
    item_id_str_to_index_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(all_item_ids_str_tf, tf.range(tf.shape(all_item_ids_str_tf)[0], dtype=tf.int32)), 
        -1
    )
    
    print(f"Created maps for {len(unique_item_ids_list)} unique items.")

    from collections import Counter
    print("\n--- Category-wise Item Counts (for 'Total Items' column) ---")
    
    # item_to_category_map_np の値（カテゴリID）を集計
    category_counts = Counter(item_to_category_map_np.values())
    
    total_items_in_train = 0
    # 集計結果をソートして表示
    for cat_id, count in sorted(category_counts.items()):
        print(f"  - Category {cat_id}: {count} items")
        total_items_in_train += count
    print(f"Total unique items in training set: {total_items_in_train}\n")

    # 3. GPU計算のコアロジック
    @tf.function
    def _compute_hard_negs_batch_gpu(batch_target_item_ids_str_tf, 
                                    global_all_item_features_tf, 
                                    global_all_item_categories_tf,
                                    global_item_id_str_to_index_table,
                                    global_all_item_ids_str_tf,
                                    current_similarity_threshold_tf, 
                                    max_candidate_neg_num_tf):
        
        batch_size = tf.shape(batch_target_item_ids_str_tf)[0]
        
        # RaggedTensorを構築するためのTensorArray
        hard_neg_ids_flat_values_ta = tf.TensorArray(tf.string, size=0, dynamic_size=True, clear_after_read=False, element_shape=())
        hard_neg_row_lengths_ta = tf.TensorArray(tf.int32, size=batch_size, dynamic_size=False, clear_after_read=False)
        
        current_flat_idx = tf.constant(0, dtype=tf.int32)

        for i in tf.range(batch_size):
            item_id_str = batch_target_item_ids_str_tf[i]
            item_idx_in_all = global_item_id_str_to_index_table.lookup(item_id_str)
            
            # アイテムが見つからない場合はスキップ
            if item_idx_in_all == -1:
                hard_neg_row_lengths_ta = hard_neg_row_lengths_ta.write(i, 0)
                continue 

            features_norm = tf.linalg.normalize(global_all_item_features_tf[item_idx_in_all], axis=-1)[0]
            item_cat = global_all_item_categories_tf[item_idx_in_all]

            is_same_category_mask = (global_all_item_categories_tf == item_cat)
            is_not_self_mask = (tf.range(tf.shape(global_all_item_features_tf)[0], dtype=tf.int32) != item_idx_in_all)
            candidate_mask = tf.logical_and(is_same_category_mask, is_not_self_mask)
            
            candidate_feats_norm = tf.linalg.normalize(tf.boolean_mask(global_all_item_features_tf, candidate_mask), axis=-1)[0]
            candidate_ids_str = tf.boolean_mask(global_all_item_ids_str_tf, candidate_mask)

            # 候補がなければスキップ
            if tf.shape(candidate_ids_str)[0] == 0:
                hard_neg_row_lengths_ta = hard_neg_row_lengths_ta.write(i, 0)
                continue 

            similarities = tf.reduce_sum(tf.expand_dims(features_norm, 0) * candidate_feats_norm, axis=1)
            hard_neg_indices = tf.where(similarities >= current_similarity_threshold_tf)[:, 0]
            
            # ハードネガティブが見つからなかった場合はスキップ
            if tf.shape(hard_neg_indices)[0] == 0:
                hard_neg_row_lengths_ta = hard_neg_row_lengths_ta.write(i, 0)
                continue

            # ❌ 削除: 類似度順ソート
            # hard_neg_sims = tf.gather(similarities, hard_neg_indices)
            # sorted_indices_in_hard_pool = tf.argsort(hard_neg_sims, direction='DESCENDING')
            # num_to_sample = tf.minimum(tf.shape(sorted_indices_in_hard_pool)[0], max_candidate_neg_num_tf)
            # final_indices_to_gather = tf.gather(hard_neg_indices, sorted_indices_in_hard_pool[:num_to_sample])

            # ✅ 修正: 一様分布でサンプリング
            if tf.shape(hard_neg_indices)[0] > max_candidate_neg_num_tf:
                # ランダムサンプリング
                shuffled_indices = tf.random.shuffle(hard_neg_indices)
                final_indices_to_gather = shuffled_indices[:max_candidate_neg_num_tf]
            else:
                final_indices_to_gather = hard_neg_indices
            
            final_hard_neg_item_ids_str = tf.gather(candidate_ids_str, final_indices_to_gather)
            
            # ▼▼▼▼▼▼ ここからが重要な修正 ▼▼▼▼▼▼
            num_negs_found = tf.shape(final_hard_neg_item_ids_str, out_type=tf.int32)[0]
            
            # forループを使ってスカラー値を一つずつ書き込む
            for j in tf.range(num_negs_found):
                scalar_id = final_hard_neg_item_ids_str[j]
                hard_neg_ids_flat_values_ta = hard_neg_ids_flat_values_ta.write(current_flat_idx, scalar_id)
                current_flat_idx += 1
            
            hard_neg_row_lengths_ta = hard_neg_row_lengths_ta.write(i, num_negs_found)
            # ▲▲▲▲▲▲ ここまでが重要な修正 ▲▲▲▲▲▲
        
        # RaggedTensor を返す
        return tf.RaggedTensor.from_row_lengths(
            values=hard_neg_ids_flat_values_ta.stack(),
            row_lengths=hard_neg_row_lengths_ta.stack() # row_lengthsは必ず整数型
        )

    # 4. メインループで各バッチを処理
    print("Starting GPU computation for negative cache...")
    final_negative_cache_dict = {}

    similarity_threshold_tf = tf.constant(similarity_threshold, dtype=tf.float32)
    candidate_neg_num_tf = tf.constant(candidate_neg_num, dtype=tf.int32)
    
    dataset = tf.data.Dataset.from_tensor_slices(all_item_ids_str_tf).batch(128)
    pbar = tqdm(total=tf.data.experimental.cardinality(dataset).numpy(), desc="Generating cache on GPU")

    for batch_ids_str in dataset:
        ragged_neg_ids = _compute_hard_negs_batch_gpu(
            batch_ids_str, all_item_features_tf, all_item_categories_tf,
            item_id_str_to_index_table, all_item_ids_str_tf,
            similarity_threshold_tf, candidate_neg_num_tf
        )
        
        # 結果をPythonの辞書に変換
        for i in range(tf.shape(batch_ids_str)[0]):
            item_id = batch_ids_str[i].numpy().decode('utf-8')
            neg_ids = [neg_id.numpy().decode('utf-8') for neg_id in ragged_neg_ids[i]]
            final_negative_cache_dict[item_id] = neg_ids
        pbar.update(1)
        
    pbar.close()

    # 5. 最終的なキャッシュファイルを保存
    final_cache_data = {
        'item_feature_map': item_feature_map_np,
        'negative_cache': final_negative_cache_dict
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(final_cache_data, f)

    print(f"✅ Successfully created new cache file at: {output_path}")
    print(f"Found hard negatives for {len(final_negative_cache_dict)} items.")

# このスクリプトが単独で実行された場合の処理（run.pyからは呼び出されない）
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-compute hard negatives for TPaNeg on GPU.")
    parser.add_argument('--dataset', default='IQON3000', help="Dataset name")
    parser.add_argument('--dataset_dir', default='./datasets', help="Root directory for datasets")
    parser.add_argument('--output_dir', required=True, help="Directory to save the cache file")
    parser.add_argument('--candidate_neg_num', type=int, default=50, help='Number of candidate negatives for TPaNeg')
    parser.add_argument('--pa_neg_epsilon', type=float, default=0.1, help='Margin epsilon for Prediction-Aware Negative selection')

    args = parser.parse_args()

    # run.py と同じ出力ディレクトリ構造を生成
    output_base_dir = os.path.join("experiments", args.dataset, f"batch128_L2_H2_LR0.0001_use_tpaneg_seed{42}") # ここはrun.pyのseedに合わせる
    os.makedirs(output_base_dir, exist_ok=True)
    cache_file_path = os.path.join(output_base_dir, 'hard_negative_cache.pkl')
    
    precompute_negatives_gpu(dataset_path=os.path.join(args.dataset_dir, args.dataset), 
                             output_path=cache_file_path, 
                             similarity_threshold=args.pa_neg_epsilon,
                             candidate_neg_num=args.candidate_neg_num)