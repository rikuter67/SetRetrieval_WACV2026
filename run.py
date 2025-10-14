import os
import argparse
import json
import pickle
import gzip
import numpy as np
import random
import time
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data_generator import DataGenerator
from helpers import build_image_path_map
from util import evaluate_model, setup_gpu_memory, collect_test_data, save_whitening_params, load_whitening_params, compute_input_whitening_stats, HardNegativeMiner, save_hard_negative_cache, load_hard_negative_cache
from config import get_dataset_config
from plot import plot_training_curves, generate_qualitative_examples
from models import SetRetrieval
from precompute_negatives import precompute_negatives_gpu 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

tf.config.optimizer.set_jit(False)
tf.config.experimental.enable_op_determinism()

class TimeHistory(tf.keras.callbacks.Callback):
    """Callback that measures learning time (per epoch) and outputs the average at the end"""
    def on_train_begin(self, logs={}):
        self.epoch_times = []
        print("⏱️  Training time measurement started.")

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = time.perf_counter()

    def on_epoch_end(self, epoch, logs={}):
        elapsed_seconds = time.perf_counter() - self.epoch_start_time
        # The first epoch is excluded from the average calculation.
        if epoch > 0:
            self.epoch_times.append(elapsed_seconds)
        print(f"Epoch {epoch+1} Time: {elapsed_seconds:.2f} seconds")

    def on_train_end(self, logs={}):
        if not self.epoch_times:
            print("⚠️ No epoch times were recorded (training might have been too short).")
            return

        avg_seconds = np.mean(self.epoch_times)
        avg_minutes = avg_seconds / 60
        print("\n" + "="*40)
        print("Training Time Summary")
        print(f"   Average time per epoch: {avg_seconds:.2f} seconds")
        print(f"   Average time per epoch: {avg_minutes:.4f} minutes")
        print("="*40)

        
class EpochUpdateCallback(tf.keras.callbacks.Callback):
    """エポック開始時にDataGeneratorのepochを更新するためのコールバック"""
    def __init__(self, data_generator, model=None):  # modelパラメータを追加
        super().__init__()
        self.data_generator = data_generator
        self.tpaneg_model = model  # modelを保存

    def on_epoch_begin(self, epoch, logs=None):
        if hasattr(self.data_generator, 'set_epoch'):
            self.data_generator.set_epoch(epoch)
        
        if self.tpaneg_model and hasattr(self.tpaneg_model, 'set_current_epoch'):
            self.tpaneg_model.set_current_epoch(epoch)

def create_output_dir(args: argparse.Namespace) -> str:
    """実験設定に基づいて出力ディレクトリのパスを生成する"""
    if args.output_dir:
        return args.output_dir
        
    components = [
        f"Epoch{args.epochs}"
        f"batch{args.batch_size}",
        f"L{args.num_layers}",
        f"H{args.num_heads}",
        f"LR{args.learning_rate}"
    ]
    
    components.append(f"primary{args.primary_loss}")

    if args.use_whitening:
        components.append("use_whitening")
    if args.use_tpaneg:
        components.append("use_tpaneg")
    if args.use_cycle_loss:
        components.append("use_cycle_loss")
        components.append(f"lambda{args.cycle_lambda}")

    if args.style_loss_weight > 0:
        components.append(f"styleW{args.style_loss_weight:g}")
        components.append(f"styleMode{args.style_loss_mode}")

    if args.use_tpaneg:
        components.append(f"TaNegInit{args.taneg_t_gamma_init}")
        components.append(f"TaNegFinal{args.taneg_t_gamma_final}")
        components.append(f"TaNegEps{args.taneg_curriculum_epochs}")
        components.append(f"PaNegEps{args.paneg_epsilon}")

    components.append(f"seed{args.seed}")
    
    experiment_name = "_".join(components)
    output_dir = os.path.join("experiments", args.dataset, experiment_name)
    return output_dir


def detect_num_categories(dataset_path: str) -> int:
    config = get_dataset_config(os.path.basename(dataset_path))
    return config.get('num_categories', 16)

def create_data_generators(dataset_path: str, batch_size: int,
                           use_negatives: bool = False, candidate_neg_num: int = 50,
                           seed: int = 42, random_split: bool = True,
                           whitening_params: dict = None,
                           negative_cache_path: str = None) -> tuple:
    
    train_gen = DataGenerator(
        split_path=os.path.join(dataset_path, 'train.pkl'),
        batch_size=batch_size, shuffle=True, seed=seed,
        use_negatives=use_negatives, negative_cache_path=negative_cache_path,
        candidate_neg_num=candidate_neg_num, random_split=random_split,
        whitening_params=whitening_params
    )
    
    val_gen = DataGenerator(
        split_path=os.path.join(dataset_path, 'validation.pkl'),
        batch_size=batch_size, shuffle=False, seed=seed,
        use_negatives=False,  # ✅ 検証時はTPaNeg無効
        random_split=False,
        whitening_params=whitening_params
    )
    
    test_gen = DataGenerator(
        split_path=os.path.join(dataset_path, 'test.pkl'),
        batch_size=batch_size, shuffle=False, seed=seed,
        use_negatives=False, random_split=False, # テスト時は通常、フルギャラリー検索のためネガティブは不要
        whitening_params=whitening_params,
        include_set_ids=True
    )

    def create_tf_dataset(data_gen: DataGenerator, is_training: bool) -> tf.data.Dataset:
        def generator_fn():
            for i in range(len(data_gen)):
                yield data_gen[i]

        output_signature = {
            'query_features': tf.TensorSpec(shape=(None, data_gen.max_item_num, data_gen.feature_dim), dtype=tf.float32),
            'target_features': tf.TensorSpec(shape=(None, data_gen.max_item_num, data_gen.feature_dim), dtype=tf.float32),
            'query_categories': tf.TensorSpec(shape=(None, data_gen.max_item_num), dtype=tf.int32),
            'target_categories': tf.TensorSpec(shape=(None, data_gen.max_item_num), dtype=tf.int32),
            'query_item_ids': tf.TensorSpec(shape=(None, data_gen.max_item_num), dtype=tf.int32),
            'target_item_ids': tf.TensorSpec(shape=(None, data_gen.max_item_num), dtype=tf.int32),
        }
        
        if is_training and data_gen.use_negatives:
            # 論文準拠のダブル双方向損失のため、クエリ用のネガティブも定義
            output_signature['candidate_negative_features'] = tf.TensorSpec(
                shape=(None, data_gen.max_item_num, data_gen.candidate_neg_num, data_gen.feature_dim), dtype=tf.float32)
            output_signature['candidate_negative_masks'] = tf.TensorSpec(
                shape=(None, data_gen.max_item_num, data_gen.candidate_neg_num), dtype=tf.bool)
            output_signature['query_candidate_negative_features'] = tf.TensorSpec(
                shape=(None, data_gen.max_item_num, data_gen.candidate_neg_num, data_gen.feature_dim), dtype=tf.float32)
            output_signature['query_candidate_negative_masks'] = tf.TensorSpec(
                shape=(None, data_gen.max_item_num, data_gen.candidate_neg_num), dtype=tf.bool)

        if not is_training and data_gen.include_SetIDs:
            output_signature['set_ids'] = tf.TensorSpec(shape=(None,), dtype=tf.string)

        dataset = tf.data.Dataset.from_generator(generator_fn, output_signature=output_signature)
        return dataset.prefetch(tf.data.AUTOTUNE)
    
    train_dataset = create_tf_dataset(train_gen, is_training=True)
    val_dataset = create_tf_dataset(val_gen, is_training=False)
    test_dataset = create_tf_dataset(test_gen, is_training=False)
    
    train_steps = len(train_gen)
    val_steps = len(val_gen)
    
    return train_dataset, val_dataset, test_dataset, train_gen, val_gen, train_steps, val_steps



def main():
    parser = argparse.ArgumentParser(description="Set Retrieval Training with DataGenerator")
    parser.add_argument('--dataset', default='IQON3000', choices=['IQON3000', 'DeepFurniture'], help="Dataset to use.")
    parser.add_argument('--mode', default='train', choices=['train', 'test'], help="Mode to run: 'train' or 'test'.")
    parser.add_argument('--data_dir', default='data', help="Root directory for data")
    parser.add_argument('--dataset_dir', default='datasets', help="Root directory for datasets.")
    parser.add_argument('--output_dir', default=None, help="Specify a custom output directory.")
    parser.add_argument('--model_path', default=None, help="Path to pre-trained model weights for testing.")

    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.8, help="Temperature for contrastive loss")
    parser.add_argument('--dropout_rate', type=float, default=0.1, help="Dropout rate for regularization")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")

    parser.add_argument('--topk_values', type=int, nargs='+', default=[1, 5, 10, 20], help="TopK values to track.")
    parser.add_argument('--use_weighted_topk', action='store_true', help='Enable weighted TopK accuracy metrics during evaluation')
    parser.add_argument('--use_cluster_centering', action='store_true', help='Cluster center base loss')
    parser.add_argument('--use_whitening', action='store_true', help='Enable whitening transformation during evaluation')

    parser.add_argument('--use_cycle_loss', action='store_true', help='Include cycle consistency loss')
    parser.add_argument('--cycle_lambda', type=float, default=0.2, help='Weight for cycle consistency loss')

    parser.add_argument('--style_loss_weight', type=float, default=0.0,
                        help='Weight for the style loss (set to 0 to disable).')
    parser.add_argument('--style_loss_mode', choices=['gram', 'attention_gram'], default='gram',
                        help='Style loss pattern: gram (図1) or attention_gram (図2).')
    parser.add_argument('--primary_loss', choices=['inbatch', 'tpaneg', 'style'], default=None,
                        help='Primary training loss to optimize. Supports inbatch, tpaneg, or style (図1/図2).')

    parser.add_argument('--use_tpaneg', action='store_true', help='Enable TPaNeg dynamic hard negative learning')
    parser.add_argument('--candidate_neg_num', type=int, default=50, help='Number of candidate negatives for TPaNeg')
    parser.add_argument('--taneg_t_gamma_init', type=float, default=0.2, help='Initial T_gamma (TaNeg similarity threshold). Used for precomputing and as start of curriculum.')
    parser.add_argument('--taneg_t_gamma_final', type=float, default=0.4, help='Final T_gamma (TaNeg similarity threshold). End point of curriculum learning.')
    parser.add_argument('--taneg_curriculum_epochs', type=int, default=100, help='Number of epochs over which TaNeg T_gamma linearly increases.')
    parser.add_argument('--paneg_epsilon', type=float, default=0.2, help='Epsilon (ε) margin for Prediction-Aware Negative selection (PaNeg). Fixed during training.')
    
    args = parser.parse_args()

    if args.primary_loss is None:
        args.primary_loss = 'tpaneg' if args.use_tpaneg else 'inbatch'

    if args.primary_loss == 'tpaneg':
        args.use_tpaneg = True
    else:
        if args.use_tpaneg:
            print(f"[INFO] Disabling --use_tpaneg because primary_loss='{args.primary_loss}'.")
        args.use_tpaneg = False

    if args.primary_loss == 'style' and args.style_loss_weight <= 0.0:
        parser.error("--style_loss_weight must be > 0 when primary_loss is 'style'.")

    model_path_for_testing = args.model_path

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)
    tf.config.optimizer.set_jit(False)

    output_dir = create_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📦 Output will be saved to: {output_dir}")

    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    setup_gpu_memory()

    dataset_path = os.path.join(args.dataset_dir, args.dataset)
    num_categories = detect_num_categories(dataset_path)
    
    whitening_params = None
    if args.use_whitening:
        whitening_params_path = os.path.join(output_dir, 'input_whitening_params.pkl')
        if os.path.exists(whitening_params_path):
            whitening_params = load_whitening_params(whitening_params_path)
            print("[INFO] ✅ Loaded existing input whitening parameters.")
        else:
            whitening_params = compute_input_whitening_stats(dataset_path, args.feature_dim)
            if whitening_params:
                save_whitening_params(whitening_params, whitening_params_path)
                print(f"✅ Input whitening parameters computed and saved to: {whitening_params_path}")

    if args.primary_loss == 'tpaneg':
        cache_path = os.path.join(args.dataset_dir, args.dataset, 'hard_negative_cache.pkl')
        print("Checking for TPaNeg negative cache...")

        if not os.path.exists(cache_path):
            print(f"❌ Cache not found. Generating new cache at: {cache_path}")
            # TaNegのT_gamma初期値でハードネガティブを事前計算
            precompute_negatives_gpu(
                dataset_path=os.path.join(args.dataset_dir, args.dataset),
                output_path=cache_path,
                similarity_threshold=args.taneg_t_gamma_init, # <-- TaNegの初期T_gammaで事前計算
                candidate_neg_num=args.candidate_neg_num
            )
        else:
            print(f"✅ Found existing cache: {cache_path}")

    model = SetRetrieval(
        feature_dim=args.feature_dim, num_heads=args.num_heads, num_layers=args.num_layers, num_categories=num_categories,  hidden_dim=args.hidden_dim,
        temperature=args.temperature, dropout_rate=args.dropout_rate, k_values=args.topk_values, use_cycle_loss=args.use_cycle_loss, cycle_lambda=args.cycle_lambda, cluster_centering=args.use_cluster_centering,
        primary_loss=args.primary_loss, style_loss_weight=args.style_loss_weight, style_loss_mode=args.style_loss_mode,
        use_tpaneg=args.use_tpaneg,taneg_t_gamma_init=args.taneg_t_gamma_init, taneg_t_gamma_final=args.taneg_t_gamma_final, taneg_curriculum_epochs=args.taneg_curriculum_epochs, paneg_epsilon=args.paneg_epsilon
    )

    centers_path = os.path.join(dataset_path, 'category_centers.pkl.gz')
    if os.path.exists(centers_path):
        with gzip.open(centers_path, 'rb') as f: 
            centers_dict = pickle.load(f)
        centers_array = np.zeros((num_categories, args.feature_dim), dtype=np.float32)
        for cat_id, center in centers_dict.items():
            if 1 <= int(cat_id) <= num_categories: 
                centers_array[int(cat_id) - 1] = center
        model.set_category_centers(centers_array, whitening_params=whitening_params)

    dummy_input = {'query_features': tf.random.normal((2, 10, 512))}
    _ = model(dummy_input)
    model.summary()

    train_data, val_data, test_data, train_gen, val_gen, steps_per_epoch, validation_steps = create_data_generators(
        dataset_path,
        args.batch_size, 
        use_negatives=(args.primary_loss == 'tpaneg'),
        candidate_neg_num=args.candidate_neg_num,
        seed=args.seed,
        random_split=True,
        whitening_params=whitening_params,
        negative_cache_path=cache_path if args.primary_loss == 'tpaneg' else None # minerの代わりにcache_pathを渡す
    )

    if args.mode == 'train':
        print(f"\n--- Starting Training with TopK={args.topk_values} ---")

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), 
                         loss=None, 
                         metrics=model.metrics)
        
        callbacks = [
            EarlyStopping(monitor='val_val_top10_accuracy', patience=args.early_stop, mode='max', verbose=1, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_val_top10_accuracy', factor=0.8, patience=15, min_lr=1e-5, mode='min'),
            EpochUpdateCallback(train_gen, model),
            TimeHistory()
        ]
        
        print(f"[INFO] TopK metrics will be tracked automatically: {args.topk_values}")
        
        history = model.fit(
            train_data.repeat(), 
            validation_data=val_data.repeat(), 
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        model_path_for_testing = os.path.join(output_dir, 'final_model.weights.h5')
        model.save_weights(model_path_for_testing)
        print(f"\n Model saved to: {model_path_for_testing}")

        print("\n--- Creating Training Visualization ---")
        try:
            plot_training_curves(history.history, output_dir, args.dataset)
            print("Training visualization complete!")
        except Exception as e:
            print(f"Visualization failed: {e}")
            print(f"Available keys: {list(history.history.keys())}")
            

    if not model_path_for_testing:
        model_path_for_testing = os.path.join(output_dir, 'final_model.weights.h5')

    print("\n--- 🧪 Starting Evaluation ---")
    if os.path.exists(model_path_for_testing):
        model.load_weights(model_path_for_testing)
    
        results_eval, test_items, gallery, dataset_type = evaluate_model(model, test_data, output_dir, args.dataset_dir, use_weighted_topk=args.use_weighted_topk, category_centers=centers_array)

        print("\n--- 🖼️ Creating Image Collages ---")
        collage_dir = os.path.join(output_dir, "collages")
        os.makedirs(collage_dir, exist_ok=True)
        generate_qualitative_examples(
            model=model, test_items=test_items, gallery=gallery,
            image_path_map=build_image_path_map(args.data_dir, args.dataset),
            config=get_dataset_config(args.dataset), output_dir=collage_dir,
            num_examples=500, top_k=10,
            min_target_items=4 
        )
        print(f"Image collages saved to: {collage_dir}")
        print(f"All results saved to: {output_dir}")
    else:
        print(f"❌ Model weights not found: {model_path_for_testing}")

if __name__ == "__main__":
    main()