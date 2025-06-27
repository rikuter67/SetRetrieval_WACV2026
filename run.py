"""
run.py CUDA_VISIBLE_DEVICES=0 python run.py --dataset DeepFurniture --mode train --batch_size 128 --epochs 100 --num_layers 2 --num_heads 2 --learning_rate 1e-4  --use_cycle_loss  --cycle_lambda 0.2
"""
import os
import argparse
import json
import pickle
import gzip
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# from models import create_model, DebugCallback, ProgressTrackingCallback
from util import evaluate_model, setup_gpu_memory
from config import get_dataset_config
from plot import plot_training_curves  # 既存の可視化関数を使用
from models import SetRetrievalModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def create_output_dir(args: argparse.Namespace) -> str:
    """Creates an informative output directory path based on experiment parameters."""
    if args.output_dir:
        return args.output_dir

    components = [
        f"B{args.batch_size}",
        f"L{args.num_layers}",
        f"H{args.num_heads}",
        f"LR{args.learning_rate:.0e}",
        f"Cycle{args.cycle_lambda}"
    ]
    experiment_name = "_".join(components)
    return os.path.join("experiments", args.dataset, experiment_name)

def detect_num_categories(dataset_path: str) -> int:
    """Detects the number of categories from the dataset configuration."""
    config = get_dataset_config(os.path.basename(dataset_path))
    return config.get('num_categories', 7)

def load_data(dataset_path: str, split: str, batch_size: int) -> tf.data.Dataset:
    """Loads data from .pkl file and creates a tf.data.Dataset."""
    file_path = os.path.join(dataset_path, f'{split}.pkl')
    print(f"Loading data from: {file_path}")
    with open(file_path, 'rb') as f:
        data_tuple = pickle.load(f)

    data_dict = {
        'query_features': np.array(data_tuple[0], dtype=np.float32),
        'target_features': np.array(data_tuple[1], dtype=np.float32),
        'query_categories': np.array(data_tuple[3], dtype=np.int32),
        'target_categories': np.array(data_tuple[4], dtype=np.int32),
    }

    if split == 'test':
        data_dict['query_item_ids'] = np.array(data_tuple[6], dtype=str)
        data_dict['target_item_ids'] = np.array(data_tuple[7], dtype=str)

    dataset = tf.data.Dataset.from_tensor_slices(data_dict)

    if split == 'train':
        dataset = dataset.shuffle(buffer_size=1000, seed=42)

    is_training = (split == 'train')
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def main():
    parser = argparse.ArgumentParser(description="Set Retrieval Training with Built-in TopK Metrics")
    parser.add_argument('--dataset', default='IQON3000', choices=['IQON3000', 'DeepFurniture'], help="Dataset to use.")
    parser.add_argument('--mode', default='train', choices=['train', 'test'], help="Mode to run: 'train' or 'test'.")
    parser.add_argument('--data_dir', default='datasets', help="Root directory for datasets.")
    parser.add_argument('--output_dir', default=None, help="Specify a custom output directory.")
    # Model and training hyperparameters
    parser.add_argument('--model_path', default=None, help="Path to pre-trained model weights for testing.")
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.2, help="Temperature for contrastive loss")
    parser.add_argument('--dropout_rate', type=float, default=0.1, help="Dropout rate for regularization")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--topk_values', type=int, nargs='+', default=[1, 5, 10, 20], help="TopK values to track.")
    # Optional loss components
    parser.add_argument('--use_cycle_loss', action='store_true', help='Include cycle consistency loss')
    parser.add_argument('--cycle_lambda',   type=float,        default=0.0,  help='Weight for cycle consistency loss')
    parser.add_argument('--use_clneg_loss', action='store_true', help='Include contrastive negative loss')

    args = parser.parse_args()

    # シード設定
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    # 出力ディレクトリ作成
    output_dir = create_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📦 Output will be saved to: {output_dir}")

    # 引数保存
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    setup_gpu_memory()

    # データセット設定
    dataset_path = os.path.join(args.data_dir, args.dataset)
    num_categories = detect_num_categories(dataset_path)
    
    model = SetRetrievalModel(
        feature_dim=args.feature_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_categories=num_categories,
        temperature=args.temperature,
        dropout_rate=args.dropout_rate,
        use_cycle_loss=args.use_cycle_loss,
        cycle_lambda=args.cycle_lambda,
        k_values=args.topk_values
    )


    # カテゴリ中心設定
    centers_path = os.path.join(dataset_path, 'category_centers.pkl.gz')
    if os.path.exists(centers_path):
        with gzip.open(centers_path, 'rb') as f: 
            centers_dict = pickle.load(f)
        centers_array = np.zeros((num_categories, args.feature_dim), dtype=np.float32)
        for cat_id, center in centers_dict.items():
            if 1 <= int(cat_id) <= num_categories: 
                centers_array[int(cat_id) - 1] = center
        model.set_category_centers(centers_array)

    # モデル初期化
    dummy_input = {'query_features': tf.random.normal((2, 10, 512))}
    _ = model(dummy_input)
    model.summary()

    model_path_for_testing = args.model_path

    if args.mode == 'train':
        print(f"\n--- 🚂 Starting Training with TopK={args.topk_values} ---")
        train_data = load_data(dataset_path, 'train', args.batch_size)
        val_data = load_data(dataset_path, 'validation', args.batch_size)
        
        # オプティマイザ設定
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss=None, run_eagerly=False)
        
        # サンプルバッチを取得
        sample_batch = next(iter(train_data))

        # デバッグコールバックを追加
        # debug_callback = DebugCallback(sample_batch)
        # progress_callback = ProgressTrackingCallback(sample_batch, track_every=5)

        # コールバック設定（シンプル）
        callbacks = [
            # debug_callback,
            # progress_callback,
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)
        ]
        
        print(f"[INFO] TopK metrics will be tracked automatically: {args.topk_values}")
        
        # 学習実行（TopKメトリックは自動で計算される）
        history = model.fit(
            train_data, 
            validation_data=val_data, 
            epochs=args.epochs, 
            callbacks=callbacks,
            verbose=1  # TopK正解率がリアルタイムで表示される
        )

        # モデル保存
        model_path_for_testing = os.path.join(output_dir, 'final_model.weights.h5')
        model.save_weights(model_path_for_testing)
        print(f"\n💾 Model saved to: {model_path_for_testing}")

        # 可視化（既存のplot_training_curvesを使用）
        print("\n--- 📊 Creating Training Visualization ---")
        
        try:
            # 元の関数の代わりに修正版を使用
            plot_training_curves(history.history, output_dir, args.dataset)
            print("✅ Training visualization complete!")
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")
            # デバッグ情報
            print(f"Available keys: {list(history.history.keys())}")

    # モデルパス設定
    if not model_path_for_testing:
        model_path_for_testing = os.path.join(output_dir, 'final_model.weights.h5')

    # 評価実行
    print("\n--- 🧪 Starting Evaluation ---")
    if os.path.exists(model_path_for_testing):
        model.load_weights(model_path_for_testing)
        test_data = load_data(dataset_path, 'test', args.batch_size)

        # 評価実行
        evaluate_model(model, test_data, output_dir, args.data_dir)
        
        print(f"\n🎉 Training and evaluation completed!")
        print(f"📁 All results saved to: {output_dir}")
        
        # 最終結果の表示
        if args.mode == 'train':
            print(f"\n📈 Final TopK Results:")
            for k in args.topk_values:
                if f'top{k}_accuracy' in history.history and f'val_top{k}_accuracy' in history.history:
                    final_train = history.history[f'top{k}_accuracy'][-1] * 100
                    final_val = history.history[f'val_top{k}_accuracy'][-1] * 100
                    best_train = max(history.history[f'top{k}_accuracy']) * 100
                    best_val = max(history.history[f'val_top{k}_accuracy']) * 100
                    print(f"   Top-{k}: Train={final_train:.1f}% (best={best_train:.1f}%), "
                          f"Val={final_val:.1f}% (best={best_val:.1f}%)")
                else:
                    print(f"   Top-{k}: データが見つかりませんでした")
    else:
        print(f"❌ Model weights not found: {model_path_for_testing}")

if __name__ == "__main__":
    main()