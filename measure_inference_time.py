# python measure_inference_time.py --model_path "experiments/DeepFurniture/.../final_model.weights.h5" --dataset DeepFurniture --num_categories 11
# python measure_inference_time.py --model_path "experiments/IQON3000/.../final_model.weights.h5" --dataset IQON3000 --num_categories 7

import os
import time
import numpy as np
import tensorflow as tf
import argparse

from models import TPaNegModel
from data_generator import DataGenerator
from util import setup_gpu_memory

def measure_inference(args):
    setup_gpu_memory()

    # 1. 学習済みモデルをロード
    print(f"🚀 Loading model from: {args.model_path}")
    model = TPaNegModel(num_categories=args.num_categories)
    # ダミー入力でモデルをビルド
    dummy_input = {'query_features': tf.random.normal((1, 10, 512))}
    _ = model(dummy_input)
    model.load_weights(args.model_path)
    print("✅ Model loaded successfully.")

    # 2. テストデータを1バッチだけロード
    print("📥 Loading one batch of test data...")
    test_gen = DataGenerator(
        split_path=os.path.join(args.dataset_dir, args.dataset, 'test.pkl'),
        batch_size=args.batch_size,
        shuffle=True # ランダムなサンプルを使うためにTrueに
    )
    test_batch = test_gen[0]
    query_sets = test_batch['query_features']
    print(f"✅ Loaded {len(query_sets)} query sets.")

    # 3. 推論時間を計測
    timings = []
    
    # 最初の1回はウォームアップとして実行（グラフ構築などで遅いため）
    print("🔥 Performing warm-up inference...")
    _ = model.infer_single_set(query_sets[0])
    print("✅ Warm-up complete.")

    print(f"⏱️  Measuring inference time over {args.num_samples} samples...")
    for i in range(min(args.num_samples, len(query_sets))):
        start_time = time.perf_counter()
        _ = model.infer_single_set(query_sets[i])
        end_time = time.perf_counter()
        timings.append((end_time - start_time) * 1000) # ミリ秒に変換

    # 4. 結果を表示
    avg_time = np.mean(timings)
    std_dev = np.std(timings)
    print("\n--- Inference Time Results ---")
    print(f"Average: {avg_time:.4f} ms")
    print(f"Std Dev: {std_dev:.4f} ms")
    print("----------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help="Path to the trained model weights (.h5 file)")
    parser.add_argument('--dataset', required=True, choices=['IQON3000', 'DeepFurniture'])
    parser.add_argument('--dataset_dir', default='./datasets')
    parser.add_argument('--num_categories', type=int, default=11, help="Number of categories (7 for IQON3000, 11 for DeepFurniture)")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_samples', type=int, default=100, help="Number of samples to average over")
    args = parser.parse_args()
    
    measure_inference(args)