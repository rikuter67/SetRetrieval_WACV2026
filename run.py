#!/usr/bin/env python3
"""
SetRetrieval - Unified Set Retrieval Framework
統一された家具・ファッションアイテムセット検索フレームワーク

Usage:
    python run.py --dataset DeepFurniture --mode train --config configs/DeepFurniture_config.yaml
    python run.py --dataset IQON3000 --mode test --config configs/IQON3000_config.yaml
    python run.py --dataset IQON3000 --mode train --batch-size 64 --epochs 200
"""

import argparse
import os
import sys
import random
import time
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import yaml

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from data_generator import DataGenerator
from models import SetRetrievalModel
from util import main_evaluation_pipeline, append_dataframe_to_csv
from plot import plot_training_metrics

# 再現性の確保
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# GPU最適化設定
os.environ.update({
    'TF_DETERMINISTIC_OPS': '0',
    'TF_ENABLE_ONEDNN_OPTS': '0',
    'TF_NUM_INTEROP_THREADS': '1',
    'TF_NUM_INTRAOP_THREADS': '1',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1'
})

def setup_logging(output_dir: Path, verbose: bool = False):
    """ログ設定"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path: str) -> Dict[str, Any]:
    """設定ファイル読み込み"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load config {config_path}: {e}")
        sys.exit(1)

def get_default_config(dataset: str) -> Dict[str, Any]:
    """デフォルト設定を返す"""
    base_config = {
        'model': {
            'embedding_dim': 512,
            'num_heads': 8,
            'num_layers': 6,
            'ff_dim': 512,
            'dropout': 0.1
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'epochs': 100,
            'patience': 10,
            'use_cycle_loss': True,
            'cycle_lambda': 0.2,
            'use_clneg_loss': True,
            'use_center_base': True
        },
        'data': {
            'max_items_per_set': 10,
            'min_items_per_set': 4,
            'test_size': 0.1,
            'val_size': 0.1
        }
    }
    
    if dataset == 'DeepFurniture':
        base_config['dataset'] = {
            'name': 'DeepFurniture',
            'feature_dim': 256,
            'num_categories': 11
        }
        base_config['data'].update({
            'train_pkl': 'data/DeepFurniture/processed/train.pkl',
            'val_pkl': 'data/DeepFurniture/processed/validation.pkl',
            'test_pkl': 'data/DeepFurniture/processed/test.pkl',
            'center_pkl': 'data/DeepFurniture/processed/category_centers.pkl.gz'
        })
    elif dataset == 'IQON3000':
        base_config['dataset'] = {
            'name': 'IQON3000',
            'feature_dim': 512,
            'num_categories': 11
        }
        base_config['data'].update({
            'train_pkl': 'data/IQON3000/processed/train.pkl',
            'val_pkl': 'data/IQON3000/processed/validation.pkl',
            'test_pkl': 'data/IQON3000/processed/test.pkl',
            'center_pkl': 'data/IQON3000/processed/category_centers.pkl.gz'
        })
    
    return base_config

def configure_gpu() -> bool:
    """GPU設定"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.enable_tensor_float_32_execution(True)
            logging.info(f"Configured {len(gpus)} GPU(s)")
            return True
        except RuntimeError as e:
            logging.warning(f"GPU config error: {e}")
    else:
        logging.info("No GPU detected, using CPU")
    return False

def parse_args() -> argparse.Namespace:
    """コマンドライン引数解析"""
    parser = argparse.ArgumentParser(
        description="SetRetrieval: Unified Set Retrieval Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基本設定
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['DeepFurniture', 'IQON3000'],
                        help='使用するデータセット')
    parser.add_argument('--mode', choices=['train', 'test', 'evaluate'], 
                        default='train', help='実行モード')
    parser.add_argument('--config', type=str, default=None,
                        help='設定ファイルパス (省略時はデフォルト設定)')
    
    # 出力設定
    parser.add_argument('--output-dir', type=str, default=None,
                        help='出力ディレクトリ (自動生成される場合あり)')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='実験名 (出力ディレクトリ名に使用)')
    
    # モデル設定 (設定ファイルを上書き)
    parser.add_argument('--embedding-dim', type=int, default=None)
    parser.add_argument('--num-layers', type=int, default=None)
    parser.add_argument('--num-heads', type=int, default=None)
    parser.add_argument('--ff-dim', type=int, default=None)
    
    # 学習設定 (設定ファイルを上書き)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    
    # 損失設定
    parser.add_argument('--use-cycle-loss', action='store_true', default=None)
    parser.add_argument('--no-cycle-loss', action='store_true')
    parser.add_argument('--cycle-lambda', type=float, default=None)
    parser.add_argument('--use-clneg-loss', action='store_true', default=None)
    parser.add_argument('--no-clneg-loss', action='store_true')
    parser.add_argument('--use-center-base', action='store_true', default=None)
    parser.add_argument('--no-center-base', action='store_true')
    
    # テスト設定
    parser.add_argument('--weights-path', type=str, default=None,
                        help='テスト時に使用する重みファイルパス')
    parser.add_argument('--hard-negative-threshold', type=float, default=0.9)
    
    # その他
    parser.add_argument('--gpu', type=int, default=None,
                        help='使用するGPU ID')
    parser.add_argument('--verbose', action='store_true',
                        help='詳細ログ出力')
    
    return parser.parse_args()

def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """設定ファイルとコマンドライン引数をマージ"""
    
    # モデル設定
    if args.embedding_dim is not None:
        config['model']['embedding_dim'] = args.embedding_dim
    if args.num_layers is not None:
        config['model']['num_layers'] = args.num_layers
    if args.num_heads is not None:
        config['model']['num_heads'] = args.num_heads
    if args.ff_dim is not None:
        config['model']['ff_dim'] = args.ff_dim
    
    # 学習設定
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.patience is not None:
        config['training']['patience'] = args.patience
    
    # 損失設定
    if args.use_cycle_loss:
        config['training']['use_cycle_loss'] = True
    elif args.no_cycle_loss:
        config['training']['use_cycle_loss'] = False
    
    if args.cycle_lambda is not None:
        config['training']['cycle_lambda'] = args.cycle_lambda
    
    if args.use_clneg_loss:
        config['training']['use_clneg_loss'] = True
    elif args.no_clneg_loss:
        config['training']['use_clneg_loss'] = False
    
    if args.use_center_base:
        config['training']['use_center_base'] = True
    elif args.no_center_base:
        config['training']['use_center_base'] = False
    
    return config

def create_output_dir(args: argparse.Namespace, config: Dict[str, Any]) -> Path:
    """出力ディレクトリを作成"""
    if args.output_dir:
        return Path(args.output_dir)
    
    # 自動生成
    dataset_name = args.dataset
    
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        # 実験名を自動生成
        exp_name = "exp"
        
        # モデル設定
        exp_name += f"_dim{config['model']['embedding_dim']}"
        exp_name += f"_l{config['model']['num_layers']}"
        exp_name += f"_h{config['model']['num_heads']}"
        
        # 損失設定
        if config['training']['use_center_base']:
            exp_name += "_CB"
        if config['training']['use_clneg_loss']:
            exp_name += "_CLNeg"
        if config['training']['use_cycle_loss']:
            lambda_val = config['training']['cycle_lambda']
            exp_name += f"_Cycle{lambda_val}"
    
    return Path("experiments") / dataset_name / exp_name

def prepare_data_generators(config: Dict[str, Any]) -> Tuple[DataGenerator, DataGenerator, DataGenerator]:
    """データジェネレータを準備"""
    logging.info("Loading data generators...")
    start_time = time.time()
    
    data_config = config['data']
    batch_size = config['training']['batch_size']
    
    # パスの存在確認
    for key in ['train_pkl', 'val_pkl', 'test_pkl']:
        path = data_config[key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
    
    train_gen = DataGenerator(
        data_config['train_pkl'], 
        batch_size=batch_size, 
        shuffle=True, 
        seed=SEED, 
        center_path=data_config['center_pkl']
    )
    
    val_gen = DataGenerator(
        data_config['val_pkl'], 
        batch_size=batch_size, 
        shuffle=False, 
        seed=SEED, 
        center_path=data_config['center_pkl']
    )
    
    test_gen = DataGenerator(
        data_config['test_pkl'], 
        batch_size=batch_size, 
        shuffle=False, 
        seed=SEED, 
        center_path=data_config['center_pkl']
    )
    
    load_time = time.time() - start_time
    logging.info(f"Data loaded in {load_time:.2f}s: "
                f"Train={len(train_gen)}, Val={len(val_gen)}, Test={len(test_gen)} batches")
    
    return train_gen, val_gen, test_gen

def create_model(config: Dict[str, Any], train_gen: DataGenerator) -> SetRetrievalModel:
    """モデルを作成"""
    logging.info("Creating model...")
    
    model_config = config['model']
    training_config = config['training']
    
    model = SetRetrievalModel(
        dim=model_config['embedding_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        ff_dim=model_config['ff_dim'],
        cycle_lambda=training_config['cycle_lambda'],
        use_cycle_loss=training_config['use_cycle_loss'],
        use_CLNeg_loss=training_config['use_clneg_loss'],
        use_center_base=training_config['use_center_base']
    )
    
    model.set_cluster_center(train_gen.cluster_centers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(training_config['learning_rate']),
        loss=None,
        metrics=[model.train_avg_rank, model.val_avg_rank]
    )
    
    # モデルを構築
    dummy_input = tf.zeros((training_config['batch_size'], train_gen.max_item_num, model_config['embedding_dim']))
    _ = model.forward_pass(dummy_input)
    
    logging.info("Model built successfully")
    model.display_parameter_summary()
    
    return model

def create_checkpoint_path(output_dir: Path, config: Dict[str, Any]) -> Path:
    """チェックポイントパス作成"""
    training_config = config['training']
    
    flags = [
        f"bs{training_config['batch_size']}",
        f"ep{training_config['epochs']}"
    ]
    
    if training_config['use_cycle_loss']:
        flags.append(f"cy{training_config['cycle_lambda']}")
    if training_config['use_clneg_loss']:
        flags.append("clneg")
    if training_config['use_center_base']:
        flags.append("cb")
    
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    return checkpoint_dir / f"best_model_{'_'.join(flags)}.weights.h5"

def train_model(model: SetRetrievalModel, 
                train_gen: DataGenerator, 
                val_gen: DataGenerator,
                config: Dict[str, Any], 
                output_dir: Path) -> tf.keras.callbacks.History:
    """モデル学習"""
    training_config = config['training']
    checkpoint_path = create_checkpoint_path(output_dir, config)
    
    logging.info(f"Training for {training_config['epochs']} epochs")
    logging.info(f"Checkpoint will be saved to: {checkpoint_path}")
    
    callbacks = [
        ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_val_avg_rank',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_val_avg_rank',
            factor=0.5,
            patience=training_config['patience'],
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_val_avg_rank',
            patience=training_config['patience'],
            mode='min',
            verbose=1,
            restore_best_weights=True
        )
    ]
    
    start_time = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=training_config['epochs'],
        callbacks=callbacks,
        shuffle=False
    )
    
    training_time = time.time() - start_time
    logging.info(f"Training completed in {training_time:.2f}s "
                f"({training_time/training_config['epochs']:.2f}s/epoch)")
    
    # 学習曲線を保存
    plot_training_metrics(
        output_dir=str(output_dir),
        batch_size=training_config['batch_size'],
        epochs=list(history.epoch),
        train_loss=history.history['loss'],
        val_loss=history.history.get('val_loss', []),
        train_avg_rank=history.history.get("train_avg_rank", []),
        val_avg_rank=history.history.get('val_val_avg_rank', [])
    )
    
    return history

def test_model(model: SetRetrievalModel, 
               test_gen: DataGenerator,
               config: Dict[str, Any], 
               output_dir: Path,
               weights_path: str = None,
               hard_negative_threshold: float = 0.9):
    """モデルテスト"""
    # 重みの読み込み
    if weights_path:
        weights_file = Path(weights_path)
    else:
        weights_file = create_checkpoint_path(output_dir, config)
    
    if not weights_file.exists():
        raise FileNotFoundError(f"Weights not found: {weights_file}")
    
    logging.info(f"Loading weights: {weights_file}")
    model.load_weights(str(weights_file))
    
    # 評価実行
    logging.info("Running comprehensive evaluation...")
    main_evaluation_pipeline(
        model=model,
        test_generator=test_gen,
        output_dir=str(output_dir),
        checkpoint_path=str(weights_file),
        hard_negative_threshold=hard_negative_threshold,
        top_k_percentages=[1, 3, 5, 10, 20],
        combine_directions=True,
        enable_visualization=False
    )
    
    logging.info("Evaluation completed")

def save_config(config: Dict[str, Any], output_dir: Path):
    """設定をファイルに保存"""
    config_path = output_dir / 'config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    logging.info(f"Configuration saved to: {config_path}")

def main():
    """メイン関数"""
    args = parse_args()
    
    # GPU設定
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    gpu_available = configure_gpu()
    
    # 設定読み込み
    if args.config:
        config = load_config(args.config)
        logging.info(f"Loaded config from: {args.config}")
    else:
        config = get_default_config(args.dataset)
        logging.info(f"Using default config for dataset: {args.dataset}")
    
    # コマンドライン引数で設定を上書き
    config = merge_config_with_args(config, args)
    
    # 出力ディレクトリ作成
    output_dir = create_output_dir(args, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ログ設定
    setup_logging(output_dir, args.verbose)
    
    logging.info(f"SetRetrieval Framework - {args.mode.upper()} mode")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Output directory: {output_dir}")
    
    # 設定を保存
    save_config(config, output_dir)
    
    try:
        # データ準備
        train_gen, val_gen, test_gen = prepare_data_generators(config)
        
        # モデル作成
        model = create_model(config, train_gen)
        
        # モード別実行
        if args.mode == 'train':
            history = train_model(model, train_gen, val_gen, config, output_dir)
            logging.info("Training completed successfully")
            
            # 学習後に自動テスト実行
            logging.info("Running post-training evaluation...")
            test_model(model, test_gen, config, output_dir, 
                      hard_negative_threshold=args.hard_negative_threshold)
            
        elif args.mode == 'test':
            test_model(model, test_gen, config, output_dir, 
                      args.weights_path, args.hard_negative_threshold)
            
        elif args.mode == 'evaluate':
            test_model(model, test_gen, config, output_dir, 
                      args.weights_path, args.hard_negative_threshold)
        
        logging.info("SetRetrieval execution completed successfully!")
        
    except Exception as e:
        logging.error(f"Execution failed: {e}")
        raise

if __name__ == '__main__':
    main()