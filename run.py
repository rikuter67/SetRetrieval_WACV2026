# #!/usr/bin/env python3
# """
# run.py - Correct Set Retrieval Training Script
# ===============================================
# Query set → Target set prediction with gallery evaluation
# """

# import os
# import sys
# import warnings
# warnings.filterwarnings('ignore')

# # Suppress TensorFlow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# import argparse
# import json
# import pickle
# import gzip
# from datetime import datetime
# import time

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# # Suppress TF warnings
# tf.get_logger().setLevel('ERROR')

# from models import create_model, contrastive_loss_with_negatives
# from util import evaluate_model


# def setup_gpu():
#     """Setup GPU configuration"""
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         try:
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#             print("🚀 GPU setup completed")
#         except RuntimeError as e:
#             print(f"GPU setup error: {e}")


# def load_data(dataset_path, split, batch_size):
#     """Load dataset with correct format"""
#     file_path = os.path.join(dataset_path, f'{split}.pkl')
    
#     try:
#         # Try the tuple format first
#         with open(file_path, 'rb') as f:
#             data_tuple = pickle.load(f)
        
#         if isinstance(data_tuple, (list, tuple)) and len(data_tuple) >= 5:
#             # Format: (query_features, target_features, scene_ids, query_categories, target_categories)
#             query_features, target_features, query_categories, target_categories = data_tuple[0], data_tuple[1], data_tuple[3], data_tuple[4]
#             print(f"📋 Loaded data tuple format: {len(data_tuple)} elements")
#         else:
#             raise ValueError("Unexpected tuple format")
            
#     except (EOFError, ValueError, IndexError):
#         try:
#             # Try the sequential format
#             with open(file_path, 'rb') as f:
#                 query_features = pickle.load(f)
#                 target_features = pickle.load(f)
#                 query_categories = pickle.load(f)
#                 target_categories = pickle.load(f)
#                 query_item_ids = pickle.load(f)
#                 target_item_ids = pickle.load(f)
#             print("📋 Loaded sequential pickle format")
#         except EOFError:
#             raise ValueError(f"Could not read pickle file: {file_path}")
    
#     # Convert to numpy arrays
#     query_features = np.array(query_features, dtype=np.float32)
#     target_features = np.array(target_features, dtype=np.float32)
#     query_categories = np.array(query_categories, dtype=np.int32)
#     target_categories = np.array(target_categories, dtype=np.int32)
    
#     print(f"📊 Data shapes: query_features={query_features.shape}, target_features={target_features.shape}")
#     print(f"📊 Category shapes: query_categories={query_categories.shape}, target_categories={target_categories.shape}")
    
#     # Create data dictionary - CORRECT FORMAT FOR CROSS-ATTENTION
#     data = {
#         'query_features': query_features,      # Query set items
#         'query_categories': query_categories,  # Query categories (for reference)
#         'target_features': target_features,    # Target set items (ground truth)
#         'target_categories': target_categories # Target categories (for loss)
#     }
    
#     # For gallery evaluation, also collect all target items
#     if split == 'test':
#         # Flatten all target items for gallery
#         all_target_items = target_features.reshape(-1, target_features.shape[-1])
#         # Remove zero vectors (padding)
#         valid_mask = np.sum(np.abs(all_target_items), axis=1) > 0
#         all_target_items = all_target_items[valid_mask]
        
#         return dataset_from_data(data, batch_size, split), len(query_features), all_target_items
    
#     return dataset_from_data(data, batch_size, split), len(query_features), None


# def dataset_from_data(data, batch_size, split):
#     """Create TensorFlow dataset from data in the format expected by util.py"""
#     # Create dummy targets for Keras
#     dummy_targets = np.zeros((len(data['query_features']), 1), dtype=np.float32)
    
#     # Format data as individual tensors (not nested dict) for util.py compatibility
#     dataset = tf.data.Dataset.from_tensor_slices({
#         'query_features': data['query_features'],
#         'query_categories': data['query_categories'],
#         'target_features': data['target_features'],
#         'target_categories': data['target_categories']
#     })
    
#     # Optimize dataset pipeline
#     if split == 'train':
#         dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    
#     dataset = dataset.batch(batch_size, drop_remainder=True)
#     dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
#     return dataset


# def detect_num_categories(dataset_path):
#     """Detect number of categories from dataset"""
#     # Try category_info.pkl first
#     category_info_path = os.path.join(dataset_path, 'category_info.pkl')
#     if os.path.exists(category_info_path):
#         try:
#             with open(category_info_path, 'rb') as f:
#                 category_info = pickle.load(f)
#             if 'main_categories_def' in category_info:
#                 return len(category_info['main_categories_def'])
#         except:
#             pass
    
#     # Fallback to train.pkl
#     train_path = os.path.join(dataset_path, 'train.pkl')
#     if os.path.exists(train_path):
#         try:
#             # Try tuple format first
#             with open(train_path, 'rb') as f:
#                 data_tuple = pickle.load(f)
            
#             if isinstance(data_tuple, (list, tuple)) and len(data_tuple) >= 5:
#                 query_categories, target_categories = data_tuple[3], data_tuple[4]
#             else:
#                 raise ValueError("Not tuple format")
                
#         except (EOFError, ValueError, IndexError):
#             try:
#                 # Try sequential format
#                 with open(train_path, 'rb') as f:
#                     query_features = pickle.load(f)
#                     target_features = pickle.load(f)
#                     query_categories = pickle.load(f)
#                     target_categories = pickle.load(f)
#             except:
#                 return 7  # Default fallback
        
#         try:
#             all_cats = np.concatenate([
#                 np.array(query_categories).flatten(), 
#                 np.array(target_categories).flatten()
#             ])
#             max_cat = int(np.max(all_cats[all_cats > 0]))
#             print(f"📊 Detected {max_cat} categories from data")
#             return max_cat
#         except:
#             pass
    
#     print("📊 Using default 7 categories")
#     return 7


# def main():
#     parser = argparse.ArgumentParser(description="Correct Set Retrieval Training")
#     parser.add_argument('--dataset', default='IQON3000', choices=['IQON3000', 'DeepFurniture'])
#     parser.add_argument('--mode', default='train', choices=['train', 'test'])
#     parser.add_argument('--batch-size', type=int, default=64)
#     parser.add_argument('--num-heads', type=int, default=8)
#     parser.add_argument('--num-layers', type=int, default=6)
#     parser.add_argument('--epochs', type=int, default=200)
#     parser.add_argument('--learning-rate', type=float, default=1e-4)
#     parser.add_argument('--temperature', type=float, default=1.0)
#     parser.add_argument('--weight-decay', type=float, default=1e-4)
#     parser.add_argument('--use-cycle-loss', action='store_true')
#     parser.add_argument('--data-dir', default='datasets')
#     parser.add_argument('--output-dir', default=None)
#     parser.add_argument('--model-path', default=None)
    
#     args = parser.parse_args()
    
#     # Setup
#     if args.output_dir is None:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         args.output_dir = f"experiments/{args.dataset}/Correct_{timestamp}"
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     setup_gpu()
    
#     # Dataset
#     dataset_path = os.path.join(args.data_dir, args.dataset)
#     if not os.path.exists(dataset_path):
#         print(f"Error: Dataset not found at {dataset_path}")
#         return 1
    
#     # Load data
#     print("📊 Loading dataset...")
#     train_data, n_train, _ = load_data(dataset_path, 'train', args.batch_size)
#     val_data, n_val, _ = load_data(dataset_path, 'validation', args.batch_size)
#     test_data, n_test, test_gallery = load_data(dataset_path, 'test', args.batch_size)
    
#     print(f"📊 Dataset: {args.dataset}")
#     print(f"📈 Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")
#     print(f"🏛️ Test gallery: {test_gallery.shape[0]:,} items" if test_gallery is not None else "")
#     print(f"⚡ Batch size: {args.batch_size}")
    
#     # Model
#     num_categories = detect_num_categories(dataset_path)
#     model_config = {
#         'feature_dim': 512,
#         'num_heads': args.num_heads,
#         'num_layers': args.num_layers,
#         'num_categories': num_categories,
#         'hidden_dim': 512,
#         'use_cycle_loss': args.use_cycle_loss,
#         'temperature': args.temperature,
#         'dropout_rate': 0.1
#     }
    
#     model = create_model(model_config)
    
#     # Load category centers BEFORE building model
#     centers_path = os.path.join(dataset_path, 'category_centers.pkl.gz')
#     if os.path.exists(centers_path):
#         try:
#             with gzip.open(centers_path, 'rb') as f:
#                 centers = pickle.load(f)
#             if isinstance(centers, dict):
#                 centers_array = np.zeros((num_categories, 512))
#                 for cat_id, center in centers.items():
#                     if 1 <= cat_id <= num_categories:
#                         centers_array[cat_id - 1] = center
#                 model.set_category_centers(centers_array)
#                 print("✅ Category centers loaded and initialized")
#         except Exception as e:
#             print(f"⚠️ Warning: Could not load category centers: {e}")
#             # Initialize with random centers
#             random_centers = np.random.normal(0, 0.02, (num_categories, 512)).astype(np.float32)
#             model.set_category_centers(random_centers)
#             print("🎲 Random category centers initialized")
#     else:
#         # Initialize with random centers if file doesn't exist
#         random_centers = np.random.normal(0, 0.02, (num_categories, 512)).astype(np.float32)
#         model.set_category_centers(random_centers)
#         print("🎲 Random category centers initialized (no file found)")
    
#     # Build model with dummy data AFTER setting category centers
#     print("🔨 Building model...")
#     dummy_input = {
#         'query_features': tf.random.normal((args.batch_size, 10, 512), dtype=tf.float32),
#         'query_categories': tf.random.uniform((args.batch_size, 10), 1, num_categories + 1, dtype=tf.int32)
#     }
#     _ = model(dummy_input, training=False)
    
#     total_params = model.count_params()
#     print(f"🚀 Model: {args.num_heads}H-{args.num_layers}L-{num_categories}C | {total_params:,} params")
#     print(f"🎯 Cross-attention: Category centers → Query set")
    
#     # Training
#     if args.mode == 'train':
#         print(f"\n🎯 Training: {args.epochs} epochs")
#         print(f"💡 Problem: Query set → Target set prediction (cross-attention)")
        
#         # Compile model with custom training step
#         print("🔧 Compiling model with custom contrastive loss...")
        
#         model.compile(
#             optimizer=tf.keras.optimizers.Adam(
#                 learning_rate=args.learning_rate,
#                 weight_decay=args.weight_decay
#             ),
#             run_eagerly=False
#         )
        
#         # Callbacks
#         callbacks = [
#             EarlyStopping(
#                 monitor='val_loss', 
#                 patience=15,
#                 restore_best_weights=True, 
#                 verbose=1
#             ),
#             ReduceLROnPlateau(
#                 monitor='val_loss', 
#                 factor=0.5,
#                 patience=8, 
#                 min_lr=1e-6,
#                 verbose=1
#             )
#         ]
        
#         try:
#             print("🔥 Starting training...")
#             start_time = time.time()
            
#             history = model.fit(
#                 train_data,
#                 validation_data=val_data,
#                 epochs=args.epochs,
#                 callbacks=callbacks,
#                 verbose=1
#             )
            
#             total_time = time.time() - start_time
#             print(f"🏁 Training completed in {total_time:.1f}s")
            
#             # Save model
#             model_path = os.path.join(args.output_dir, 'model.weights.h5')
#             model.save_weights(model_path)
            
#             with open(os.path.join(args.output_dir, 'model_config.json'), 'w') as f:
#                 json.dump(model_config, f, indent=2)
            
#             print(f"💾 Model saved: {model_path}")
            
#         except Exception as e:
#             print(f"❌ Training failed: {e}")
#             return 1
    
#     # Load model for testing
#     if args.model_path:
#         try:
#             model.load_weights(args.model_path)
#             print(f"✅ Model loaded: {args.model_path}")
#         except Exception as e:
#             print(f"❌ Failed to load model: {e}")
#             return 1
    
#     # Evaluation with comprehensive metrics (using util.py)
#     if args.mode == 'test' or (args.mode == 'train'):
#         print("🔍 Running comprehensive evaluation with gallery...")
#         try:
#             # Use the comprehensive evaluation from util.py
#             results = evaluate_model(model, test_data, args.output_dir)
            
#             # ▼▼▼▼▼▼▼▼▼▼【ここから修正】▼▼▼▼▼▼▼▼▼▼
#             if results and 'overall' in results:
#                 overall = results['overall']
#                 print("\n\n==================================================")
#                 print("✅ FINAL EVALUATION SUMMARY")
#                 print("==================================================")
#                 print(f"   R@1 : {overall.get('r_at_1', 0):.2%}")
#                 print(f"   R@5 : {overall.get('r_at_5', 0):.2%}")
#                 print(f"   R@10: {overall.get('r_at_10', 0):.2%}")
#                 print(f"   MnR : {overall.get('mnr', 0):.2f}")
#                 print(f"   MdR : {overall.get('mdr', 0):.2f}")
#                 print(f"   MRR : {overall.get('mrr', 0):.4f}")
#                 print("==================================================")
#                 # ▲▲▲▲▲▲▲▲▲▲【ここまで修正】▲▲▲▲▲▲▲▲▲▲
#             else:
#                 print("⚠️ No evaluation results returned")
                
#         except Exception as e:
#             print(f"❌ Comprehensive evaluation failed: {e}")
#             import traceback
#             traceback.print_exc()
            
#             # Fallback to simple evaluation
#             try:
#                 print("🔄 Trying simple evaluation...")
#                 simple_loss = model.evaluate(test_data, verbose=0)
#                 print(f"✅ Test Loss: {simple_loss:.4f}")
#             except Exception as e2:
#                 print(f"❌ Simple evaluation also failed: {e2}")
#                 return 1
    
#     return 0


# if __name__ == "__main__":
#     exit(main())


#!/usr/bin/env python3
"""
run.py - Main script for Set Retrieval Training and Evaluation (Refactored)
"""
import os
import argparse
import json
import pickle
import gzip
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from models import create_model
from util import evaluate_model, setup_gpu_memory
from config import get_dataset_config

def create_output_dir(args: argparse.Namespace) -> str:
    """
    Creates an informative output directory path based on experiment parameters.
    This replaces the old timestamp-based naming for better reproducibility.

    Args:
        args: Command line arguments.

    Returns:
        The path to the output directory.
    """
    if args.output_dir:
        # If a specific output directory is provided, use it.
        return args.output_dir

    # Build the experiment name from key hyperparameters
    components = [
        f"B{args.batch_size}",
        f"L{args.num_layers}",
        f"H{args.num_heads}",
        f"LR{args.learning_rate:.0e}"  # Format learning rate in scientific notation (e.g., 1e-4)
    ]
    experiment_name = "_".join(components)

    # Combine to create the full path: experiments/{dataset_name}/{experiment_name}
    return os.path.join("experiments", args.dataset, experiment_name)

def detect_num_categories(dataset_path: str) -> int:
    """Detects the number of categories from the dataset configuration."""
    config = get_dataset_config(os.path.basename(dataset_path))
    return config.get('num_categories', 7) # Default to 7 if not found

def load_data(dataset_path: str, split: str, batch_size: int) -> tf.data.Dataset:
    """Loads data from .pkl file and creates a tf.data.Dataset."""
    file_path = os.path.join(dataset_path, f'{split}.pkl')
    print(f"Loading data from: {file_path}")
    with open(file_path, 'rb') as f:
        data_tuple = pickle.load(f)

    # Assumes tuple format: (q_feats, t_feats, scene_ids, q_cats, t_cats, set_sizes, q_ids, t_ids)
    data_dict = {
        'query_features': np.array(data_tuple[0], dtype=np.float32),
        'target_features': np.array(data_tuple[1], dtype=np.float32),
        'query_categories': np.array(data_tuple[3], dtype=np.int32),
        'target_categories': np.array(data_tuple[4], dtype=np.int32),
    }

    # Only include item IDs for the test split to avoid potential XLA errors during training
    if split == 'test':
        data_dict['query_item_ids'] = np.array(data_tuple[6], dtype=str)
        data_dict['target_item_ids'] = np.array(data_tuple[7], dtype=str)

    dataset = tf.data.Dataset.from_tensor_slices(data_dict)

    if split == 'train':
        dataset = dataset.shuffle(buffer_size=1000, seed=42)

    # For training, drop the last batch if it's smaller than the batch size
    is_training = (split == 'train')
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def main():
    parser = argparse.ArgumentParser(description="Set Retrieval Training and Evaluation Script")
    parser.add_argument('--dataset', default='IQON3000', choices=['IQON3000', 'DeepFurniture'], help="Dataset to use.")
    parser.add_argument('--mode', default='train', choices=['train', 'test'], help="Mode to run: 'train' or 'test'.")
    parser.add_argument('--data-dir', default='datasets', help="Root directory for datasets.")
    parser.add_argument('--output-dir', default=None, help="Specify a custom output directory. If None, one is generated automatically.")
    parser.add_argument('--model-path', default=None, help="Path to pre-trained model weights for testing.")
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-heads', type=int, default=2)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    # Create an informative output directory based on parameters
    output_dir = create_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📦 Output will be saved to: {output_dir}")

    # Save the arguments for reproducibility
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    setup_gpu_memory()

    dataset_path = os.path.join(args.data_dir, args.dataset)
    num_categories = detect_num_categories(dataset_path)

    model_config = {
        'feature_dim': 512, 'num_heads': args.num_heads, 'num_layers': args.num_layers,
        'num_categories': num_categories, 'temperature': 1.0
    }
    model = create_model(model_config)

    centers_path = os.path.join(dataset_path, 'category_centers.pkl.gz')
    if os.path.exists(centers_path):
        with gzip.open(centers_path, 'rb') as f: centers_dict = pickle.load(f)
        centers_array = np.zeros((num_categories, model_config['feature_dim']), dtype=np.float32)
        for cat_id, center in centers_dict.items():
            if 1 <= int(cat_id) <= num_categories: centers_array[int(cat_id) - 1] = center
        model.set_category_centers(centers_array)

    # Build model with a dummy input to initialize weights and show summary
    dummy_input = {'query_features': tf.random.normal((2, 10, 512))}
    _ = model(dummy_input)
    model.summary()

    model_path_for_testing = args.model_path

    if args.mode == 'train':
        print("\n--- 🚂 Starting Training ---")
        train_data = load_data(dataset_path, 'train', args.batch_size)
        val_data = load_data(dataset_path, 'validation', args.batch_size)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)
        ]
        model.fit(train_data, validation_data=val_data, epochs=args.epochs, callbacks=callbacks)

        # Define the path for the newly trained model
        model_path_for_testing = os.path.join(output_dir, 'final_model.weights.h5')
        model.save_weights(model_path_for_testing)
        print(f"\n💾 Model saved to: {model_path_for_testing}")

    if not model_path_for_testing:
        print("\n[ERROR] No model path for testing. Use --model-path or run in 'train' mode first.")
        return

    print("\n--- 🧪 Starting Evaluation ---")
    model.load_weights(model_path_for_testing)
    test_data = load_data(dataset_path, 'test', args.batch_size)

    # Pass the generated output_dir and data_dir to the evaluation pipeline
    evaluate_model(model, test_data, output_dir, args.data_dir)

if __name__ == "__main__":
    main()