"""
plot.py - All visualization functions for the Set Retrieval project.
Enhanced with training curves, embedding visualization, and retrieval result images.
"""
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import math
import json
import pdb

from config import get_dataset_config
from helpers import build_image_path_map, safe_load_image
from tensorflow.keras.callbacks import Callback

try:
    import matplotlib.pyplot as plt
    import japanize_matplotlib
    import matplotlib
    matplotlib.use('Agg')
    from sklearn.decomposition import PCA
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

# tf.config.run_functions_eagerly(True) 

def log_summary_results(results: Dict[str, Any], config: Dict[str, Any], output_dir: str):
    """
    Logs the summary of experiment results to a CSV file.
    If the file already exists, it appends the new results without a header.
    """
    log_file_path = os.path.join(output_dir, "results_log.csv")
    print(f"[INFO] 📝 Logging summary results to: {log_file_path}")

    try:
        # Extract key summary metrics from the results dictionary
        metrics = results.get('overall_metrics', {})
        summary_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset': results.get('dataset', 'N/A'),
            'top1_acc': metrics.get('r_at_1', 0),
            'top10_acc': metrics.get('r_at_10', 0),
            'mrr': metrics.get('mrr', 0),
            'mean_rank': metrics.get('mnr', 0),
            'median_rank': metrics.get('mdr', 0),
            # Extract hyperparameters from config for context
            'batch_size': config.get('batch_size'),
            'num_layers': config.get('num_layers'),
            'num_heads': config.get('num_heads')
        }
        
        df = pd.DataFrame([summary_data])

        # Check if the file exists to decide whether to write the header
        file_exists = os.path.isfile(log_file_path)
        
        # Open the file in append mode ('a')
        df.to_csv(log_file_path, mode='a', header=not file_exists, index=False)
        
        print(f"[INFO] ✅ Successfully appended results to log file.")
        
    except Exception as e:
        print(f"[ERROR] ❌ Failed to log experiment results: {e}")


def plot_training_curves(history_data, output_dir, dataset_name="Unknown"):
    """修正版プロット関数 - 正しいloss表示 + Top20対応"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # GUI不要のバックエンドを指定
        import numpy as np
        import os
        
        # print(f"[DEBUG] Creating plots for {dataset_name}")
        # print(f"[DEBUG] Available history keys: {list(history_data.keys())}")
        
        # データの詳細をチェック
        for key, values in history_data.items():
            if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                print(f"[DEBUG] {key}: {len(values)} values, sample: {values[0]}, type: {type(values[0])}")
        
        # Extract available metrics from history
        epochs = list(range(1, len(history_data.get('loss', [0])) + 1))
        
        if len(epochs) <= 1:
            print("[WARN] Not enough epochs to plot")
            return
        
        # 2つのサブプロット：Loss（左）、TopK精度（右）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{dataset_name} Training Progress', fontsize=16, fontweight='bold')
        
        # Dataset-specific colors
        colors = {
            'train_loss': '#2E8B57', 'val_loss': '#FF6347',
            'top1': '#1f77b4', 'top5': '#ff7f0e', 'top10': '#2ca02c', 'top20': '#d62728'
        }
        
        # ===== 左グラフ：Loss曲線（修正版） =====
        loss_plotted = False
        
        # ★ 修正：正しいloss値を使用
        train_loss_key = None
        if "X->Y' Loss" in history_data and len(history_data["X->Y' Loss"]) > 0:
            train_loss_key = "X->Y' Loss"
            # print("[DEBUG] Using X->Y' Loss for training loss")
        elif 'loss' in history_data and len(history_data['loss']) > 0:
            train_loss_key = 'loss'
            # print("[DEBUG] Using loss for training loss (fallback)")

        if train_loss_key:
            loss_values = history_data[train_loss_key]
            # NumPy配列に変換
            if not isinstance(loss_values, np.ndarray):
                loss_values = np.array(loss_values)
            
            # NaN や Inf をチェック
            valid_indices = np.isfinite(loss_values)
            if np.any(valid_indices):
                epochs_valid = np.array(epochs)[valid_indices]
                loss_valid = loss_values[valid_indices]
                
                ax1.plot(epochs_valid, loss_valid, 
                        color=colors['train_loss'], linewidth=2.5, 
                        marker='o', markersize=4, label='Training Loss', alpha=0.8)
                
                # ベストポイントをマーク
                best_idx = np.argmin(loss_valid)
                best_val = loss_valid[best_idx]
                ax1.annotate(f'Best: {best_val:.4f}', 
                            xy=(epochs_valid[best_idx], best_val),
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                loss_plotted = True
                # print(f"[DEBUG] Plotted training loss: range {loss_valid.min():.4f}-{loss_valid.max():.4f}")
        
        # Validation Loss
        val_loss_key = None
        if "val_X->Y' Loss" in history_data and len(history_data["val_X->Y' Loss"]) > 0:
            val_loss_key = "val_X->Y' Loss"
            # print("[DEBUG] Using val_X->Y' Loss for validation loss")
        elif 'val_loss' in history_data and len(history_data['val_loss']) > 0:
            val_loss_key = 'val_loss'
            # print("[DEBUG] Using val_loss for validation loss (fallback)")
        
        if val_loss_key:
            val_loss_values = history_data[val_loss_key]
            if not isinstance(val_loss_values, np.ndarray):
                val_loss_values = np.array(val_loss_values)
            
            valid_indices = np.isfinite(val_loss_values)
            if np.any(valid_indices):
                epochs_valid = np.array(epochs)[valid_indices]
                val_loss_valid = val_loss_values[valid_indices]
                
                ax1.plot(epochs_valid, val_loss_valid, 
                        color=colors['val_loss'], linewidth=2.5,
                        marker='s', markersize=4, label='Validation Loss', alpha=0.8)
                
                # ベストポイントをマーク
                best_idx = np.argmin(val_loss_valid)
                best_val = val_loss_valid[best_idx]
                ax1.annotate(f'Best: {best_val:.4f}', 
                            xy=(epochs_valid[best_idx], best_val),
                            xytext=(10, -15), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                loss_plotted = True
                # print(f"[DEBUG] Plotted validation loss: range {val_loss_valid.min():.4f}-{val_loss_valid.max():.4f}")
        
        if loss_plotted:
            ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No Loss Data\nAvailable', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=14, color='gray')
            ax1.set_title('Loss Information', fontsize=14, fontweight='bold')
        
        # ===== 右グラフ：TopK精度曲線（Top20対応） =====
        topk_plotted = False
        k_values = [1, 5, 10, 20]  # ★ Top20を追加
        line_styles = ['-', '--', '-.', ':']  # ★ Top20用のスタイルを追加
        markers = ['o', 's', '^', 'D']  # ★ Top20用のマーカーを追加
        
        # メトリック名の正確なマッピング（モデルの実際の出力に基づく）
        metric_mapping = {}
        
        for key in history_data.keys():
            # print(f"[DEBUG] Checking key: {key}")
            
            # Training metrics
            for k in k_values:
                if key == f'top{k}_accuracy':
                    metric_mapping[f'train_top{k}'] = key
                    # print(f"[DEBUG] Found training metric: {key} -> train_top{k}")
                
                # Validation metrics - 複数のパターンをチェック
                val_patterns = [f'val_top{k}_accuracy', f'val_val_top{k}_accuracy']
                for pattern in val_patterns:
                    if key == pattern:
                        metric_mapping[f'val_top{k}'] = key
                        # print(f"[DEBUG] Found validation metric: {key} -> val_top{k}")
        
        # print(f"[DEBUG] Final metric mapping: {metric_mapping}")
        
        for i, k in enumerate(k_values):
            color = colors.get(f'top{k}', f'C{i}')
            
            # Training TopK Accuracy
            train_key = metric_mapping.get(f'train_top{k}')
            if train_key and train_key in history_data:
                data = history_data[train_key]
                if len(data) > 0:
                    # データ型をチェック
                    if not isinstance(data, np.ndarray):
                        data = np.array(data)
                    
                    # TensorFlowのTensorオブジェクトの場合はnumpy()で変換
                    if hasattr(data[0], 'numpy'):
                        data = np.array([x.numpy() if hasattr(x, 'numpy') else x for x in data])
                    
                    # パーセント変換（0-1の場合のみ）
                    if np.all(data <= 1.0) and np.all(data >= 0.0):
                        train_acc = data * 100
                    else:
                        train_acc = data
                    
                    # 有効なデータのみプロット
                    valid_indices = np.isfinite(train_acc)
                    if np.any(valid_indices):
                        epochs_valid = np.array(epochs)[valid_indices]
                        train_acc_valid = train_acc[valid_indices]
                        
                        ax2.plot(epochs_valid, train_acc_valid, 
                                color=color, linewidth=2.5, linestyle=line_styles[i], 
                                marker=markers[i], markersize=4, label=f'Train Top-{k}', alpha=0.8)
                        topk_plotted = True
                        
                        # print(f"[DEBUG] Plotted train top{k}: {len(train_acc_valid)} points, range: {train_acc_valid.min():.2f}-{train_acc_valid.max():.2f}")
            
            # Validation TopK Accuracy
            val_key = metric_mapping.get(f'val_top{k}')
            if val_key and val_key in history_data:
                data = history_data[val_key]
                if len(data) > 0:
                    if not isinstance(data, np.ndarray):
                        data = np.array(data)
                    
                    # TensorFlowのTensorオブジェクトの場合はnumpy()で変換
                    if hasattr(data[0], 'numpy'):
                        data = np.array([x.numpy() if hasattr(x, 'numpy') else x for x in data])
                    
                    # パーセント変換（0-1の場合のみ）
                    if np.all(data <= 1.0) and np.all(data >= 0.0):
                        val_acc = data * 100
                    else:
                        val_acc = data
                    
                    # 有効なデータのみプロット
                    valid_indices = np.isfinite(val_acc)
                    if np.any(valid_indices):
                        epochs_valid = np.array(epochs)[valid_indices]
                        val_acc_valid = val_acc[valid_indices]
                        
                        ax2.plot(epochs_valid, val_acc_valid, 
                                color=color, linewidth=2.0, linestyle=line_styles[i], 
                                marker=markers[i], markersize=3, label=f'Val Top-{k}', alpha=0.6)
                        topk_plotted = True
                        
                        # print(f"[DEBUG] Plotted val top{k}: {len(val_acc_valid)} points, range: {val_acc_valid.min():.2f}-{val_acc_valid.max():.2f}")
        
        if topk_plotted:
            ax2.set_title('TopK Accuracy', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='lower right', fontsize=9)  # フォントサイズを少し小さく
            ax2.set_ylim(0, 100)
        else:
            ax2.text(0.5, 0.5, 'No TopK Data\nFound', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=14, color='gray')
            ax2.set_title('TopK Accuracy Information', fontsize=14, fontweight='bold')
            # print("[DEBUG] No TopK data was plotted")
        
        plt.tight_layout()
        
        # 保存
        output_path = os.path.join(output_dir, f'training_curves_{dataset_name.lower()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[INFO] ✅ Training curves saved to: {output_path}")
        
        # 実際に保存されたファイルのサイズもチェック
        import os
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            # print(f"[DEBUG] Saved file size: {file_size} bytes")
            if file_size < 1000:  # 1KB未満なら問題あり
                print("[WARN] File size is suspiciously small - plot might be empty")
        
    except Exception as e:
        print(f"[ERROR] ❌ Failed to create training curves: {e}")
        import traceback
        traceback.print_exc()

def plot_performance_charts(results: Dict[str, Any], config: Dict[str, Any], output_dir: str):
    """Generates and saves bar charts for performance metrics."""
    if not HAS_LIBS: 
        print("[WARN] Visualization libraries not available. Skipping performance charts.")
        return
        
    print("[INFO] 🎨 Creating performance charts...")
    try:
        categories = results.get('categories')
        if not categories: 
            print("[WARN] No category data found. Skipping performance charts.")
            return

        dataset_name = results.get('dataset', 'Unknown')
        cat_ids = sorted([int(k) for k in categories.keys()])
        cat_labels = [config.get('category_names', {}).get(cid, f"Cat {cid}") for cid in cat_ids]
        
        df = pd.DataFrame({
            'Top-1% Acc': [categories[str(cid)].get('r_at_1', 0) * 100 for cid in cat_ids],
            'Top-10% Acc': [categories[str(cid)].get('r_at_10', 0) * 100 for cid in cat_ids],
            'Mean Rank': [categories[str(cid)].get('mnr', 0) for cid in cat_ids],
            'MRR': [categories[str(cid)].get('mrr', 0) for cid in cat_ids]
        }, index=cat_labels)

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(f'Performance by Category - {dataset_name}', fontsize=20)
        
        # Fixed rotation issue - removed rot parameter from plot() and added manual rotation
        df['Top-1% Acc'].plot(kind='bar', ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('Top-1% Accuracy', fontsize=14)
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        df['Top-10% Acc'].plot(kind='bar', ax=axes[0, 1], color='salmon')
        axes[0, 1].set_title('Top-10% Accuracy', fontsize=14)
        axes[0, 1].tick_params(axis='x', rotation=45)

        df['Mean Rank'].plot(kind='bar', ax=axes[1, 0], color='lightgreen')
        axes[1, 0].set_title('Mean Rank (Lower is Better)', fontsize=14)
        axes[1, 0].set_ylabel('Average Rank')
        axes[1, 0].tick_params(axis='x', rotation=45)

        df['MRR'].plot(kind='bar', ax=axes[1, 1], color='plum')
        axes[1, 1].set_title('Mean Reciprocal Rank (Higher is Better)', fontsize=14)
        axes[1, 1].tick_params(axis='x', rotation=45)

        for ax in axes.flatten(): 
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            # Fix x-axis label alignment
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_horizontalalignment('right')
                
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        output_path = os.path.join(output_dir, 'performance_by_category.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[INFO] ✅ Performance bar chart saved to: {output_path}")
        
    except Exception as e: 
        print(f"[ERROR] ❌ Failed to create performance visualization: {e}")
        import traceback
        traceback.print_exc()

def plot_pca_embedding_space(gallery: Dict[int, Dict], config: Dict[str, Any], output_dir: str, dataset_name: str):
    """Generates and saves a PCA plot of the gallery feature space."""
    if not HAS_LIBS: 
        print("[WARN] Visualization libraries not available. Skipping PCA visualization.")
        return
        
    print("[INFO] 🎨 Creating PCA embedding space visualization...")
    try:
        features, labels = [], []
        for cat_id, data in gallery.items():
            if data.get('features') is not None:
                features.append(data['features'])
                labels.extend([int(cat_id)] * len(data['features']))
        
        if not features: 
            print("[WARN] No features found for PCA visualization.")
            return

        features = np.vstack(features)
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)

        plt.figure(figsize=(12, 10))
        cat_names = config.get('category_names', {})
        colors = plt.cm.get_cmap('tab10', config.get('num_categories', 1))

        for cat_id in sorted(np.unique(labels)):
            mask = (np.array(labels) == cat_id)
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                        color=colors(cat_id - 1), label=cat_names.get(cat_id, f"Cat {cat_id}"),
                        alpha=0.7, s=15)
        
        # タイトルを正しく設定
        plt.title(f"PCA of Gallery Item Embeddings ({dataset_name})", fontsize=16)
        plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.legend(title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # ファイル名を動的に変更
        output_path = os.path.join(output_dir, f'pca_embedding_space_{dataset_name.lower()}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] ✅ PCA visualization saved to: {output_path}")
        
    except Exception as e: 
        print(f"[ERROR] ❌ Failed to create PCA visualization: {e}")
        import traceback
        traceback.print_exc()

def create_retrieval_result_collage(query_images: List[Image.Image],
                                    target_images: List[Image.Image],
                                    retrieved_images: List[List[Tuple[Image.Image, float, str]]],
                                    save_path: str,
                                    dataset_type: str = "DeepFurniture",
                                    thumb_size: Tuple[int, int] = (150, 150),
                                    set_id: str = None,
                                    max_top_k: int = None,
                                    query_ids: List[str] = None,
                                    target_ids: List[str] = None,
                                    # ✅ 新しい引数を追加
                                    query_categories: List[int] = None,
                                    target_categories: List[int] = None,
                                    category_names: Dict[int, str] = None):
    """
    改版：統一色・大きな文字・グラデーション背景対応 (類似度スコア削除 & 下部空白削減)
    """
    try:
        cw, ch = thumb_size
        
        actual_query_count = len([img for img in query_images if img is not None])
        actual_target_count = len([img for img in target_images if img is not None])
        max_items = max(actual_query_count, actual_target_count, 1)
        
        actual_top_k = 0
        actual_retrieved_count = 0
        
        for retrieved_list in retrieved_images:
            if retrieved_list and len(retrieved_list) > 0:
                actual_retrieved_count += 1
                actual_top_k = max(actual_top_k, len(retrieved_list))
        
        if max_top_k is not None:
            top_k_count = max_top_k
        elif actual_top_k > 0:
            top_k_count = actual_top_k
        else:
            top_k_count = 3
        
        top_k_count = min(top_k_count, 5) # 最大5まで
        max_items = max(max_items, actual_retrieved_count)
        
        # 動的レイアウト
        # Query, Target, Top-1, Top-2, ... の行数
        content_rows = 2 + top_k_count # "Query" と "Target" の2行 + TopKの行数
        cols = max_items + 1           # +1 for labels column
        
        # Dataset-specific styling
        bg_color = '#ffffff'
        accent_color = '#007bff'
        
        # Create canvas
        title_height = 80 # タイトル領域
        
        canvas_width = cols * cw
        # ★ 修正: キャンバスの高さ計算をコンテンツの高さに合わせる
        # 各行がchの高さを持つと仮定して、合計の高さを計算
        canvas_height = content_rows * ch + title_height
        
        canvas = Image.new("RGB", (canvas_width, canvas_height), bg_color)
        draw = ImageDraw.Draw(canvas)
        
        # フォント設定（変更なし）
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_tiny = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except IOError: # フォントが見つからない場合のフォールバック
            print("[WARN] DejaVuSans font not found, falling back to default PIL fonts.")
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_tiny = ImageFont.load_default()

        # Title with SetID and TopK info
        title_text = f"{dataset_type} Retrieval Results (Top-{top_k_count})"
        setID = f"Set ID: {set_id}"
        draw.text((10, 10), title_text, fill=accent_color, font=font_medium)
        draw.text((10, 35), setID, fill=accent_color, font=font_medium)
        
        # 統計情報も表示
        stats_text = f"Query: {actual_query_count}, Target: {actual_target_count}, Retrieved: {actual_retrieved_count}"
        draw.text((10, 60), stats_text, fill="gray", font=font_tiny)
        
        y_offset = title_height
        
        # 動的にRow labelsを生成
        row_labels = ["Query", "Target"] + [f"Top-{i+1}" for i in range(top_k_count)]
        
        # 背景色をTopKランクに応じてオレンジ系グラデーションに
        row_colors = [accent_color, "#dc3545"]
        
        import colorsys
        for i in range(top_k_count):
            if i == 0:
                color = "#ff4500"
            elif i == 1:
                color = "#ff8c00"
            elif i == 2:
                color = "#ffa500"
            else:
                intensity = max(0.3, 1.0 - (i - 3) * 0.1)
                r = int(255 * intensity)
                g = int(165 * intensity)
                b = int(50 * intensity)
                color = f"#{r:02x}{g:02x}{b:02x}"
            row_colors.append(color)
        
        for row_idx, (label, color) in enumerate(zip(row_labels, row_colors)):
            y = y_offset + row_idx * ch
            
            # Row label in first column
            draw.rectangle([0, y, cw-1, y+ch-1], fill=color)
            
            if row_idx <= 1:
                font_to_use = font_medium
            else:
                font_to_use = font_small
            
            text_bbox = font_to_use.getbbox(label)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (cw - text_width) // 2
            text_y = y + (ch - text_height) // 2
            draw.text((text_x, text_y), label, fill="white", font=font_to_use)
            
            # Images in remaining columns
            if row_idx == 0:  # Query row
                for i in range(max_items):
                    x = (i + 1) * cw
                    if i < len(query_images) and query_images[i]:
                        img = query_images[i]
                        img_resized = img.copy()
                        img_resized.thumbnail(thumb_size)
                        
                        # Center the image within its cell
                        img_x = x + (cw - img_resized.width) // 2
                        img_y = y + (ch - img_resized.height) // 2
                        canvas.paste(img_resized, (img_x, img_y))

                        if query_ids and i < len(query_ids):
                            item_id = query_ids[i]
                            id_text_x = x + 5
                            id_text_y = y + 5
                            draw.text((id_text_x, id_text_y), item_id, fill="cyan", font=font_tiny, stroke_width=1, stroke_fill="black")
                        
                        draw.rectangle([x, y, x+cw-1, y+ch-1], outline="black", width=2)

                        # ✅ カテゴリラベルを追加
                        if query_categories and i < len(query_categories) and category_names:
                            cat_id = query_categories[i]
                            cat_name = category_names.get(cat_id, f"Cat{cat_id}")
                            cat_text_x = x + 5
                            cat_text_y = y + ch - 25  # 下部に表示
                            # 背景付きでカテゴリ名を表示
                            draw.rectangle([cat_text_x-2, cat_text_y-2, 
                                        cat_text_x+80, cat_text_y+15], 
                                        fill="blue", outline="white")
                            draw.text((cat_text_x, cat_text_y), cat_name, 
                                    fill="white", font=font_tiny)

                    else:
                        draw.rectangle([x, y, x+cw-1, y+ch-1], outline="lightgray", width=1)

            

                        
            elif row_idx == 1:  # Target row
                for i in range(max_items):
                    x = (i + 1) * cw
                    if i < len(target_images) and target_images[i]:
                        img = target_images[i]
                        img_resized = img.copy()
                        img_resized.thumbnail(thumb_size)
                        
                        # Center the image within its cell
                        img_x = x + (cw - img_resized.width) // 2
                        img_y = y + (ch - img_resized.height) // 2
                        canvas.paste(img_resized, (img_x, img_y))

                        if target_ids and i < len(target_ids):
                            item_id = target_ids[i]
                            id_text_x = x + 5
                            id_text_y = y + 5
                            draw.text((id_text_x, id_text_y), item_id, fill="cyan", font=font_tiny, stroke_width=1, stroke_fill="black")
                        
                        draw.rectangle([x, y, x+cw-1, y+ch-1], outline="black", width=2)

                        # ✅ カテゴリラベルを追加
                        if target_categories and i < len(target_categories) and category_names:
                            cat_id = target_categories[i]
                            cat_name = category_names.get(cat_id, f"Cat{cat_id}")
                            cat_text_x = x + 5
                            cat_text_y = y + ch - 25
                            draw.rectangle([cat_text_x-2, cat_text_y-2, 
                                        cat_text_x+80, cat_text_y+15], 
                                        fill="red", outline="white")
                            draw.text((cat_text_x, cat_text_y), cat_name, 
                                    fill="white", font=font_tiny)

                    else:
                        draw.rectangle([x, y, x+cw-1, y+ch-1], outline="lightgray", width=1)
                        
            else:  # TopK rows
                top_k_idx = row_idx - 2
                for target_idx in range(max_items):
                    x = (target_idx + 1) * cw
                    if (target_idx < len(retrieved_images) and
                        retrieved_images[target_idx] and
                        top_k_idx < len(retrieved_images[target_idx])):
                        
                        img, score, item_id = retrieved_images[target_idx][top_k_idx]

                        if img:
                            img_resized = img.copy()
                            img_resized.thumbnail(thumb_size)
                            
                            # Center the image within its cell
                            img_x = x + (cw - img_resized.width) // 2
                            img_y = y + (ch - img_resized.height) // 2
                            canvas.paste(img_resized, (img_x, img_y))
                            
                            id_text_x = x + 5
                            id_text_y = y + 5
                            draw.text((id_text_x, id_text_y), item_id, fill="cyan", font=font_tiny, stroke_width=1, stroke_fill="black")
                            
                            # 類似度スコアの描画をコメントアウト
                            # score_text = f"{score:.3f}"
                            # score_bbox = font_tiny.getbbox(score_text)
                            # score_width = score_bbox[2] - score_bbox[0]
                            # score_height = score_bbox[3] - score_bbox[1]
                            
                            # score_bg_x = x + cw - score_width - 10
                            # score_bg_y = y + ch - score_height - 10
                            
                            # draw.rectangle([score_bg_x-2, score_bg_y-2, 
                            #                 score_bg_x+score_width+2, score_bg_y+score_height+2],
                            #                 fill="black", outline="white")
                            # draw.text((score_bg_x, score_bg_y), score_text, 
                            #           fill="white", font=font_tiny)
                            
                            draw.rectangle([x, y, x+cw-1, y+ch-1], outline="black", width=1)
                        else:
                            draw.rectangle([x, y, x+cw-1, y+ch-1], fill="lightgray", outline="gray", width=1)
                            draw.text((x+10, y+ch//2), "No Image", fill="darkgray", font=font_tiny)
                    else:
                        draw.rectangle([x, y, x+cw-1, y+ch-1], outline="lightgray", width=1)
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        canvas.save(save_path)
        # print(f"[INFO] ✅ {dataset_type} retrieval collage (Top-{top_k_count}) saved to: {save_path}")
        
    except Exception as e:
        print(f"[ERROR] ❌ Failed to create retrieval collage: {e}")
        import traceback
        traceback.print_exc()


def find_topk_similar_items(query_vector: np.ndarray, 
                           gallery_vectors: Dict[str, np.ndarray], 
                           k: int = 5,
                           exclude_ids: set = None) -> List[Tuple[str, float]]:
    """コサイン類似度によるTop-k類似アイテム検索（動的K対応）"""
    
    # print(f"[DEBUG] find_topk_similar_items called:")
    # print(f"  query_vector shape: {query_vector.shape}")
    # print(f"  gallery_vectors count: {len(gallery_vectors)}")
    # print(f"  k: {k}")
    # print(f"  exclude_ids count: {len(exclude_ids) if exclude_ids else 0}")
    
    exclude_ids = set()
    
    similarities = []
    query_norm = np.linalg.norm(query_vector)
    
    if query_norm == 0:
        print("[WARN] Query vector has zero norm")
        return []
    
    processed_count = 0
    excluded_count = 0
    
    for item_id, gallery_vector in gallery_vectors.items():
        if item_id in exclude_ids:
            excluded_count += 1
            continue
            
        gallery_norm = np.linalg.norm(gallery_vector)
        if gallery_norm == 0:
            similarity = 0.0
        else:
            similarity = np.dot(query_vector, gallery_vector) / (query_norm * gallery_norm)
        
        similarities.append((item_id, float(similarity)))
        processed_count += 1
    
    # print(f"[DEBUG] Processed: {processed_count}, Excluded: {excluded_count}")
    
    # 類似度でソート（降順）してTop-kを返す
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    result = similarities[:k]
    # print(f"[DEBUG] Top-{k} results:")
    # for i, (item_id, score) in enumerate(result[:3]):  # 最初の3つだけ表示
    #     print(f"  {i+1}. ID: {item_id}, Score: {score:.4f}")
    
    return result


def generate_qualitative_examples(model, test_items, gallery, image_path_map, config, output_dir, 
                                num_examples=10, top_k=5, min_target_items=4):
    """
    Generate qualitative retrieval examples with actual images using category-specific galleries.
    
    Args:
        min_target_items: 可視化に必要な最小ターゲットアイテム数（デフォルト: 4）
    """
    if not image_path_map or not test_items:
        print("[WARN] No image path map or test items available. Skipping qualitative examples.")
        return
        
    print(f"[INFO] 🎨 Generating qualitative retrieval examples (min_target_items >= {min_target_items})...")
    vis_dir = output_dir
    os.makedirs(vis_dir, exist_ok=True)
    
    try:
        # Convert test_items to list if it's a generator/iterator
        if hasattr(test_items, '__iter__') and not isinstance(test_items, list):
            test_items = list(test_items)
        
        # ★ 修正: Target数が条件を満たすアイテムのみフィルタリング
        filtered_items = []
        for item_set in test_items:
            t_ids = item_set.get('target_item_ids', [])
            t_cats = item_set.get('target_categories', [])
            
            # 有効なターゲット数をカウント（パディングを除く）
            valid_target_count = sum(1 for tid, tcat in zip(t_ids, t_cats) 
                                   if tid != '0' and tid != 0 and tcat != 0)
            
            if valid_target_count >= min_target_items:
                filtered_items.append(item_set)
        
        print(f"[INFO] Filtered items: {len(filtered_items)}/{len(test_items)} items have >= {min_target_items} targets")
        
        if len(filtered_items) == 0:
            print(f"[WARN] No test items have >= {min_target_items} target items. Skipping visualization.")
            return
        
        # Take a sample of filtered test items
        num_samples = min(len(filtered_items), num_examples)
        samples = random.sample(filtered_items, num_samples)
        
        # ★ 修正：カテゴリ別ギャラリーを構築
        print(f"[DEBUG] Building category-specific galleries...")
        category_galleries = {}  # カテゴリID -> {item_id: feature}
        
        for cat_id, cat_data in gallery.items():
            if isinstance(cat_data, dict) and 'features' in cat_data and 'ids' in cat_data:
                features = cat_data['features']
                item_ids = cat_data['ids']
                
                # カテゴリ固有のギャラリーを構築
                category_galleries[cat_id] = {}
                for item_id, feature in zip(item_ids, features):
                    category_galleries[cat_id][str(item_id)] = feature
                
                print(f"[DEBUG] Category {cat_id}: {len(category_galleries[cat_id])} items")
            else:
                print(f"[DEBUG] Category {cat_id}: Invalid structure")
        
        # print(f"[DEBUG] Built {len(category_galleries)} category galleries")
        
        dataset_type = config.get('dataset', 'DeepFurniture')
        
        # print(f"[DEBUG] Processing {num_samples} samples...")
        
        generated_count = 0
        
        for i, item_set in enumerate(samples):
            try:
                q_feats = item_set.get('query_features')
                q_ids = item_set.get('query_item_ids', [])
                q_cats = item_set.get('query_categories', [])  # ✅ クエリカテゴリ取得
                t_cats = item_set.get('target_categories', [])
                t_ids = item_set.get('target_item_ids', [])
                
                if q_feats is None:
                    print(f"[WARN] No query features in sample {i+1}")
                    continue
                
                # ★ 再確認: Target数をチェック
                valid_target_count = sum(1 for tid, tcat in zip(t_ids, t_cats) 
                                       if tid != '0' and tid != 0 and tcat != 0)
                
                if valid_target_count < min_target_items:
                    print(f"[SKIP] Sample {i+1}: Only {valid_target_count} targets (< {min_target_items})")
                    continue
                
                # Get model predictions
                pred_vectors = model({'query_features': tf.constant(q_feats[np.newaxis, ...])}).numpy()[0]
                
                print(f"[DEBUG] Sample {i+1}: pred_vectors shape = {pred_vectors.shape}, targets = {valid_target_count}")
                
                # Load query images
                query_imgs = []
                query_ids_for_collage = []
                for qid in q_ids:
                    if qid != '0' and qid != 0:  # パディングをスキップ
                        img = safe_load_image(str(qid), image_path_map)
                        if img:
                            query_imgs.append(img)
                            query_ids_for_collage.append(str(qid))
                
                # Load target images
                target_imgs = []
                target_ids_for_collage = []
                valid_target_cats = []
                for tid, tcat in zip(t_ids, t_cats):
                    if tid != '0' and tid != 0 and tcat != 0:  # パディングをスキップ
                        img = safe_load_image(str(tid), image_path_map)
                        if img:
                            target_imgs.append(img)
                            target_ids_for_collage.append(str(tid))
                            valid_target_cats.append(tcat)
                
                print(f"[DEBUG] Loaded {len(target_imgs)} target images (expected: {valid_target_count})")
                
                # ★ 修正：除外リストを作成
                exclude_ids = set(str(qid) for qid in q_ids if qid != '0' and qid != 0)
                exclude_ids.update(str(tid) for tid in t_ids if tid != '0' and tid != 0)
                
                # Find top-k retrievals for each target category
                retrieved_results = []
                
                for j, t_cat in enumerate(valid_target_cats):
                    print(f"[DEBUG] Processing target category {t_cat} (index {j})")
                    
                    if t_cat > 0 and t_cat <= len(pred_vectors):
                        query_vec = pred_vectors[t_cat - 1]  # Convert to 0-based index
                        
                        # ★ 修正：カテゴリ固有のギャラリーを使用
                        if t_cat in category_galleries:
                            category_gallery = category_galleries[t_cat]
                            print(f"[DEBUG] Using category {t_cat} specific gallery with {len(category_gallery)} items")
                            
                            # ★ カテゴリ固有の検索を実行
                            similar_items = find_topk_similar_items(
                                query_vec, category_gallery, k=top_k, exclude_ids=exclude_ids
                            )
                            
                            print(f"[DEBUG] Found {len(similar_items)} similar items for category {t_cat}")
                            
                        else:
                            print(f"[WARN] Category {t_cat} not found in galleries. Available: {list(category_galleries.keys())}")
                            similar_items = []
                        
                        # Load retrieved images
                        retrieved_imgs = []
                        for item_id, score in similar_items:
                            img = safe_load_image(item_id, image_path_map)
                            if img:
                                retrieved_imgs.append((img, score, str(item_id)))
                                # print(f"[DEBUG] Loaded retrieved image: {item_id} (score: {score:.3f})")
                            else:
                                # Create placeholder if image not found
                                placeholder = Image.new('RGB', (150, 150), 'lightgray')
                                retrieved_imgs.append((placeholder, score, str(item_id)))
                                # print(f"[DEBUG] Created placeholder for: {item_id}")
                        
                        retrieved_results.append(retrieved_imgs)
                        print(f"[DEBUG] Added {len(retrieved_imgs)} retrieved images for category {t_cat}")
                        
                    else:
                        print(f"[DEBUG] Invalid category {t_cat} (pred_vectors length: {len(pred_vectors)})")
                        retrieved_results.append([])
                
                print(f"[DEBUG] Total retrieved results: {len(retrieved_results)} lists")

                # ✅ 有効なカテゴリ情報を抽出
                query_cats_for_collage = []
                for qcat in q_cats:
                    if qcat != 0:  # パディングをスキップ
                        query_cats_for_collage.append(qcat)
                
                target_cats_for_collage = []
                for tcat in valid_target_cats:
                    target_cats_for_collage.append(tcat)
                
                # ✅ カテゴリ名マッピングを取得
                category_names = config.get('category_names', {})
                
                # SetIDを取得
                set_id = item_set.get('set_id', f'unknown_set_{i+1}')
                safe_set_id = "".join(c for c in str(set_id) if c.isalnum() or c in "._-")
                
                # Create collage with SetID
                collage_path = os.path.join(vis_dir, f"{safe_set_id}_targets{valid_target_count}.jpg")
                create_retrieval_result_collage(
                    query_images=query_imgs,
                    target_images=target_imgs,
                    retrieved_images=retrieved_results,
                    save_path=collage_path,
                    dataset_type=config.get('dataset', 'Unknown'),
                    set_id=f"{set_id} (T:{valid_target_count})",  # Target数も表示
                    max_top_k=top_k,
                    query_ids=query_ids_for_collage,
                    target_ids=target_ids_for_collage,
                    # ✅ カテゴリ情報を追加
                    query_categories=query_cats_for_collage,
                    target_categories=target_cats_for_collage,
                    category_names=category_names
                )
                
                generated_count += 1
                print(f"[INFO] ✅ Generated example {generated_count}: {collage_path}")
                
            except Exception as e:
                print(f"[ERROR] Failed to process qualitative example {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"[INFO] 🎉 Generated {generated_count} qualitative examples with >= {min_target_items} targets")
                
    except Exception as e:
        print(f"[ERROR] ❌ Failed to generate qualitative examples: {e}")
        import traceback
        traceback.print_exc()


def save_training_history(model_history, output_dir: str, dataset_name: str):
    """
    Save and visualize training history from Keras model.fit() history.
    
    Args:
        model_history: History object from model.fit()
        output_dir: Directory to save plots
        dataset_name: Name of dataset for styling
    """
    if model_history is None:
        print("[WARN] No training history provided.")
        return
        
    try:
        # Extract history data
        if hasattr(model_history, 'history'):
            history_data = model_history.history
        else:
            history_data = model_history
            
        # Create training curves
        plot_training_curves(history_data, output_dir, dataset_name)
        
        # Save history as JSON for later use
        history_path = os.path.join(output_dir, f'training_history_{dataset_name.lower()}.json')
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_history = {}
            for key, values in history_data.items():
                if isinstance(values, np.ndarray):
                    json_history[key] = values.tolist()
                elif isinstance(values, list):
                    json_history[key] = values
                else:
                    json_history[key] = [values]
            
            json.dump(json_history, f, indent=2)
        
        print(f"[INFO] ✅ Training history saved to: {history_path}")
        
    except Exception as e:
        print(f"[ERROR] ❌ Failed to save training history: {e}")

def generate_all_visualizations(model, results, test_items, gallery, config, output_dir, data_dir):
    """Main entry point for plotting, called by util.py."""
    if not HAS_LIBS:
        print("[WARN] Visualization libraries not installed. Skipping all visualizations.")
        return
    
    print("[INFO] 🎨 Starting comprehensive visualization pipeline...")
    
    try:
        # 4. Generate retrieval result images
        dataset_type = results.get('dataset', 'N/A')
        print(f"[INFO] Building image path map from {data_dir}...")

        # 1. Log summary results
        log_summary_results(results, config, output_dir)
        
        # 2. Create performance charts
        plot_performance_charts(results, config, output_dir)
        
        # 3. Create PCA embedding visualization
        plot_pca_embedding_space(gallery, config, output_dir, dataset_name=dataset_type)
        
        config['dataset_name'] = dataset_type

        image_path_map = build_image_path_map(data_dir, dataset_type)
        print(f"[INFO] Image path map built with {len(image_path_map)} items.")
        
        # if image_path_map and test_items:
        #     generate_qualitative_examples(model, test_items, gallery, image_path_map, config, output_dir)
        # else:
        #     print("[WARN] No images found for qualitative examples.")
        
        # 5. Create visualization summary
        create_visualization_summary(results, config, output_dir, dataset_type)
        
        print("[INFO] 🎉 Comprehensive visualization pipeline completed!")
        
    except Exception as e:
        print(f"[ERROR] ❌ Visualization pipeline failed: {e}")
        import traceback
        traceback.print_exc()

def create_visualization_summary(results: Dict[str, Any], config: Dict[str, Any], 
                               output_dir: str, dataset_type: str):
    """Create a summary report of all generated visualizations."""
    try:
        summary_path = os.path.join(output_dir, f"visualization_summary_{dataset_type.lower()}.txt")
        
        # Count generated files
        generated_files = []
        for file in os.listdir(output_dir):
            if file.endswith(('.png', '.jpg', '.jpeg', '.csv', '.json')):
                generated_files.append(file)
        
        # Check qualitative examples directory
        qual_dir = os.path.join(output_dir, "qualitative_examples")
        qual_files = []
        if os.path.exists(qual_dir):
            qual_files = [f for f in os.listdir(qual_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"{dataset_type.upper()} VISUALIZATION SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Dataset: {dataset_type}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {output_dir}\n\n")
            
            # Overall metrics
            overall_metrics = results.get('overall_metrics', {})
            if overall_metrics:
                f.write("PERFORMANCE METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Top-1 Accuracy: {overall_metrics.get('r_at_1', 0):.4f}\n")
                f.write(f"Top-10 Accuracy: {overall_metrics.get('r_at_10', 0):.4f}\n")
                f.write(f"Mean Reciprocal Rank: {overall_metrics.get('mrr', 0):.4f}\n")
                f.write(f"Mean Rank: {overall_metrics.get('mnr', 0):.2f}\n")
                f.write(f"Median Rank: {overall_metrics.get('mdr', 0):.2f}\n\n")
            
            # Generated files
            f.write("GENERATED VISUALIZATIONS:\n")
            f.write("-" * 40 + "\n")
            
            # Main directory files
            chart_files = [f for f in generated_files if 'chart' in f or 'performance' in f]
            pca_files = [f for f in generated_files if 'pca' in f or 'embedding' in f]
            curve_files = [f for f in generated_files if 'training' in f and f.endswith('.png')]
            data_files = [f for f in generated_files if f.endswith(('.csv', '.json'))]
            
            if chart_files:
                f.write("📊 Performance Charts:\n")
                for file in chart_files:
                    f.write(f"   • {file}\n")
                f.write("\n")
            
            if pca_files:
                f.write("🔬 Embedding Visualizations:\n")
                for file in pca_files:
                    f.write(f"   • {file}\n")
                f.write("\n")
            
            if curve_files:
                f.write("📈 Training Curves:\n")
                for file in curve_files:
                    f.write(f"   • {file}\n")
                f.write("\n")
            
            if qual_files:
                f.write("🖼️ Qualitative Examples:\n")
                for file in qual_files[:5]:  # Show first 5
                    f.write(f"   • qualitative_examples/{file}\n")
                if len(qual_files) > 5:
                    f.write(f"   ... and {len(qual_files) - 5} more files\n")
                f.write("\n")
            
            if data_files:
                f.write("📋 Data Files:\n")
                for file in data_files:
                    f.write(f"   • {file}\n")
                f.write("\n")
            
            # Configuration
            f.write("EXPERIMENT CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            for key, value in config.items():
                if key in ['batch_size', 'num_layers', 'num_heads', 'learning_rate', 'epochs']:
                    f.write(f"{key}: {value}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Total files generated: {len(generated_files) + len(qual_files)}\n")
            f.write("=" * 80 + "\n")
        
        print(f"[INFO] ✅ Visualization summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"[ERROR] ❌ Failed to create visualization summary: {e}")

# Integration functions for run.py
def integrate_training_visualization(history, output_dir, dataset_name):
    """
    統合トレーニング可視化関数（TopK対応版）
    """
    try:
        # 基本的な学習曲線
        plot_training_curves(history.history, output_dir, dataset_name)
        
        # 他の既存可視化があれば継続実行
        # save_training_history(history, output_dir, dataset_name)
        # 他の分析関数...
        
        print(f"[INFO] 🎉 Training visualization completed for {dataset_name}!")
        
    except Exception as e:
        print(f"[ERROR] ❌ Training visualization failed: {e}")
        import traceback
        traceback.print_exc()

def load_and_visualize_training_history(history_file: str, output_dir: str, dataset_name: str):
    """
    Load training history from file and create visualizations.
    
    Args:
        history_file: Path to saved training history JSON
        output_dir: Output directory
        dataset_name: Dataset name
    """
    try:
        with open(history_file, 'r') as f:
            history_data = json.load(f)
        
        plot_training_curves(history_data, output_dir, dataset_name)
        print(f"[INFO] ✅ Training visualization loaded from: {history_file}")
        
    except Exception as e:
        print(f"[ERROR] ❌ Failed to load training history: {e}")

# Helper function for dataset-specific image loading
def load_dataset_image(item_id: str, image_path_map: Dict[str, str], 
                      dataset_type: str, default_size: Tuple[int, int] = (150, 150)) -> Optional[Image.Image]:
    """
    Load image with dataset-specific handling.
    
    Args:
        item_id: Item identifier
        image_path_map: Mapping from item_id to image path
        dataset_type: Dataset type (IQON3000 or DeepFurniture)
        default_size: Default size for placeholder images
        
    Returns:
        PIL Image or None if not found
    """
    try:
        if item_id in image_path_map:
            img_path = image_path_map[item_id]
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                return img
        
        # Create dataset-specific placeholder
        placeholder_colors = {
            'IQON3000': '#e8f5e8',  # Light green
            'DeepFurniture': '#e3f2fd'  # Light blue
        }
        
        color = placeholder_colors.get(dataset_type, '#f5f5f5')
        placeholder = Image.new('RGB', default_size, color)
        draw = ImageDraw.Draw(placeholder)
        
        # Add text
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        text = f"{dataset_type}\nNo Image"
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_x = (default_size[0] - text_width) // 2
        text_y = (default_size[1] - text_height) // 2
        
        draw.text((text_x, text_y), text, fill='gray', font=font)
        
        return placeholder
        
    except Exception as e:
        print(f"[WARN] Failed to load image for {item_id}: {e}")
        return None


# Export key functions for run.py integration
__all__ = [
    'generate_all_visualizations',
    'integrate_training_visualization', 
    'save_training_history',
    'plot_training_curves',
    'create_retrieval_result_collage',
    'load_and_visualize_training_history'
]