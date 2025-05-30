# plot.py - Updated for IQON3000 dataset (11 categories: 1-11)
import os, math
import pdb
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines   import Line2D
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA

from util import append_dataframe_to_csv, load_scene_image, safe_load_furniture_image, find_topk_similar_items_by_euclidean

# ------------------------------------------------------------------
# 1.  Learning curve (Loss & Avg-Rank)
# ------------------------------------------------------------------
def plot_training_metrics(output_dir: str, batch_size: int, epochs: List[int],
                          train_loss: List[float], val_loss: List[float],
                          train_avg_rank: List[float], val_avg_rank: List[float]):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(14,5))

    # Avg‑Rank
    ax[0].plot(epochs, train_avg_rank, label="Train", color="blue")
    ax[0].plot(epochs, val_avg_rank,   label="Val.",  color="red")
    ax[0].set_title("Average Rank (IQON3000)"); ax[0].set_xlabel("Epoch"); ax[0].legend()

    # Loss
    ax[1].plot(epochs, train_loss, label="Train", color="blue")
    ax[1].plot(epochs, val_loss,   label="Val.",  color="red")
    ax[1].set_title("Loss (IQON3000)"); ax[1].set_xlabel("Epoch"); ax[1].legend()

    fn = os.path.join(output_dir, f"iqon3000_loss_rank_bs{batch_size}.png")
    plt.tight_layout(); plt.savefig(fn); plt.close()
    print(f"[INFO] saved IQON3000 curves → {fn}")

    df = pd.DataFrame(dict(epoch=epochs,
                           train_loss=train_loss,   val_loss=val_loss,
                           train_avg_rank=train_avg_rank,   val_avg_rank=val_avg_rank,
                           dataset="IQON3000"))
    append_dataframe_to_csv(df, output_dir, "iqon3000_training_metrics.csv")

# ------------------------------------------------------------------
# 2.  PCA Visualization (individual/whole batch/comprehensive version)
# ------------------------------------------------------------------
def _scatter(ax, xy, color, marker, label=None, s=80, alpha=1.0):
    ax.scatter(xy[0], xy[1], color=color, marker=marker, s=s, alpha=alpha, label=label)

def visualize_predictions_with_pca(pred, gt, cats, centers,
                                   loss_val: float, out_dir: str,
                                   step: int = 0):
    """
    Each batch of samples is displayed individually in 2D-PCA.
    Updated for IQON3000 dataset with 11 categories.
    """
    os.makedirs(out_dir, exist_ok=True)
    p = pred.numpy(); g = gt.numpy(); c = cats.numpy()
    B = p.shape[0]; colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, B, figsize=(6*B,5))
    axes = np.atleast_1d(axes)

    for b, ax in enumerate(axes):
        valid = c[b] > 0
        
        # Handle different prediction shapes
        if len(p.shape) == 3 and p.shape[1] == 11:
            # p is (B, 11, D) - category predictions
            q = p[b, c[b,valid]-1]  # Select predictions for valid categories (1-based to 0-based)
        else:
            # p is (B, N, D) - direct predictions
            q = p[b, valid]
            
        y = g[b]
        data = np.vstack([q, y, centers.numpy()])
        emb = PCA(2).fit_transform(data)
        n_q = q.shape[0]; n_y = y.shape[0]
        
        for i, cid in enumerate(c[b,valid]):
            if 1 <= cid <= 11:  # Valid IQON3000 categories
                col = colors[(int(cid)-1) % len(colors)]
                _scatter(ax, emb[i], col, "x",
                         f"Pred {cid}" if i==0 else None)
        
        for j in range(n_y):
            cid = c[b,j]
            if 1 <= cid <= 11:
                col = colors[(int(cid)-1) % len(colors)]
            else:
                col = "gray"
            _scatter(ax, emb[n_q+j], col, "o", None, alpha=.6)
            
        ax.set_title(f"IQON3000 sample {b}")
    
    fig.suptitle(f"IQON3000 PCA - step {step}  loss {loss_val:.4f}")
    fn = os.path.join(out_dir, f"iqon3000_pca_step{step}.png")
    plt.savefig(fn); plt.close(); print(f"[INFO] IQON3000 PCA → {fn}")

# ------------------------------------------------------------------
# 3.  Retrieval Collage
# ------------------------------------------------------------------
def create_retrieval_collage(scene_img: Image.Image,
                             query_imgs: List[Image.Image],
                             target_imgs: List[Image.Image],
                             topk: List[List[Tuple[Image.Image,float]]],
                             save_path: str,
                             thumb=(150,150)):
    """
    Left: Entire scene Top: Query Middle: Target Bottom: Top-K×3
    Updated for IQON3000 dataset structure
    """
    cw,ch = thumb; rows = 5; cols = 5+max(len(query_imgs), len(target_imgs))
    canvas = Image.new("RGB", (cols*cw, rows*ch), "white")
    draw   = ImageDraw.Draw(canvas)
    font   = ImageFont.load_default()

    # scene
    if scene_img:
        sc = scene_img.copy(); sc.thumbnail((5*cw,5*ch))
        canvas.paste(sc, (0,0)); draw.text((5,5),"IQON3000 Scene", fill="black", font=font)

    # query row
    for i, img in enumerate(query_imgs):
        x,y = (5+i)*cw, 0
        tmp = img.copy() if img else Image.new("RGB",thumb,"white")
        tmp.thumbnail(thumb); canvas.paste(tmp,(x,y))
        draw.text((x+5, y+ch-18), f"Q{i+1}", fill="black", font=font)

    # target + top‑k
    for t, timg in enumerate(target_imgs):
        x0 = (5+t)*cw
        # target
        y = ch
        tmp = timg.copy() if timg else Image.new("RGB",thumb,"white")
        tmp.thumbnail(thumb); canvas.paste(tmp,(x0,y))
        draw.text((x0+5, y+ch-18), f"T{t+1}", fill="black", font=font)
        # top‑k rows
        for k in range(3):
            yk = (2+k)*ch
            if t < len(topk) and k < len(topk[t]):
                img, dist = topk[t][k]
                thumb_img = img.copy(); thumb_img.thumbnail(thumb)
                canvas.paste(thumb_img,(x0,yk))
                draw.rectangle((x0,yk+ch-18,x0+cw,yk+ch), fill="black")
                draw.text((x0+5, yk+ch-18), f"Top{k+1} ({dist:.1f})",
                          fill="white", font=font)
            else:
                canvas.paste(Image.new("RGB",thumb,"white"), (x0,yk))
    canvas.save(save_path); print(f"[INFO] IQON3000 collage → {save_path}")

# ------------------------------------------------------------------
# 4.  Visualize 1 batch of test set with PCA + Collage
# ------------------------------------------------------------------
def visualize_test_sets_and_collages(model, test_generator, item_vecs: dict,
                                     scene_root="data/IQON3000",
                                     furn_root ="data/IQON3000",
                                     out_dir   ="visuals",
                                     pca_background="iqon3000_pca_background",
                                     top_k=3):
    """
    Updated for IQON3000 dataset with 11 categories
    """
    os.makedirs(out_dir, exist_ok=True)

    # background PCA
    from util import load_background_pca, compute_and_save_background_pca
    try:
        pca_all, embX, embY, embC, catX, catY = load_background_pca(pca_background)
    except (FileNotFoundError, KeyError) as e:
        print(f"[WARN] IQON3000 PCAバックグラウンドが見つからないか不正です: {e}。新規作成します...")
        compute_and_save_background_pca(model, test_generator, path=pca_background)
        pca_all, embX, embY, embC, catX, catY = load_background_pca(pca_background)
        print(f"[INFO] 新しいIQON3000 PCAバックグラウンドを作成し読み込みました。")

    # 1 batch only processing
    ((Xc, _, _, catQ, catP, qIDs, tIDs, _), setIDs) = test_generator[0]
    B = Xc.shape[0]//2
    cmap = plt.cm.get_cmap("tab10", 11)  # 11 categories for IQON3000

    scene_ids = [f"iqon3000_b0_s{s}_{setIDs[s]}" for s in range(B)]

    for s in range(B):
        Xin, Yin = Xc[s].numpy(), Xc[s+B].numpy()
        cat_in, cat_tg = catQ[s].numpy(), catP[s].numpy()
        pred11 = model.infer_single_set(Xin)  # (11, dim)

        # ---------- PCA plot ----------
        comb = np.vstack([Xin, pred11, Yin])
        emb  = pca_all.transform(comb)
        nx = Xin.shape[0]
        fig = plt.figure(figsize=(8,6))
        
        # background
        plt.scatter(embX[:,0], embX[:,1], c="lightgray", s=5, alpha=.02)
        plt.scatter(embY[:,0], embY[:,1], c="lightgray", s=5, alpha=.02)
        
        # cluster centers (11 categories)
        for i,v in enumerate(embC): 
            plt.scatter(v[0],v[1],marker="s", c=[cmap(i)], s=120, 
                       label=f"Center {i+1}" if i < 5 else "")
        
        # current sample - input items
        for v, cid in zip(emb[:nx], cat_in):
            if 1 <= cid <= 11: 
                plt.scatter(*v, marker="o", c=[cmap(cid-1)], s=120)
        
        # predicted category embeddings
        for i,v in enumerate(emb[nx:nx+11]):
            plt.scatter(*v, marker="x", c=[cmap(i)], s=120)
        
        # target items
        for v, cid in zip(emb[nx+11:], cat_tg):
            if 1 <= cid <= 11: 
                plt.scatter(*v, marker="^", c=[cmap(cid-1)], s=120)

        sid = scene_ids[s]
        plt.title(f"IQON3000 PCA – {sid}"); plt.grid(True)
        
        # Add legend for first few categories
        if s == 0:
            category_names = ["インナー", "ボトムス", "シューズ", "バッグ", "アクセサリー"]
            legend_elements = [plt.scatter([], [], c=[cmap(i)], s=120, label=name) 
                             for i, name in enumerate(category_names)]
            plt.legend(loc='upper right', fontsize=8)
        
        fn = os.path.join(out_dir,f"{sid}_pca.png")
        plt.tight_layout(); plt.savefig(fn); plt.close()

        # ---------- collage ----------
        scene   = load_scene_image(setIDs[s], scene_root)
        q_imgs  = [safe_load_furniture_image(str(int(i)), furn_root) for i in qIDs[s].numpy()]
        t_imgs  = [safe_load_furniture_image(str(int(i)), furn_root) for i in tIDs[s].numpy()]

        topks = []
        for cid in cat_tg:
            if not 1 <= cid <= 11:
                topks.append([]); continue
            vec = pred11[cid-1]  # 1-based to 0-based
            excl = {*(qIDs[s].numpy()), *(tIDs[s].numpy())}
            rs = find_topk_similar_items_by_euclidean(vec, item_vecs, k=top_k, 
                                                    exclude={str(int(i)) for i in excl})
            topks.append([(safe_load_furniture_image(iid, furn_root), dist) for iid,dist in rs])

        collage_fn = os.path.join(out_dir,f"{sid}_collage.jpg")
        create_retrieval_collage(scene, q_imgs, t_imgs, topks, collage_fn, thumb=(150,150))

def visualize_rank_until_correct(
    query_vec: np.ndarray,
    correct_ids: set[str],
    item_dict: dict[str, np.ndarray],
    furn_root: str,
    thumb_size: tuple[int,int],
    out_path: str,
    min_items: int,
    n_cols: int
):
    """
    Visualize ranking until correct item is found
    Updated for IQON3000 dataset structure
    """
    import math
    from PIL import Image, ImageDraw, ImageFont

    # 1) クエリベクトルはすでにホワイトニング・正規化済み
    q = query_vec

    # 2) 全アイテムとのコサイン類似度リスト
    sims = []
    for iid, v in item_dict.items():
        # v もホワイトニング・正規化済み
        sim = float(np.dot(q, v))  # -1〜1
        sims.append((iid, sim))
    sims.sort(key=lambda x: x[1], reverse=True)

    # 3) 正解位置取得
    try:
        idx_correct = next(i for i, (iid, _) in enumerate(sims) if iid in correct_ids)
    except StopIteration:
        idx_correct = len(sims) - 1

    # 4) 描画する枚数
    n = max(idx_correct + 1, min_items)
    n = min(n, len(sims))

    # 5) グリッド描画
    rows = math.ceil(n / n_cols)
    cw, ch = thumb_size
    # 上部にテキスト用余白 20px
    canvas = Image.new("RGB", (n_cols * cw, rows * ch + 20), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    
    # Title
    draw.text((10, 2), f"IQON3000 Ranking (Correct at rank {idx_correct + 1})", 
              fill="black", font=font)

    for i, (iid, sim) in enumerate(sims[:n]):
        img = safe_load_furniture_image(iid, furn_root, thumb_size)
        img.thumbnail(thumb_size)
        row, col = divmod(i, n_cols)
        x0 = col * cw
        y0 = row * ch + 20
        canvas.paste(img, (x0, y0))

        # 枠：正解なら赤，それ以外は黒
        color = "red" if iid in correct_ids else "black"
        draw.rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], outline=color, width=4)

        # 下中央にコサイン類似度を表示
        text = f"{sim:.2f}"
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = x0 + (cw - tw) // 2
        ty = row * ch + 20 - th - 2
        draw.text((tx, ty), text, fill="red" if iid in correct_ids else "black", font=font)

    canvas.save(out_path)
    print(f"[INFO] Saved IQON3000 ranking visualization → {out_path}")

# ------------------------------------------------------------------
# 5. Category-specific visualization for IQON3000
# ------------------------------------------------------------------
def plot_category_performance(results_csv: str, output_dir: str):
    """
    Plot category-specific performance metrics for IQON3000 dataset
    """
    if not os.path.exists(results_csv):
        print(f"[WARN] Results CSV not found: {results_csv}")
        return
    
    df = pd.read_csv(results_csv)
    
    # Category names for IQON3000
    category_names = {
        1: "インナー系",
        2: "ボトムス系", 
        3: "シューズ系",
        4: "バッグ系",
        5: "アクセサリー系",
        6: "帽子",
        7: "Tシャツ・カットソー系",
        8: "シャツ・ブラウス系", 
        9: "ニット・セーター系",
        10: "アウター系",
        11: "その他"
    }
    
    # Filter for category data
    cat_df = df[df['category'].between(1, 11)].copy()
    cat_df['category_name'] = cat_df['category'].map(category_names)
    
    if len(cat_df) == 0:
        print("[WARN] No category data found in results CSV")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Mean rank by category
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sort by mean rank for better visualization
    cat_df_sorted = cat_df.sort_values('mean_rank')
    
    bars1 = ax1.bar(range(len(cat_df_sorted)), cat_df_sorted['mean_rank'])
    ax1.set_title('IQON3000: Mean Rank by Category')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Mean Rank')
    ax1.set_xticks(range(len(cat_df_sorted)))
    ax1.set_xticklabels([f"{row['category']}\n{row['category_name'][:8]}" 
                        for _, row in cat_df_sorted.iterrows()], rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    
    # 2. MRR by category
    bars2 = ax2.bar(range(len(cat_df_sorted)), cat_df_sorted['mrr'])
    ax2.set_title('IQON3000: MRR by Category')
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Mean Reciprocal Rank')
    ax2.set_xticks(range(len(cat_df_sorted)))
    ax2.set_xticklabels([f"{row['category']}\n{row['category_name'][:8]}" 
                        for _, row in cat_df_sorted.iterrows()], rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iqon3000_category_performance.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Top-K% accuracy comparison
    topk_cols = [col for col in cat_df.columns if 'top' in col and '%_accuracy' in col]
    if topk_cols:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(cat_df))
        width = 0.15
        
        for i, col in enumerate(topk_cols[:5]):  # Max 5 metrics
            values = cat_df[col].values
            ax.bar(x + i * width, values, width, label=col.replace('_accuracy', ''))
        
        ax.set_title('IQON3000: Top-K% Accuracy by Category')
        ax.set_xlabel('Category')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([f"{row['category']}\n{row['category_name'][:8]}" 
                           for _, row in cat_df.iterrows()], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'iqon3000_topk_accuracy.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Sample count and gallery size visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sample count
    bars1 = ax1.bar(range(len(cat_df)), cat_df['count'])
    ax1.set_title('IQON3000: Sample Count by Category')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Sample Count')
    ax1.set_xticks(range(len(cat_df)))
    ax1.set_xticklabels([f"{row['category']}\n{row['category_name'][:8]}" 
                        for _, row in cat_df.iterrows()], rotation=45)
    
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # Average gallery size
    bars2 = ax2.bar(range(len(cat_df)), cat_df['avg_gallery_size'])
    ax2.set_title('IQON3000: Average Gallery Size by Category')
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Average Gallery Size')
    ax2.set_xticks(range(len(cat_df)))
    ax2.set_xticklabels([f"{row['category']}\n{row['category_name'][:8]}" 
                        for _, row in cat_df.iterrows()], rotation=45)
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iqon3000_data_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] IQON3000 category performance plots saved to {output_dir}")

# ------------------------------------------------------------------
# 6. Comprehensive visualization pipeline for IQON3000
# ------------------------------------------------------------------
def create_comprehensive_iqon3000_visualization(model, test_generator, item_vecs, 
                                               results_csv, output_dir="iqon3000_visuals"):
    """
    Create comprehensive visualization for IQON3000 dataset including:
    - Category performance plots
    - PCA visualizations
    - Sample retrieval collages
    """
    print("[INFO] Creating comprehensive IQON3000 visualization")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Category performance plots
    if os.path.exists(results_csv):
        plot_category_performance(results_csv, output_dir)
    else:
        print(f"[WARN] Results CSV not found: {results_csv}")
    
    # 2. Test set visualizations with collages
    try:
        visualize_test_sets_and_collages(
            model, test_generator, item_vecs,
            scene_root="data/IQON3000",
            furn_root="data/IQON3000", 
            out_dir=output_dir,
            pca_background=os.path.join(output_dir, "iqon3000_pca_background.pkl"),
            top_k=3
        )
    except Exception as e:
        print(f"[WARN] Error creating test set visualizations: {e}")
    
    # 3. Create summary report
    create_iqon3000_summary_report(results_csv, output_dir)
    
    print(f"[INFO] Comprehensive IQON3000 visualization completed. Results in {output_dir}")

def create_iqon3000_summary_report(results_csv: str, output_dir: str):
    """
    Create a text summary report for IQON3000 results
    """
    if not os.path.exists(results_csv):
        return
    
    df = pd.read_csv(results_csv)
    cat_df = df[df['category'].between(1, 11)].copy()
    
    if len(cat_df) == 0:
        return
    
    category_names = {
        1: "インナー系", 2: "ボトムス系", 3: "シューズ系", 4: "バッグ系", 5: "アクセサリー系",
        6: "帽子", 7: "Tシャツ・カットソー系", 8: "シャツ・ブラウス系", 9: "ニット・セーター系",
        10: "アウター系", 11: "その他"
    }
    
    report_path = os.path.join(output_dir, "iqon3000_summary_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("IQON3000 DATASET EVALUATION SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        total_samples = cat_df['count'].sum()
        avg_mean_rank = cat_df['mean_rank'].mean()
        avg_mrr = cat_df['mrr'].mean()
        
        f.write(f"Overall Statistics:\n")
        f.write(f"  Total Samples: {total_samples:,}\n")
        f.write(f"  Average Mean Rank: {avg_mean_rank:.3f}\n")
        f.write(f"  Average MRR: {avg_mrr:.3f}\n\n")
        
        # Category breakdown
        f.write("Category Performance Breakdown:\n")
        f.write("-" * 80 + "\n")
        
        for _, row in cat_df.iterrows():
            cat_id = int(row['category'])
            cat_name = category_names.get(cat_id, f"Category {cat_id}")
            
            f.write(f"Category {cat_id}: {cat_name}\n")
            f.write(f"  Sample Count: {int(row['count']):,}\n")
            f.write(f"  Mean Rank: {row['mean_rank']:.3f}\n")
            f.write(f"  Median Rank: {row['median_rank']:.3f}\n")
            f.write(f"  MRR: {row['mrr']:.3f}\n")
            f.write(f"  Average Gallery Size: {row['avg_gallery_size']:.0f}\n")
            
            # Top-K% metrics if available
            topk_cols = [col for col in row.index if 'top' in col and '%_accuracy' in col]
            if topk_cols:
                f.write(f"  Top-K% Accuracy: ")
                topk_values = [f"{col.split('top')[1].split('%')[0]}%={row[col]:.1%}" 
                              for col in sorted(topk_cols)[:3]]
                f.write(", ".join(topk_values))
                f.write("\n")
            
            f.write("\n")
        
        # Best and worst performing categories
        best_cat = cat_df.loc[cat_df['mrr'].idxmax()]
        worst_cat = cat_df.loc[cat_df['mrr'].idxmin()]
        
        f.write("Performance Highlights:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Performing Category: {category_names.get(int(best_cat['category']))} "
                f"(MRR: {best_cat['mrr']:.3f})\n")
        f.write(f"Worst Performing Category: {category_names.get(int(worst_cat['category']))} "
                f"(MRR: {worst_cat['mrr']:.3f})\n\n")
        
        # Data distribution insights
        largest_cat = cat_df.loc[cat_df['count'].idxmax()]
        smallest_cat = cat_df.loc[cat_df['count'].idxmin()]
        
        f.write("Data Distribution:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Largest Category: {category_names.get(int(largest_cat['category']))} "
                f"({int(largest_cat['count']):,} samples)\n")
        f.write(f"Smallest Category: {category_names.get(int(smallest_cat['category']))} "
                f"({int(smallest_cat['count']):,} samples)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Report generated for IQON3000 dataset evaluation\n")
        f.write("="*80 + "\n")
    
    print(f"[INFO] IQON3000 summary report saved to {report_path}")