# # plot.py - Integrated DeepFurniture-style visualization for all datasets
# import os, math
# import pdb
# from typing import List, Tuple, Optional

# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# from matplotlib.lines import Line2D
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from PIL import Image, ImageDraw, ImageFont
# from sklearn.decomposition import PCA

# from util import (append_dataframe_to_csv, load_scene_image, safe_load_furniture_image, 
#                   find_topk_similar_items_by_euclidean, DATASET_CONFIGS)

# # ------------------------------------------------------------------
# # 1. Training curves (Loss & Avg-Rank) - Dataset agnostic
# # ------------------------------------------------------------------
# def plot_training_metrics(output_dir: str, batch_size: int, epochs: List[int],
#                           train_loss: List[float], val_loss: List[float],
#                           train_avg_rank: List[float], val_avg_rank: List[float],
#                           dataset_type: str = "DeepFurniture"):
#     """Plot training metrics with dataset-specific styling"""
#     os.makedirs(output_dir, exist_ok=True)
#     fig, ax = plt.subplots(1, 2, figsize=(14, 5))

#     # Color scheme by dataset
#     colors = {"DeepFurniture": ("blue", "red"), 
#               "IQON3000": ("green", "orange")}
#     train_color, val_color = colors.get(dataset_type, ("blue", "red"))

#     # Avg‑Rank
#     ax[0].plot(epochs, train_avg_rank, label="Train", color=train_color, linewidth=2)
#     ax[0].plot(epochs, val_avg_rank, label="Val.", color=val_color, linewidth=2)
#     ax[0].set_title(f"Average Rank ({dataset_type})")
#     ax[0].set_xlabel("Epoch")
#     ax[0].set_ylabel("Average Rank")
#     ax[0].legend()
#     ax[0].grid(True, alpha=0.3)

#     # Loss
#     ax[1].plot(epochs, train_loss, label="Train", color=train_color, linewidth=2)
#     ax[1].plot(epochs, val_loss, label="Val.", color=val_color, linewidth=2)
#     ax[1].set_title(f"Loss ({dataset_type})")
#     ax[1].set_xlabel("Epoch")
#     ax[1].set_ylabel("Loss")
#     ax[1].legend()
#     ax[1].grid(True, alpha=0.3)

#     fn = os.path.join(output_dir, f"{dataset_type.lower()}_loss_rank_bs{batch_size}.png")
#     plt.tight_layout()
#     plt.savefig(fn, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"[INFO] Training curves saved → {fn}")

#     # Save data as CSV
#     df = pd.DataFrame(dict(epoch=epochs,
#                           train_loss=train_loss, val_loss=val_loss,
#                           train_avg_rank=train_avg_rank, val_avg_rank=val_avg_rank,
#                           dataset=dataset_type))
#     append_dataframe_to_csv(df, output_dir, f"{dataset_type.lower()}_training_metrics.csv")

# # ------------------------------------------------------------------
# # 2. PCA Visualization (enhanced for both datasets)
# # ------------------------------------------------------------------
# def _scatter(ax, xy, color, marker, label=None, s=80, alpha=1.0):
#     """Helper function for consistent scatter plotting"""
#     ax.scatter(xy[0], xy[1], color=color, marker=marker, s=s, alpha=alpha, label=label)

# def visualize_predictions_with_pca(pred, gt, cats, centers,
#                                    loss_val: float, out_dir: str,
#                                    step: int = 0, dataset_type: str = "DeepFurniture"):
#     """
#     Enhanced PCA visualization for both datasets
#     Each batch of samples is displayed individually in 2D-PCA.
#     """
#     os.makedirs(out_dir, exist_ok=True)
#     p = pred.numpy(); g = gt.numpy(); c = cats.numpy()
#     B = p.shape[0]; 
    
#     # Get dataset configuration
#     config = DATASET_CONFIGS[dataset_type]
#     num_categories = config['num_categories']
#     min_cat, max_cat = config['category_range']
#     colors = plt.cm.tab10.colors

#     fig, axes = plt.subplots(1, B, figsize=(6*B, 5))
#     axes = np.atleast_1d(axes)

#     for b, ax in enumerate(axes):
#         valid = c[b] > 0
        
#         # Handle different prediction shapes
#         if len(p.shape) == 3 and p.shape[1] == num_categories:
#             # p is (B, num_categories, D) - category predictions
#             q = p[b, c[b, valid] - min_cat]  # Convert to 0-based index
#         else:
#             # p is (B, N, D) - direct predictions
#             q = p[b, valid]
            
#         y = g[b]
#         data = np.vstack([q, y, centers.numpy()])
        
#         # Apply PCA
#         emb = PCA(2).fit_transform(data)
#         n_q = q.shape[0]; n_y = y.shape[0]
        
#         # Plot predicted items
#         for i, cid in enumerate(c[b, valid]):
#             if min_cat <= cid <= max_cat:
#                 col = colors[(int(cid) - min_cat) % len(colors)]
#                 _scatter(ax, emb[i], col, "x",
#                          f"Pred {cid}" if i == 0 else None, s=100)
        
#         # Plot ground truth items
#         for j in range(n_y):
#             cid = c[b, j]
#             if min_cat <= cid <= max_cat:
#                 col = colors[(int(cid) - min_cat) % len(colors)]
#             else:
#                 col = "gray"
#             _scatter(ax, emb[n_q + j], col, "o", None, alpha=0.7, s=80)
        
#         # Plot category centers
#         for k in range(num_categories):
#             center_idx = n_q + n_y + k
#             if center_idx < len(emb):
#                 col = colors[k % len(colors)]
#                 _scatter(ax, emb[center_idx], col, "s", None, s=120, alpha=0.8)
            
#         ax.set_title(f"{dataset_type} sample {b}")
#         ax.grid(True, alpha=0.3)
    
#     fig.suptitle(f"{dataset_type} PCA Embedding - step {step}, loss {loss_val:.4f}")
#     fn = os.path.join(out_dir, f"{dataset_type.lower()}_pca_step{step}.png")
#     plt.tight_layout()
#     plt.savefig(fn, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"[INFO] {dataset_type} PCA visualization saved → {fn}")

# # ------------------------------------------------------------------
# # 3. Enhanced Retrieval Collage (DeepFurniture-style)
# # ------------------------------------------------------------------
# def create_retrieval_collage(scene_img: Image.Image,
#                              query_imgs: List[Image.Image],
#                              target_imgs: List[Image.Image],
#                              topk: List[List[Tuple[Image.Image, float]]],
#                              save_path: str,
#                              thumb=(150, 150),
#                              dataset_type: str = "DeepFurniture"):
#     """
#     Enhanced retrieval collage with dataset-specific styling
#     Layout: Scene | Query row | Target row | Top-1/2/3 rows
#     """
#     cw, ch = thumb
#     max_items = max(len(query_imgs), len(target_imgs)) if query_imgs or target_imgs else 3
#     rows = 5  # Scene, Query, Target, Top-1, Top-2, Top-3
#     cols = 1 + max_items  # Scene column + item columns
    
#     # Dataset-specific background colors
#     bg_colors = {
#         "DeepFurniture": "white",
#         "IQON3000": "#f8f9fa"
#     }
#     bg_color = bg_colors.get(dataset_type, "white")
    
#     canvas = Image.new("RGB", (cols * cw, rows * ch), bg_color)
#     draw = ImageDraw.Draw(canvas)
    
#     try:
#         font = ImageFont.truetype("arial.ttf", 12)
#     except:
#         font = ImageFont.load_default()

#     # Add title at top
#     title_height = 25
#     full_canvas = Image.new("RGB", (cols * cw, rows * ch + title_height), bg_color)
#     title_draw = ImageDraw.Draw(full_canvas)
#     title_draw.text((10, 5), f"{dataset_type} Retrieval Results", fill="black", font=font)
    
#     # Paste main canvas below title
#     full_canvas.paste(canvas, (0, title_height))
#     canvas = full_canvas
#     draw = ImageDraw.Draw(canvas)
    
#     # Adjust y coordinates for title offset
#     y_offset = title_height

#     # Scene column (spans all rows)
#     if scene_img:
#         sc = scene_img.copy()
#         sc.thumbnail((cw, rows * ch))
#         canvas.paste(sc, (0, y_offset))
#         draw.text((5, y_offset + 5), "Scene", fill="black", font=font)
#     else:
#         # Create dataset-specific scene placeholder
#         scene_colors = {"DeepFurniture": "lightblue", "IQON3000": "lightgreen"}
#         scene_color = scene_colors.get(dataset_type, "lightblue")
#         scene_placeholder = Image.new("RGB", (cw, rows * ch), scene_color)
#         scene_draw = ImageDraw.Draw(scene_placeholder)
#         scene_draw.text((10, ch), f"{dataset_type}\nScene", fill="black", font=font)
#         canvas.paste(scene_placeholder, (0, y_offset))

#     # Query row
#     y_query = y_offset
#     for i, img in enumerate(query_imgs):
#         x = (i + 1) * cw
#         if img:
#             tmp = img.copy()
#             tmp.thumbnail(thumb)
#             canvas.paste(tmp, (x, y_query))
#         # Add query label with background
#         draw.rectangle((x, y_query + ch - 20, x + 40, y_query + ch), fill="green")
#         draw.text((x + 5, y_query + ch - 18), f"Q{i+1}", fill="white", font=font)

#     # Target row
#     y_target = y_offset + ch
#     for i, img in enumerate(target_imgs):
#         x = (i + 1) * cw
#         if img:
#             tmp = img.copy()
#             tmp.thumbnail(thumb)
#             canvas.paste(tmp, (x, y_target))
#         # Add target label with background
#         draw.rectangle((x, y_target + ch - 20, x + 40, y_target + ch), fill="red")
#         draw.text((x + 5, y_target + ch - 18), f"T{i+1}", fill="white", font=font)

#     # Top-K rows
#     for k in range(3):  # Top-1, Top-2, Top-3
#         y_topk = y_offset + (k + 2) * ch
#         for t in range(max_items):
#             x = (t + 1) * cw
#             if t < len(topk) and k < len(topk[t]):
#                 img, dist = topk[t][k]
#                 if img:
#                     thumb_img = img.copy()
#                     thumb_img.thumbnail(thumb)
#                     canvas.paste(thumb_img, (x, y_topk))
#                     # Add distance label with gradient background
#                     draw.rectangle((x, y_topk + ch - 20, x + cw, y_topk + ch), fill="black")
#                     draw.text((x + 5, y_topk + ch - 18), f"Top{k+1} ({dist:.2f})",
#                               fill="white", font=font)
#             else:
#                 # Empty placeholder with border
#                 placeholder = Image.new("RGB", thumb, "white")
#                 placeholder_draw = ImageDraw.Draw(placeholder)
#                 placeholder_draw.rectangle((0, 0, thumb[0]-1, thumb[1]-1), outline="lightgray")
#                 canvas.paste(placeholder, (x, y_topk))

#     canvas.save(save_path)
#     print(f"[INFO] Enhanced {dataset_type} collage saved → {save_path}")

# # ------------------------------------------------------------------
# # 4. Advanced test set visualization with collages
# # ------------------------------------------------------------------
# def visualize_test_sets_and_collages(model, test_generator, item_vecs: dict,
#                                      scene_root: str = "data",
#                                      furn_root: str = "data",
#                                      out_dir: str = "visuals",
#                                      pca_background: str = "pca_background.pkl",
#                                      top_k: int = 3,
#                                      dataset_type: str = "DeepFurniture"):
#     """
#     Enhanced test set visualization with dataset-aware paths and styling
#     """
#     os.makedirs(out_dir, exist_ok=True)

#     # Load or create background PCA
#     from util import load_background_pca, compute_and_save_background_pca
#     try:
#         pca_all, embX, embY, embC, catX, catY = load_background_pca(pca_background)
#     except (FileNotFoundError, KeyError) as e:
#         print(f"[WARN] {dataset_type} PCA background not found: {e}. Creating new one...")
#         compute_and_save_background_pca(model, test_generator, path=pca_background)
#         pca_all, embX, embY, embC, catX, catY = load_background_pca(pca_background)
#         print(f"[INFO] New {dataset_type} PCA background created and loaded.")

#     # Process 1 batch only for demonstration
#     try:
#         batch_data = next(iter(test_generator))
#         ((Xc, _, _, catQ, catP, qIDs, tIDs, _), setIDs) = batch_data
#     except:
#         print(f"[ERROR] Failed to get batch from {dataset_type} generator")
#         return
    
#     B = Xc.shape[0] // 2
#     config = DATASET_CONFIGS[dataset_type]
#     cmap = plt.cm.get_cmap("tab10", config['num_categories'])

#     # Create scene IDs
#     scene_ids = [f"{dataset_type.lower()}_b0_s{s}_{setIDs[s]}" for s in range(B)]

#     for s in range(min(B, 3)):  # Limit to 3 samples
#         try:
#             Xin, Yin = Xc[s].numpy(), Xc[s+B].numpy()
#             cat_in, cat_tg = catQ[s].numpy(), catP[s].numpy()
#             pred_vecs = model.infer_single_set(Xin)  # (num_categories, dim)

#             # ---------- Enhanced PCA plot ----------
#             comb = np.vstack([Xin, pred_vecs, Yin])
#             emb = pca_all.transform(comb) if pca_all else comb[:, :2]
#             nx = Xin.shape[0]
            
#             fig = plt.figure(figsize=(10, 8))
            
#             # Background points with dataset-specific styling
#             if len(embX) > 0 and len(embY) > 0:
#                 plt.scatter(embX[:, 0], embX[:, 1], c="lightgray", s=3, alpha=0.01, label="Background")
#                 plt.scatter(embY[:, 0], embY[:, 1], c="lightgray", s=3, alpha=0.01)
            
#             # Category centers
#             for i, v in enumerate(embC[:config['num_categories']]): 
#                 plt.scatter(v[0], v[1], marker="s", c=[cmap(i)], s=150, 
#                            label=f"Center {i+1}" if i < 5 else "", edgecolors='black', linewidth=1)
            
#             # Input items (circles)
#             min_cat, max_cat = config['category_range']
#             for v, cid in zip(emb[:nx], cat_in):
#                 if min_cat <= cid <= max_cat: 
#                     plt.scatter(*v, marker="o", c=[cmap(cid-min_cat)], s=150, 
#                                edgecolors='black', linewidth=2)
            
#             # Predicted category embeddings (crosses)
#             for i, v in enumerate(emb[nx:nx+config['num_categories']]):
#                 plt.scatter(*v, marker="x", c=[cmap(i)], s=200, linewidths=4)
            
#             # Target items (triangles)
#             for v, cid in zip(emb[nx+config['num_categories']:], cat_tg):
#                 if min_cat <= cid <= max_cat: 
#                     plt.scatter(*v, marker="^", c=[cmap(cid-min_cat)], s=150,
#                                edgecolors='black', linewidth=2)

#             sid = scene_ids[s]
#             plt.title(f"{dataset_type} PCA Embedding Space – {sid}", fontsize=14, fontweight='bold')
#             plt.grid(True, alpha=0.3)
#             plt.xlabel("PC1", fontsize=12)
#             plt.ylabel("PC2", fontsize=12)
            
#             # Enhanced legend
#             legend_elements = [
#                 plt.scatter([], [], marker="o", c="black", s=150, label="● Input items"),
#                 plt.scatter([], [], marker="^", c="black", s=150, label="▲ Target items"),
#                 plt.scatter([], [], marker="x", c="black", s=200, label="✕ Predictions"),
#                 plt.scatter([], [], marker="s", c="black", s=150, label="■ Centers")
#             ]
#             plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
#             fn = os.path.join(out_dir, f"{sid}_pca.png")
#             plt.tight_layout()
#             plt.savefig(fn, dpi=300, bbox_inches='tight')
#             plt.close()
#             print(f"[INFO] {dataset_type} PCA plot saved: {fn}")

#             # ---------- Enhanced collage ----------
#             # Load images (with fallbacks for missing files)
#             scene = load_scene_image(str(setIDs[s]), scene_root, dataset_type)
#             q_imgs = [safe_load_furniture_image(str(int(i)), furn_root, dataset_type) 
#                      for i in qIDs[s].numpy()]
#             t_imgs = [safe_load_furniture_image(str(int(i)), furn_root, dataset_type) 
#                      for i in tIDs[s].numpy()]

#             # Get top-k retrievals for each target category
#             topks = []
#             for cid in cat_tg:
#                 if not min_cat <= cid <= max_cat:
#                     topks.append([])
#                     continue
                    
#                 vec = pred_vecs[cid - min_cat]  # Convert to 0-based
#                 excl = {*(str(int(i)) for i in qIDs[s].numpy()), 
#                        *(str(int(i)) for i in tIDs[s].numpy())}
#                 rs = find_topk_similar_items_by_euclidean(vec, item_vecs, k=top_k, exclude=excl)
#                 topks.append([(safe_load_furniture_image(iid, furn_root, dataset_type), dist) 
#                              for iid, dist in rs])

#             collage_fn = os.path.join(out_dir, f"{sid}_collage.jpg")
#             create_retrieval_collage(scene, q_imgs, t_imgs, topks, collage_fn, 
#                                    thumb=(150, 150), dataset_type=dataset_type)
            
#         except Exception as e:
#             print(f"[ERROR] Failed to process sample {s}: {e}")
#             continue

# # ------------------------------------------------------------------
# # 5. Category performance analysis (dataset-aware)
# # ------------------------------------------------------------------
# def plot_category_performance(results_csv: str, output_dir: str, dataset_type: str = None):
#     """
#     Enhanced category performance plotting with dataset detection
#     """
#     if not os.path.exists(results_csv):
#         print(f"[WARN] Results CSV not found: {results_csv}")
#         return
    
#     df = pd.read_csv(results_csv)
    
#     # Auto-detect dataset type if not provided
#     if dataset_type is None:
#         if 'IQON3000' in results_csv or 'iqon' in results_csv.lower():
#             dataset_type = "IQON3000"
#         else:
#             dataset_type = "DeepFurniture"
    
#     config = DATASET_CONFIGS[dataset_type]
#     category_names = config['category_names']
#     min_cat, max_cat = config['category_range']
    
#     # Filter for valid categories
#     cat_df = df[df['category_id'].between(min_cat, max_cat)].copy() if 'category_id' in df.columns else df
    
#     if len(cat_df) == 0:
#         print(f"[WARN] No category data found in results CSV for {dataset_type}")
#         return
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Dataset-specific color scheme
#     color_schemes = {
#         "DeepFurniture": plt.cm.Blues,
#         "IQON3000": plt.cm.Greens
#     }
#     cmap = color_schemes.get(dataset_type, plt.cm.Blues)
    
#     # 1. Category distribution
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
#     # Prepare data
#     if 'count' in cat_df.columns:
#         counts = cat_df['count'].values
#         categories = cat_df['category_id'].values if 'category_id' in cat_df.columns else range(len(cat_df))
#     else:
#         counts = [1] * len(cat_df)
#         categories = range(len(cat_df))
    
#     colors = [cmap(0.3 + 0.5 * i / len(counts)) for i in range(len(counts))]
    
#     # Bar plot
#     bars1 = ax1.bar(range(len(categories)), counts, color=colors, alpha=0.8, edgecolor='black')
#     ax1.set_title(f'{dataset_type}: Category Distribution', fontsize=14, fontweight='bold')
#     ax1.set_xlabel('Category', fontsize=12)
#     ax1.set_ylabel('Item Count', fontsize=12)
    
#     # Category labels
#     labels = []
#     for cat_id in categories:
#         cat_name = category_names.get(cat_id, f'Cat{cat_id}')
#         if len(cat_name) > 12:
#             cat_name = cat_name[:12] + "..."
#         labels.append(f"Cat{cat_id}\n{cat_name}")
    
#     ax1.set_xticks(range(len(categories)))
#     ax1.set_xticklabels(labels, rotation=45, ha='right')
#     ax1.grid(True, alpha=0.3)
    
#     # Add count labels on bars
#     for i, (bar, count) in enumerate(zip(bars1, counts)):
#         height = bar.get_height()
#         ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
#                 f'{int(count)}', ha='center', va='bottom', fontweight='bold')
    
#     # 2. Performance metrics (if available)
#     if 'mean_score' in cat_df.columns or any('accuracy' in col for col in cat_df.columns):
#         # Find performance columns
#         perf_cols = [col for col in cat_df.columns if 'score' in col or 'accuracy' in col or 'mrr' in col]
        
#         if perf_cols:
#             perf_data = cat_df[perf_cols[0]].values  # Use first performance column
#             bars2 = ax2.bar(range(len(categories)), perf_data, color=colors, alpha=0.8, edgecolor='black')
#             ax2.set_title(f'{dataset_type}: Performance by Category', fontsize=14, fontweight='bold')
#             ax2.set_xlabel('Category', fontsize=12)
#             ax2.set_ylabel(perf_cols[0].replace('_', ' ').title(), fontsize=12)
#             ax2.set_xticks(range(len(categories)))
#             ax2.set_xticklabels(labels, rotation=45, ha='right')
#             ax2.grid(True, alpha=0.3)
            
#             # Add performance labels
#             for i, (bar, perf) in enumerate(zip(bars2, perf_data)):
#                 height = bar.get_height()
#                 ax2.text(bar.get_x() + bar.get_width()/2., height + max(perf_data)*0.01,
#                         f'{perf:.3f}', ha='center', va='bottom', fontweight='bold')
#         else:
#             ax2.text(0.5, 0.5, 'No performance metrics found', 
#                     transform=ax2.transAxes, ha='center', va='center', fontsize=14)
#             ax2.set_title(f'{dataset_type}: Performance Data', fontsize=14)
#     else:
#         ax2.text(0.5, 0.5, 'No performance metrics available', 
#                 transform=ax2.transAxes, ha='center', va='center', fontsize=14)
#         ax2.set_title(f'{dataset_type}: Performance Data', fontsize=14)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f'{dataset_type.lower()}_category_performance.png'), 
#                 dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"[INFO] {dataset_type} category performance plots saved to {output_dir}")

# # ------------------------------------------------------------------
# # 6. Comprehensive visualization pipeline
# # ------------------------------------------------------------------
# def create_comprehensive_visualization(model, test_generator, item_vecs, 
#                                      results_data, output_dir="visuals",
#                                      dataset_type: str = "DeepFurniture"):
#     """
#     Create comprehensive visualization suite with enhanced DeepFurniture-style output
#     """
#     print(f"[INFO] 🎨 Creating comprehensive {dataset_type} visualization suite")
    
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
#     try:
#         # 1. Test set visualizations with enhanced collages
#         print(f"[INFO] Creating {dataset_type} test set visualizations...")
#         visualize_test_sets_and_collages(
#             model, test_generator, item_vecs,
#             scene_root=f"data/{dataset_type}",
#             furn_root=f"data/{dataset_type}", 
#             out_dir=os.path.join(output_dir, "visualizations"),
#             pca_background=os.path.join(output_dir, f"{dataset_type.lower()}_pca_background.pkl"),
#             top_k=3,
#             dataset_type=dataset_type
#         )
        
#         # 2. Category performance analysis
#         print(f"[INFO] Creating {dataset_type} category analysis...")
#         if 'category_distribution' in results_data:
#             # Create temporary CSV for plotting
#             temp_csv_data = []
#             for cat_id, count in results_data['category_distribution'].items():
#                 temp_csv_data.append({
#                     'category_id': cat_id,
#                     'count': count,
#                     'mean_score': results_data.get('retrieval_metrics', {}).get('category_performance', {}).get(cat_id, {}).get('mean_score', 0.5)
#                 })
            
#             if temp_csv_data:
#                 temp_df = pd.DataFrame(temp_csv_data)
#                 temp_csv_path = os.path.join(output_dir, f"temp_{dataset_type.lower()}_category_data.csv")
#                 temp_df.to_csv(temp_csv_path, index=False)
#                 plot_category_performance(temp_csv_path, os.path.join(output_dir, "visualizations"), dataset_type)
                
#                 # Clean up temp file
#                 try:
#                     os.remove(temp_csv_path)
#                 except:
#                     pass
        
#         # 3. Create summary report
#         create_visualization_summary_report(results_data, output_dir, dataset_type)
        
#         print(f"[INFO] ✅ Comprehensive {dataset_type} visualization completed. Results in {output_dir}/visualizations/")
        
#     except Exception as e:
#         print(f"[ERROR] ❌ Comprehensive visualization failed for {dataset_type}: {e}")
#         import traceback
#         traceback.print_exc()

# def create_visualization_summary_report(results_data, output_dir, dataset_type):
#     """Create a summary report for visualizations"""
#     try:
#         report_path = os.path.join(output_dir, f"{dataset_type.lower()}_visualization_summary.txt")
        
#         with open(report_path, 'w', encoding='utf-8') as f:
#             f.write("=" * 80 + "\n")
#             f.write(f"{dataset_type} VISUALIZATION SUMMARY REPORT\n")
#             f.write("=" * 80 + "\n\n")
            
#             f.write(f"Generated visualizations for: {dataset_type}\n")
#             f.write(f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
#             # List generated files
#             viz_dir = os.path.join(output_dir, "visualizations")
#             if os.path.exists(viz_dir):
#                 f.write("Generated Files:\n")
#                 f.write("-" * 40 + "\n")
                
#                 for file in sorted(os.listdir(viz_dir)):
#                     if file.endswith(('.png', '.jpg', '.jpeg')):
#                         f.write(f"  📊 {file}\n")
                        
#                         # Add description based on filename
#                         if 'pca' in file:
#                             f.write(f"      → PCA embedding space visualization\n")
#                         elif 'collage' in file:
#                             f.write(f"      → Retrieval results collage\n")
#                         elif 'category' in file:
#                             f.write(f"      → Category performance analysis\n")
#                         elif 'training' in file:
#                             f.write(f"      → Training metrics visualization\n")
#                         f.write("\n")
            
#             # Add dataset statistics
#             if 'category_distribution' in results_data:
#                 f.write("Dataset Statistics:\n")
#                 f.write("-" * 20 + "\n")
#                 total_items = sum(results_data['category_distribution'].values())
#                 f.write(f"Total Items: {total_items:,}\n")
#                 f.write(f"Categories: {len(results_data['category_distribution'])}\n")
#                 f.write(f"Success Rate: {results_data.get('success_rate', 0):.2%}\n\n")
            
#             f.write("=" * 80 + "\n")
#             f.write(f"Visualization report for {dataset_type} dataset\n")
#             f.write("=" * 80 + "\n")
        
#         print(f"[INFO] Visualization summary saved to {report_path}")
        
#     except Exception as e:
#         print(f"[ERROR] Failed to create visualization summary: {e}")

# # ------------------------------------------------------------------
# # 7. Ranking visualization (enhanced)
# # ------------------------------------------------------------------
# def visualize_rank_until_correct(
#     query_vec: np.ndarray,
#     correct_ids: set,
#     item_dict: dict,
#     furn_root: str,
#     thumb_size: tuple,
#     out_path: str,
#     min_items: int,
#     n_cols: int,
#     dataset_type: str = "DeepFurniture"
# ):
#     """
#     Enhanced ranking visualization with dataset-specific styling
#     """
#     import math
#     from PIL import Image, ImageDraw, ImageFont

#     # 1) Compute similarities
#     q = query_vec
#     sims = []
#     for iid, v in item_dict.items():
#         sim = float(np.dot(q, v))  # Assuming normalized vectors
#         sims.append((iid, sim))
#     sims.sort(key=lambda x: x[1], reverse=True)

#     # 2) Find correct position
#     try:
#         idx_correct = next(i for i, (iid, _) in enumerate(sims) if iid in correct_ids)
#     except StopIteration:
#         idx_correct = len(sims) - 1

#     # 3) Determine items to show
#     n = max(idx_correct + 1, min_items)
#     n = min(n, len(sims))

#     # 4) Create enhanced grid
#     rows = math.ceil(n / n_cols)
#     cw, ch = thumb_size
#     title_height = 30
    
#     # Dataset-specific styling
#     bg_colors = {"DeepFurniture": "white", "IQON3000": "#f8f9fa"}
#     accent_colors = {"DeepFurniture": "#2E86C1", "IQON3000": "#28B463"}
    
#     bg_color = bg_colors.get(dataset_type, "white")
#     accent_color = accent_colors.get(dataset_type, "#2E86C1")
    
#     canvas = Image.new("RGB", (n_cols * cw, rows * ch + title_height), bg_color)
#     draw = ImageDraw.Draw(canvas)
    
#     try:
#         font = ImageFont.truetype("arial.ttf", 14)
#         small_font = ImageFont.truetype("arial.ttf", 10)
#     except:
#         font = ImageFont.load_default()
#         small_font = font
    
#     # Enhanced title
#     draw.text((10, 5), f"{dataset_type} Ranking Visualization", fill=accent_color, font=font)
#     draw.text((10, 20), f"Correct item found at rank {idx_correct + 1}", fill="black", font=small_font)

#     for i, (iid, sim) in enumerate(sims[:n]):
#         img = safe_load_furniture_image(iid, furn_root, dataset_type, thumb_size)
#         img.thumbnail(thumb_size)
        
#         row, col = divmod(i, n_cols)
#         x0 = col * cw
#         y0 = row * ch + title_height
#         canvas.paste(img, (x0, y0))

#         # Enhanced border for correct items
#         if iid in correct_ids:
#             # Thick green border for correct
#             draw.rectangle([x0-2, y0-2, x0 + cw + 1, y0 + ch + 1], outline="green", width=4)
#             draw.rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], outline="darkgreen", width=2)
#         else:
#             # Thin gray border for others
#             draw.rectangle([x0, y0, x0 + cw - 1, y0 + ch - 1], outline="lightgray", width=1)

#         # Enhanced similarity display
#         text = f"{sim:.3f}"
#         text_color = "darkgreen" if iid in correct_ids else "darkblue"
        
#         # Text background for better readability
#         bbox = small_font.getbbox(text)
#         tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
#         tx = x0 + (cw - tw) // 2
#         ty = y0 + ch - th - 5
        
#         draw.rectangle([tx-2, ty-1, tx+tw+2, ty+th+1], fill="white", outline=text_color)
#         draw.text((tx, ty), text, fill=text_color, font=small_font)
        
#         # Rank number in top-left
#         rank_text = f"#{i+1}"
#         draw.rectangle([x0, y0, x0+25, y0+15], fill=accent_color)
#         draw.text((x0+2, y0+1), rank_text, fill="white", font=small_font)

#     canvas.save(out_path)
#     print(f"[INFO] Enhanced {dataset_type} ranking visualization saved → {out_path}")

# # ------------------------------------------------------------------
# # 8. Integration function for run.py
# # ------------------------------------------------------------------
# def integrate_with_evaluation_pipeline(model, test_generator, results_data, output_dir, dataset_type):
#     """
#     🎯 Main integration function called by util.py's main_evaluation_pipeline
#     This ensures all DeepFurniture-style visualizations are created automatically
#     """
#     print(f"[INFO] 🎨 Integrating plot.py visualizations for {dataset_type}")
    
#     try:
#         # Build item vectors for retrieval visualization
#         from util import build_item_feature_dict
#         print(f"[INFO] Building item feature dictionary for {dataset_type}...")
#         item_vectors = build_item_feature_dict(model, test_generator)
        
#         # Create comprehensive visualization suite
#         create_comprehensive_visualization(
#             model=model,
#             test_generator=test_generator, 
#             item_vecs=item_vectors,
#             results_data=results_data,
#             output_dir=output_dir,
#             dataset_type=dataset_type
#         )
        
#         print(f"[INFO] ✅ plot.py integration completed for {dataset_type}")
        
#     except Exception as e:
#         print(f"[ERROR] ❌ plot.py integration failed for {dataset_type}: {e}")
#         import traceback
#         traceback.print_exc()



# def create_performance_visualization(results: Dict[str, Any], output_dir: str):
#     """Create performance visualization charts"""
    
#     try:
#         import matplotlib.pyplot as plt
        
#         # Create category performance chart
#         if 'categories' in results:
#             categories = results['categories']
#             cat_ids = sorted(categories.keys())
            
#             metrics = ['top1', 'top5', 'top10', 'mrr']
#             metric_names = ['Top-1', 'Top-5', 'Top-10', 'MRR']
            
#             fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#             axes = axes.flatten()
            
#             for i, (metric, name) in enumerate(zip(metrics, metric_names)):
#                 values = [categories[cat_id][metric] for cat_id in cat_ids]
#                 axes[i].bar(cat_ids, values)
#                 axes[i].set_title(f'{name} by Category')
#                 axes[i].set_xlabel('Category ID')
#                 axes[i].set_ylabel(name)
#                 axes[i].grid(True, alpha=0.3)
            
#             plt.tight_layout()
#             viz_path = os.path.join(output_dir, 'performance_by_category.png')
#             plt.savefig(viz_path, dpi=300, bbox_inches='tight')
#             plt.close()
            
#             print(f"[INFO] Performance visualization saved: {viz_path}")
            
#     except ImportError:
#         print("[WARN] Matplotlib not available, skipping visualization")


"""
plot.py - All visualization functions for the Set Retrieval project.
"""
import os
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

from config import get_dataset_config
from helpers import build_image_path_map, safe_load_image

try:
    import matplotlib.pyplot as plt
    import japanize_matplotlib
    import matplotlib
    matplotlib.use('Agg')
    from sklearn.decomposition import PCA
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

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

def plot_performance_charts(results: Dict[str, Any], config: Dict[str, Any], output_dir: str):
    """Generates and saves bar charts for performance metrics."""
    if not HAS_LIBS: return
    print("[INFO] 🎨 Creating performance charts...")
    try:
        categories = results.get('categories')
        if not categories: return

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
        
        df['Top-1% Acc'].plot(kind='bar', ax=axes[0, 0], color='skyblue', rot=45, ha='right')
        axes[0, 0].set_title('Top-1% Accuracy', fontsize=14)
        axes[0, 0].set_ylabel('Accuracy (%)')
        
        df['Top-10% Acc'].plot(kind='bar', ax=axes[0, 1], color='salmon', rot=45, ha='right')
        axes[0, 1].set_title('Top-10% Accuracy', fontsize=14)

        df['Mean Rank'].plot(kind='bar', ax=axes[1, 0], color='lightgreen', rot=45, ha='right')
        axes[1, 0].set_title('Mean Rank (Lower is Better)', fontsize=14)
        axes[1, 0].set_ylabel('Average Rank')

        df['MRR'].plot(kind='bar', ax=axes[1, 1], color='plum', rot=45, ha='right')
        axes[1, 1].set_title('Mean Reciprocal Rank (Higher is Better)', fontsize=14)

        for ax in axes.flatten(): ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, 'performance_by_category.png'), dpi=300)
        plt.close(fig)
        print(f"[INFO] ✅ Performance bar chart saved.")
    except Exception as e: print(f"[ERROR] ❌ Failed to create performance visualization: {e}")

def plot_pca_embedding_space(gallery: Dict[int, Dict], config: Dict[str, Any], output_dir: str):
    """Generates and saves a PCA plot of the gallery feature space."""
    if not HAS_LIBS: return
    print("[INFO] 🎨 Creating PCA embedding space visualization...")
    try:
        features, labels = [], []
        for cat_id, data in gallery.items():
            if data.get('features') is not None:
                features.append(data['features'])
                labels.extend([int(cat_id)] * len(data['features']))
        if not features: return

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
                        alpha=0.7, s=30)
        
        plt.title("PCA of Gallery Item Embeddings", fontsize=16)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pca_embedding_space.png'), dpi=300)
        plt.close()
        print(f"[INFO] ✅ PCA visualization saved.")
    except Exception as e: print(f"[ERROR] ❌ Failed to create PCA visualization: {e}")

def generate_qualitative_examples(model, test_items, gallery, image_path_map, config, output_dir, num_examples=3):
    """Orchestrates the generation of qualitative result collages."""
    if not image_path_map: return
    print("[INFO] 🎨 Generating qualitative retrieval examples...")
    vis_dir = os.path.join(output_dir, "qualitative_examples")
    os.makedirs(vis_dir, exist_ok=True)
    
    samples = random.sample(test_items, min(len(test_items), num_examples))
    
    for i, item_set in enumerate(samples):
        q_feats = item_set['query_features']
        q_ids = item_set['query_item_ids']
        t_cats = item_set['target_categories']
        t_ids = item_set['target_item_ids']
        
        pred_vectors = model({'query_features': tf.constant(q_feats[np.newaxis, ...])}).numpy()[0]
        
        query_imgs = [safe_load_image(str(id), image_path_map) for id in q_ids]
        target_imgs = [safe_load_image(str(id), image_path_map) for id in t_ids]

        # NOTE: Placeholder for collage creation logic
        print(f"[INFO] Generated placeholder for qualitative example {i+1}")

def generate_all_visualizations(model, results, test_items, gallery, config, output_dir, data_dir):
    """Main entry point for plotting, called by util.py."""
    if not HAS_LIBS:
        print("[WARN] Visualization libraries not installed. Skipping all visualizations.")
        return
        
    log_summary_results(results, config, output_dir)
    dataset_type = results.get('dataset', 'N/A')
    plot_performance_charts(results, config, output_dir)
    plot_pca_embedding_space(gallery, config, output_dir)
    image_path_map = build_image_path_map(data_dir, dataset_type)
    if image_path_map:
        generate_qualitative_examples(model, test_items, gallery, image_path_map, config, output_dir)