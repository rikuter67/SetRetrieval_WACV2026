# SetRetrieval: Unified Framework for Set Completion Tasks

**Transformer-based Set Retrieval Implementation for Furniture and Fashion Domains**

This repository provides a reproducible implementation of set completion tasks across furniture and fashion domains using a unified Transformer-based architecture. The framework supports both DeepFurniture dataset (furniture scenes) and IQON3000 dataset (fashion coordination) with consistent APIs and training procedures.

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Installation](#installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Feature Extraction Pipeline](#feature-extraction-pipeline)
6. [Training & Testing](#training--testing)
7. [Model Architecture](#model-architecture)
8. [Data Generator](#data-generator)
9. [Evaluation & Visualization](#evaluation--visualization)
10. [Experimental Results](#experimental-results)
11. [Citation](#citation)

---

## Overview

Set completion aims to retrieve complementary items given a partial item set (query). This implementation provides:

- **Unified Architecture**: Single Transformer model handling both furniture and fashion domains
- **Category-aware Attention**: Explicit modeling of inter-category relationships
- **Contrastive Learning**: Hard negative sampling with category-based negatives
- **Cycle Consistency**: Optional X→Y→X reconstruction regularization
- **Reproducible Pipeline**: End-to-end scripts from raw data to trained models

---

## Directory Structure

```plaintext
SetRetrieval/
├── README.md
├── run.py                       # Main training/testing entry point
├── data_generator.py           # Unified data loading for both datasets
├── models.py                   # Transformer architecture
├── util.py                     # Logging, ranking, evaluation utilities
├── plot.py                     # Visualization tools
├── make_datasets.py            # Dataset preparation scripts
├── data/                       # Raw dataset storage
│   ├── DeepFurniture/         # DeepFurniture raw data
│   ├── IQON3000/              # IQON3000 raw data
│   └── IQON3000.zip           # Downloaded archive
├── datasets/                   # Processed dataset files
│   ├── DeepFurniture/         # Processed DeepFurniture data
│   │   ├── train.pkl
│   │   ├── validation.pkl
│   │   ├── test.pkl
│   │   └── category_centers.pkl.gz
│   └── IQON3000/              # Processed IQON3000 data
│       ├── train.pkl
│       ├── validation.pkl
│       ├── test.pkl
│       └── category_centers.pkl.gz
└── experiments/               # Experiment outputs
    ├── DeepFurniture/         # DeepFurniture results
    │   ├── checkpoints/       # Saved model weights
    │   ├── logs/             # Training logs & metrics
    │   └── visuals/          # Retrieval visualizations
    └── IQON3000/              # IQON3000 results
        ├── checkpoints/
        ├── logs/
        └── visuals/
```

---

## Installation

### Environment Setup

```bash
# Clone repository
git clone https://github.com/username/SetRetrieval.git
cd SetRetrieval

# Create conda environment
conda create -n setretrieval python=3.9
conda activate setretrieval

# Install dependencies
pip install tensorflow>=2.8 torch>=1.9 transformers scikit-learn matplotlib pillow tqdm
```

### GPU Support (Optional)

```bash
# CUDA support via conda
conda install -c nvidia cuda-toolkit=12.1 cudnn
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

## Dataset Preparation

### Initial Setup

Both datasets are prepared using the unified `make_datasets.py` script:

```bash
# Prepare both datasets
python make_datasets.py --dataset all

# Or prepare individual datasets
python make_datasets.py --dataset deepfurniture
python make_datasets.py --dataset iqon3000
```

### DeepFurniture Dataset

1. **Download and Extract**
   - Raw data should be placed in `data/DeepFurniture/`
   - Expected structure:
     ```
     data/DeepFurniture/
     ├── scenes/           # Scene images
     ├── furnitures/       # Furniture item images
     └── metadata/         # Annotation files
     ```

2. **Feature Extraction Process**
   - Extracts VGG16 fc1 features (4096D) from furniture images
   - Applies linear projection to 256D
   - Groups items by scene ID
   - Filters scenes with 4-20 items

3. **Dataset Splitting**
   - Train: 80%, Validation: 10%, Test: 10%
   - Random split of scenes to avoid data leakage
   - Each scene split into query/target halves
   - Pads/truncates to max 10 items per set

### IQON3000 Dataset

1. **Download and Extract**
   - Place IQON3000.zip in `data/` directory
   - Script automatically extracts to `data/IQON3000/`

2. **CLIP Feature Extraction**
   - Extracts CLIP ViT-B/32 features (512D) from fashion images
   - Maps original categories to 11 standardized categories
   - Maintains coordination structure from IQON dataset

3. **Category Mapping**
   ```
   1: インナー系 (Inner wear)
   2: ボトムス系 (Bottoms)
   3: シューズ系 (Shoes)
   4: バッグ系 (Bags)
   5: アクセサリー系 (Accessories)
   6: 帽子 (Hats)
   7: Tシャツ・カットソー系 (T-shirts/Cut-sew)
   8: シャツ・ブラウス系 (Shirts/Blouses)
   9: ニット・セーター系 (Knits/Sweaters)
   10: アウター系 (Outerwear)
   11: その他 (Others)
   ```

---

## Feature Extraction Pipeline

### DeepFurniture Features
- **Input**: RGB images (224×224 furniture previews)
- **Model**: VGG16 pretrained on ImageNet
- **Layer**: fc1 (before final classification)
- **Dimension**: 4096D → 256D (random linear projection)
- **Normalization**: Per-split z-score normalization

### IQON3000 Features
- **Input**: RGB images (fashion items)
- **Model**: CLIP ViT-B/32
- **Dimension**: 512D (L2 normalized)
- **Categories**: 11 classes (mapped from original IQON categories)

---

## Training & Testing

### Basic Training

```bash
# Train on DeepFurniture
python run.py \
  --dataset deepfurniture \
  --mode train \
  --batch-size 32 \
  --epochs 100 \
  --output-dir experiments/DeepFurniture

# Train on IQON3000
python run.py \
  --dataset iqon3000 \
  --mode train \
  --batch-size 32 \
  --epochs 100 \
  --output-dir experiments/IQON3000
```

### Advanced Training Options

```bash
# With cycle consistency loss
python run.py \
  --dataset deepfurniture \
  --mode train \
  --batch-size 32 \
  --epochs 100 \
  --use-cycle-loss \
  --cycle-lambda 0.2 \
  --output-dir experiments/DeepFurniture

# With category-based hard negatives
python run.py \
  --dataset iqon3000 \
  --mode train \
  --batch-size 32 \
  --epochs 100 \
  --use-clneg-loss \
  --output-dir experiments/IQON3000
```

### Testing

```bash
# Test trained model
python run.py \
  --dataset deepfurniture \
  --mode test \
  --batch-size 32 \
  --output-dir experiments/DeepFurniture
```

**Training Outputs:**
- `experiments/DeepFurniture/checkpoints/best_model_*.weights.h5`: Best model weights
- `experiments/DeepFurniture/logs/result.csv`: Per-epoch metrics
- `experiments/DeepFurniture/logs/loss_acc_*.png`: Training curves

**Testing Outputs:**
- `experiments/DeepFurniture/logs/result.csv`: Updated with test metrics
- `experiments/DeepFurniture/visuals/`: Retrieval visualization collages

---

## Model Architecture

### Core Implementation (`models.py`)

```python
class SetRetrievalModel(tf.keras.Model):
    def __init__(self, feature_dim, embedding_dim, num_heads, num_layers):
        super().__init__()
        self.item_encoder = Dense(embedding_dim)
        self.category_embedding = Embedding(num_categories, embedding_dim)
        self.transformer_layers = [
            TransformerBlock(embedding_dim, num_heads) 
            for _ in range(num_layers)
        ]
        self.set_pooling = GlobalAveragePooling1D()
```

### Key Components

1. **Item Encoder**: Projects item features to embedding space
2. **Category Embedding**: Learnable category representations
3. **Transformer Layers**: Multi-head self-attention with category awareness
4. **Set Pooling**: Permutation-invariant aggregation

### Loss Functions

1. **Contrastive Loss**
   ```python
   # InfoNCE with temperature scaling
   sim_matrix = tf.linalg.matmul(query_repr, target_repr, transpose_b=True)
   loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
       labels=tf.range(batch_size), logits=sim_matrix / temperature
   )
   ```

2. **Cycle Consistency Loss** (Optional)
   ```python
   # Bidirectional reconstruction
   query_to_target = model([query_features, target_categories])
   target_to_query = model([target_features, query_categories])
   cycle_loss = mse_loss(query_repr, target_to_query) + mse_loss(target_repr, query_to_target)
   ```

---

## Data Generator

### Unified Data Loading (`data_generator.py`)

The data generator handles both datasets with consistent interface:

```python
class SetRetrievalDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset_path, batch_size, dataset_type='deepfurniture'):
        # Load preprocessed data
        with open(dataset_path, 'rb') as f:
            self.query_features = pickle.load(f)
            self.target_features = pickle.load(f)
            self.query_categories = pickle.load(f)
            self.target_categories = pickle.load(f)
```

### Hard Negative Sampling

- **Category-based**: Samples negatives from same category
- **Cross-category**: Samples negatives from different categories
- **Scene exclusion**: Excludes items from same scene during training

### Data Format

**Input arrays:**
- `query_features`: (batch_size, max_items, feature_dim)
- `target_features`: (batch_size, max_items, feature_dim)
- `query_categories`: (batch_size, max_items)
- `target_categories`: (batch_size, max_items)

**Padding:** Zero-padded to `max_items=10` for both datasets

---

## Evaluation & Visualization

### Metrics Implementation (`util.py`)

```python
def compute_global_rank(model, test_generator, output_dir):
    """Compute ranking metrics for test set"""
    
    # Extract embeddings for all items
    all_embeddings = []
    all_categories = []
    
    for batch in test_generator:
        embeddings = model.predict(batch)
        all_embeddings.append(embeddings)
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity(query_embeddings, target_embeddings)
    
    # Calculate metrics
    top5_acc = compute_topk_accuracy(sim_matrix, k_percent=0.05)
    top10_acc = compute_topk_accuracy(sim_matrix, k_percent=0.10)
    mrr = compute_mean_reciprocal_rank(sim_matrix)
```

### Visualization Tools (`plot.py`)

```python
def create_retrieval_collage(scene_image, query_images, target_images, 
                           retrieved_images, save_path):
    """Create visualization of retrieval results"""
    
    # Layout: scene | queries | targets | retrievals
    canvas = create_canvas(width=1200, height=800)
    place_scene_image(canvas, scene_image)
    place_thumbnail_grid(canvas, query_images, position='top')
    place_thumbnail_grid(canvas, target_images, position='middle')
    place_thumbnail_grid(canvas, retrieved_images, position='bottom')
    
    canvas.save(save_path)
```

---

## Experimental Results

### DeepFurniture Performance

| Method | Top-5% | Top-10% | Top-20% | MRR |
|--------|--------|---------|---------|-----|
| Random Baseline | 0.050 | 0.100 | 0.200 | 0.025 |
| **SetRetrieval (Base)** | 0.154 | 0.238 | 0.360 | 0.089 |
| **SetRetrieval + CLNeg** | **0.187** | **0.276** | **0.402** | **0.112** |

### IQON3000 Performance

| Method | Top-5% | Top-10% | Top-20% | MRR |
|--------|--------|---------|---------|-----|
| Random Baseline | 0.050 | 0.100 | 0.200 | 0.025 |
| CLIP Similarity | 0.098 | 0.156 | 0.267 | 0.054 |
| **SetRetrieval** | **0.156** | **0.234** | **0.356** | **0.098** |

### Training Configuration

**Optimal Hyperparameters:**
- Batch size: 32
- Learning rate: 1e-3
- Embedding dimension: 256 (DeepFurniture), 512 (IQON3000)
- Transformer layers: 6
- Attention heads: 8
- Dropout: 0.1

**Training Time:**
- DeepFurniture: ~2 hours (100 epochs, RTX 3090)
- IQON3000: ~1 hour (100 epochs, RTX 3090)

### Memory Usage
- Training: ~4GB GPU memory (batch_size=32)
- Inference: ~1GB GPU memory

---

## File Formats

### Dataset Files (`datasets/*/`)

**train.pkl, validation.pkl, test.pkl** contain:
```python
# Pickle format (6 arrays in sequence)
query_features     # (num_scenes, max_items, feature_dim)
target_features    # (num_scenes, max_items, feature_dim)
query_categories   # (num_scenes, max_items)
target_categories  # (num_scenes, max_items)
query_item_ids     # (num_scenes, max_items) - string IDs
target_item_ids    # (num_scenes, max_items) - string IDs
```

**category_centers.pkl.gz** contains:
```python
# Compressed pickle with category centroids
{category_id: feature_vector}  # Dict[int, np.ndarray]
```

### Output Files (`experiments/*/`)

**logs/result.csv** contains:
```csv
epoch,loss,avg_rank,retrieval_acc,test_top5,test_top10,test_mrr
0,2.45,1247.3,0.089,,,
1,2.31,1156.7,0.102,,,
...
99,1.89,892.1,0.187,0.154,0.238,0.089
```

---

## Implementation Notes

### Data Preprocessing
- Both datasets use zero-padding for variable-length sets
- Features are L2-normalized (IQON3000) or z-score normalized (DeepFurniture)
- Category IDs start from 1 (0 reserved for padding)

### Model Training
- Uses Adam optimizer with learning rate scheduling
- Early stopping based on validation ranking performance
- Gradient clipping to prevent instability

### Hard Negative Sampling
- Builds category-wise feature pools during data loading
- Excludes same-scene items to prevent trivial negatives
- Balances negative samples across categories

---

## Citation

```bibtex
@misc{setretrieval2025,
  title={SetRetrieval: Unified Framework for Set Completion Tasks},
  author={Author Name},
  year={2025},
  url={https://github.com/username/SetRetrieval}
}
```

### Dataset Citations

**DeepFurniture:**
```bibtex
@article{liu2019furnishing,
  title={Furnishing Your Room by What You See: An End-to-End Furniture Set Retrieval Framework with Rich Annotated Benchmark Dataset},
  author={Liu, Bingyuan and Zhang, Jiantao and Zhang, Xiaoting and Zhang, Wei and Yu, Chuanhui and Zhou, Yuan},
  journal={arXiv preprint arXiv:1911.09299},
  year={2019}
}
```

**CLIP Features:**
```bibtex
@inproceedings{radford2021learning,
  title={Learning transferable visual representations from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and others},
  booktitle={International Conference on Machine Learning},
  pages={8748--8763},
  year={2021}
}
```