# SetRetrieval: Conservative Convergence-Aware Bidirectional Learning for Heterogeneous Set Retrieval

<!-- **Official Implementation of WACV 2025 Paper** 🏆 -->

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2024.XXXXX-b31b1b.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) -->

> **Abstract**: This work addresses the conservative convergence problem in heterogeneous set retrieval, where traditional unidirectional approaches converge toward statistical averages, failing to capture individual aesthetic preferences. We propose a bidirectional consistency learning framework with cycle consistency loss and curriculum-based hard negative mining to overcome this limitation.

---

## 📋 Table of Contents

<!-- 1. [Quick Start](#-quick-start) -->
2. [Project Structure](#-project-structure)
3. [Installation](#-installation)
4. [Dataset Preparation](#-dataset-preparation)
5. [Training](#-training)
6. [Evaluation](#-evaluation)
7. [Visualization](#-visualization)
8. [Automated Experiments](#-automated-experiments)
9. [Results](#-results)
10. [Troubleshooting](#-troubleshooting)
11. [Citation](#-citation)

<!-- ---

## 🚀 Quick Start

### Prerequisites
- Ubuntu 20.04+ or similar Linux distribution
- NVIDIA GPU with 8GB+ VRAM (24GB+ recommended)
- CUDA 11.0+ and cuDNN 8.0+
- Python 3.10+
- 100GB+ free disk space

### Minimal Example (5 minutes)
```bash
# 1. Clone and setup environment
git clone https://github.com/rikuter67/SetRetrieval_WACV2026.git
cd SetRetrieval_WACV2026
conda create -n setretrieval python=3.10
conda activate setretrieval

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train simplest model (2 layers, 2 heads)
CUDA_VISIBLE_DEVICES=0 python run.py \
  --dataset DeepFurniture \
  --mode train \
  --batch-size 128 \
  --epochs 100 \
  --num-layers 2 \
  --num-heads 2 \
  --learning-rate 1e-4 \
  --patience 10 \
  --use-center-base

# 4. Evaluate with visualization
python run.py \
  --dataset DeepFurniture \
  --mode test \
  --weights-path experiments/DeepFurniture/*/checkpoints/best_model.weights.h5
```

--- -->

## 🗂️ Project Structure

```
SetRetrieval_WACV2026/
├── run.py                      # Main training/evaluation script
├── models.py                   # SetRetrievalModel implementation
├── data_generator.py           # Data loading with CLNeg support
├── util.py                     # Enhanced evaluation & visualization
├── make_datasets.py            # Dataset preprocessing
├── run_exp.sh                  # Automated experiment runner
├── plot.py                     # Result plotting utilities
├── requirements.txt            # Python dependencies
├── env.yml                     # Conda environment
│
├── scripts/                    # Utility scripts
│   └── aggregate_annotations.py
│
├── data/                       # Raw datasets (not in repo)
│   ├── DeepFurniture/         # Download from HuggingFace
│   │   ├── furnitures/        # Furniture images
│   │   ├── scenes/            # Room scene images
│   │   └── metadata/          # Annotations
│   └── IQON3000/              # Manual download required
│
├── datasets/                   # Processed datasets
│   ├── DeepFurniture/
│   │   ├── train.pkl          # Training data
│   │   ├── validation.pkl     # Validation data
│   │   ├── test.pkl           # Test data
│   │   └── category_centers.pkl.gz
│   └── IQON3000/
│
└── experiments/               # Training outputs
    ├── DeepFurniture/
    │   └── [experiment_name]/
    │       ├── checkpoints/
    │       │   └── best_model.weights.h5
    │       ├── training.log
    │       ├── args.txt
    │       ├── visualizations/
    │       │   ├── scene_L3D187S8ENDI2MZOKYUG5TL46AF3P3WE888_retrieval.jpg
    │       │   ├── scene_L3D187S8ENDI3TMJYAUI5MJEXXM3P3XU888_retrieval.jpg
    │       │   └── [more_scene_visualizations].jpg
    │       ├── deepfurniture_percentage_results.csv
    │       └── deepfurniture_percentage_table.txt
    └── IQON3000/
        └── [experiment_folders]/
```

---

## 🔧 Installation

### Option 1: Step-by-Step Installation (Recommended)
```bash
# Create conda environment
conda create -n setretrieval python=3.10
conda activate setretrieval

# Install PyTorch and TensorFlow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow[and-cuda]>=2.15.0

# Install other dependencies
pip install transformers pillow numpy pandas matplotlib scikit-learn tqdm
pip install openai-clip seaborn jupyter

# Verify installation
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__, 'GPUs:', len(tf.config.list_physical_devices('GPU')))"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

### Option 2: Using Requirements File
```bash
conda create -n setretrieval python=3.10
conda activate setretrieval
pip install -r requirements.txt
```

### Option 3: Using Environment File
```bash
conda env create -f env.yml
conda activate setretrieval
```

---

## 📊 Dataset Preparation

### DeepFurniture Dataset (11 categories)

#### Step 1: Download from HuggingFace
```bash
# Clone the DeepFurniture dataset from HuggingFace
cd data/
git clone https://huggingface.co/datasets/rikuter67/DeepFurniture
mv DeepFurniture/* DeepFurniture/

#Verify Data Structure
tree -L 2 data/DeepFurniture/
# Expected:
# data/DeepFurniture/
# ├── furnitures/       # Individual furniture images
# ├── scenes/           # Room scene images  
# └── metadata/         # JSON/JSONL annotation files

# Verify key files exist
ls data/DeepFurniture/metadata/
# Should contain: annotations.json, furnitures.jsonl
```

#### Step 2: Aggregate Scene Annotations
Before processing the dataset, you must first aggregate all individual scene annotations into a single file:
```
python scripts/aggregate_annotations.py \
  --scenes_dir data/DeepFurniture/scenes \
  --output_path data/DeepFurniture/metadata/annotations.json
```


#### Step 3: Process Dataset
```bash
# Extract features and create train/val/test splits (Process without inclusion removal)
python make_datasets.py \
  --dataset deepfurniture \
  --image-dir data/DeepFurniture/furnitures \
  --annotations-json data/DeepFurniture/metadata/annotations.json \
  --furnitures-jsonl data/DeepFurniture/metadata/furnitures.jsonl \
  --output-dir datasets/DeepFurniture \
  --batch-size 32
```

### IQON3000 Dataset (17 categories) - Manual Setup Required

#### Step 1: Download IQON3000 Dataset
**MANUAL DOWNLOAD REQUIRED**: The IQON3000 dataset must be downloaded manually from the official source.

```bash
# 1. Visit the official IQON3000 dataset page
# URL: https://drive.google.com/file/d/1sTfUoNPid9zG_MgV--lWZTBP1XZpmcK8/view

# 2. Create directory and extract downloaded data
mkdir -p data/IQON3000
# Extract your downloaded file to data/IQON3000/

# 3. Verify directory structure
# Expected structure:
# data/IQON3000/
# ├── [user_id]/
# │   └── [coordinate_id]/
# │       ├── [coordinate_id].json
# │       └── [item_id]_m.jpg
```

#### Step 2: Process Dataset
```bash
# Process IQON3000 dataset with user-based splitting (prevents user overlap)
python make_datasets.py \
  --dataset iqon3000 \
  --input-dir data/IQON3000 \
  --output-dir datasets/IQON3000 \
  --batch-size 32 \
  --user-based
```

---

## Training

### Method Comparison (Paper Reproduction)

#### Baseline (CST)
```bash
CUDA_VISIBLE_DEVICES=0 python run.py --dataset DeepFurniture --mode train --data_dir ./data --dataset_dir ./datasets --epochs 500 --batch_size 64 --learning_rate 1e-4 --num_layers 2 --num_heads 2
```

## CST + Cycle (w/o TPaNeg)
```bash
CUDA_VISIBLE_DEVICES=0 python run.py --dataset DeepFurniture --mode train --data_dir ./data --dataset_dir ./datasets --epochs 500 --batch_size 64 --learning_rate 1e-4 --num_layers 2 --num_heads 2 --use_cycle_loss --cycle_lambda 0.2
```

## CST + TPaNeg (w/o Cycle)
```bash
CUDA_VISIBLE_DEVICES=0 python run.py --dataset DeepFurniture --mode train --data_dir ./data --dataset_dir ./datasets --epochs 500 --batch_size 64 --learning_rate 1e-4 --num_layers 2 --num_heads 2 --use_tpaneg --candidate_neg_num 100 --pa_neg_epsilon 0.2
```

## Integrated (Ours)
```bash
CUDA_VISIBLE_DEVICES=0 python run.py --dataset DeepFurniture --mode train --data_dir ./data --dataset_dir ./datasets --epochs 500 --batch_size 64 --learning_rate 1e-4 --num_layers 2 --num_heads 2 --use_cycle_loss --cycle_lambda 0.2 --use_tpaneg --candidate_neg_num 100 --pa_neg_epsilon 0.2
```

---

## Evaluation

### Standard Evaluation
```bash
# Evaluate trained model (simply change --mode to test)
python run.py --mode test (same parameters as training)
```

### Evaluation Metrics
The evaluation system provides comprehensive metrics:

- **Percentage-based Accuracy**: Top-5%, 10%, 20% of gallery
- **Mean Reciprocal Rank (MRR)**: Average of 1/rank
- **Category-wise Performance**: Individual metrics per furniture type


### Expected Performance (Table 3 Results)

| Method | Dataset | Top-5% | Top-10% | Top-20% |
|--------|---------|--------|---------|---------|
| CST | DeepFurniture | 45.79% | 54.79% | 63.71% |
| CST + Cycle | DeepFurniture | 45.83% | 54.71% | 64.55% |
| CST + TPaNeg | DeepFurniture | 49.75% | 57.88% | 66.48% |
| **Integrated (Ours)** | **DeepFurniture** | **48.68%** | **57.26%** | **66.50%** |
| CST | IQON3000 | 19.12% | 29.93% | 45.08% |
| **Integrated (Ours)** | **IQON3000** | **20.68%** | **32.10%** | **47.50%** |

### Training Parameters

| Parameter | Description | DeepFurniture | IQON3000 |
|-----------|-------------|---------------|----------|
| `--cycle_lambda` | Cycle consistency weight α | 0.2 | 0.5 |
| `--candidate_neg_num` | Number of negative samples | 100 | 300 |
| `--pa_neg_epsilon` | Margin parameter ε | 0.2 | 0.2 |
| `--epochs` | Training epochs | 500 | 100 |
| `--batch_size` | Batch size | 64 | 64 |


### Output
```
experiments/DeepFurniture/[experiment_name]/
├── args.json                                    # Training configuration
├── final_model.weights.h5                       # Final model weights
├── results_log.csv                             # Detailed training logs
├── results_summary.csv                         # Summary metrics
├── weighted_metrics_summary.csv                # Category-weighted metrics
├── training_curves_deepfurniture.png           # Training/validation curves
├── performance_by_category.png                 # Per-category performance
├── pca_embedding_space.png                     # Feature space visualization
├── visualization_summary_deepfurniture.txt     # Text summary of results
└── collages/                                   # Scene visualizations
├── scene_L3D187S8ENDI2MZOKYUG5TL46AF3P3WE888_retrieval.jpg
├── scene_L3D187S8ENDI3TMJYAUI5MJEXXM3P3XU888_retrieval.jpg
└── [more_scene_visualizations].jpg
```


---

## 🎨 Visualization

### Automatic Visualization
Visualizations are automatically generated during evaluation for DeepFurniture:
```bash
# Test mode automatically creates visualizations
python run.py --dataset DeepFurniture --mode test
```

### Visualization Output
```
experiments/DeepFurniture/[experiment_name]/visualizations/
├── scene_L3D187S8ENDI2MZOKYUG5TL46AF3P3WE888_retrieval.jpg    # Scene-based retrieval
├── scene_L3D187S8ENDI3TMJYAUI5MJEXXM3P3XU888_retrieval.jpg    # Another scene
├── scene_L3D187S8ENDI4ABCDEFGHIJKLMNOPQRSTUV888_retrieval.jpg    
├── scene_L3D187S8ENDI5WXYZABCDEFGHIJKLMNOP888_retrieval.jpg
└── scene_L3D187S8ENDI6QRSTUVWXYZABCDEFGHIJ888_retrieval.jpg   # Up to 5 scenes
```

Each visualization includes:
- **Scene Image**: Original room photograph
- **Query Items**: Input furniture items (green labels)
- **Target Items**: Ground truth items (red labels)
- **Top-3 Predictions**: Retrieved items with similarity scores

---

## 🔬 Automated Experiments

### Using run_exp.sh Script
```bash
# Run all experiments without CLNeg for IQON3000
./run_exp.sh --gpu 0 --clneg nouse all --mode train --DATASET IQON3000

# Run specific experiment type
./run_exp.sh --gpu 0 ablation --mode train --DATASET DeepFurniture

# Run architecture comparison
./run_exp.sh --gpu 1 architecture --mode train
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{yamazono2025setretrieval,
  title={Conservative Convergence-Aware Bidirectional Learning for Heterogeneous Set Retrieval},
  author={Yamazono, [First Name] and [Co-authors]},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2025}
}
```

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **DeepFurniture dataset** authors for the furniture retrieval benchmark
- **IQON3000 dataset** authors for the fashion retrieval benchmark  
- **CLIP model** by OpenAI for feature extraction
- **WACV 2025** reviewers for valuable feedback

---

## 📧 Contact

- **Lead Author**: Yamazono [First Name] - yamazono@example.com
- **Project Page**: https://yamazono.github.io/SetRetrieval
- **GitHub Repository**: https://github.com/rikuter67/SetRetrieval_WACV2026
- **Issues**: [GitHub Issues](https://github.com/rikuter67/SetRetrieval_WACV2026/issues)

---

**⭐ If you find this work helpful, please star the repository!**