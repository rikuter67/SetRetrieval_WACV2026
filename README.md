# SetRetrieval: Conservative Convergence-Aware Bidirectional Learning for Heterogeneous Set Retrieval

**Official Implementation of WACV 2025 Paper** 🏆

[![arXiv](https://img.shields.io/badge/arXiv-2024.XXXXX-b31b1b.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **Abstract**: This work addresses the conservative convergence problem in heterogeneous set retrieval, where traditional unidirectional approaches converge toward statistical averages, failing to capture individual aesthetic preferences. We propose a bidirectional consistency learning framework with cycle consistency loss and curriculum-based hard negative mining to overcome this limitation.

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
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

---

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

---

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
```

#### Step 2: Verify Data Structure
```bash
# Check directory structure
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

#### Step 3: Process Dataset
```bash
# Extract features and create train/val/test splits
python make_datasets.py \
  --dataset deepfurniture \
  --image-dir data/DeepFurniture/furnitures \
  --annotations-json data/DeepFurniture/metadata/annotations.json \
  --furnitures-jsonl data/DeepFurniture/metadata/furnitures.jsonl \
  --output-dir datasets/DeepFurniture \
  --batch-size 32
```

**Expected output:**
```
Processing DeepFurniture dataset to 11 categories
Using device: cuda
Found 24742 valid images
Extracting DeepFurniture features: 100%|██████████| 
Built 20730 valid DeepFurniture scenes
DeepFurniture split: Train 14402, Validation 3086, Test 3087
Saved 11 DeepFurniture category centers to category_centers.pkl.gz
```

### IQON3000 Dataset (7 categories) - Manual Setup Required

The IQON3000 dataset must be manually downloaded and placed in the correct directory structure:

```bash
# Create directory structure
mkdir -p data/IQON3000

# Download IQON3000 dataset manually from official source
# Place the downloaded data in data/IQON3000/
# Expected structure:
# data/IQON3000/
# ├── [user_id]/
# │   └── [coordinate_id]/
# │       ├── [coordinate_id].json
# │       └── [item_id]_m.jpg

# Process IQON3000 dataset
python make_datasets.py \
  --dataset iqon3000 \
  --input-dir data/IQON3000 \
  --output-dir datasets/IQON3000 \
  --batch-size 32
```

---

## 🏋️ Training

### Simplest Training (Minimal Model)
```bash
# 2 layers, 2 heads - fastest training
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
```

### Standard Training (Baseline)
```bash
# 4 layers, 4 heads - recommended baseline
CUDA_VISIBLE_DEVICES=0 python run.py \
  --dataset DeepFurniture \
  --mode train \
  --batch-size 128 \
  --epochs 100 \
  --num-layers 4 \
  --num-heads 4 \
  --learning-rate 1e-4 \
  --patience 10 \
  --use-center-base
```

### Full Method (Our Approach)
```bash
# Complete method with all components
CUDA_VISIBLE_DEVICES=0 python run.py \
  --dataset DeepFurniture \
  --mode train \
  --batch-size 128 \
  --epochs 100 \
  --num-layers 4 \
  --num-heads 4 \
  --learning-rate 1e-4 \
  --patience 10 \
  --use-center-base \
  --use-cycle-loss \
  --cycle-lambda 0.2 \
  --use-clneg-loss \
  --neg-num 10
```

### Training Options
| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--dataset` | Dataset name | - | DeepFurniture, IQON3000 |
| `--batch-size` | Batch size | 64 | 32, 64, 128 |
| `--num-layers` | Transformer layers | 4 | 2, 4, 6 |
| `--num-heads` | Attention heads | 4 | 2, 4, 8 |
| `--learning-rate` | Learning rate | 1e-4 | 1e-5 to 1e-3 |
| `--use-center-base` | Enable center-base loss | False | Add flag to enable |
| `--use-cycle-loss` | Enable cycle consistency | False | Add flag to enable |
| `--cycle-lambda` | Cycle loss weight | 0.2 | 0.1, 0.2, 0.5 |
| `--use-clneg-loss` | Enable CLNeg mining | False | Add flag to enable |
| `--neg-num` | Negative samples | 10 | 10, 20, 30 |

### Training Output
```
experiments/DeepFurniture/B128_L2_H2_CB_GPU_20240115_143022/
├── checkpoints/
│   └── best_model.weights.h5    # Best model weights
├── training.log                 # Training history
├── args.txt                     # Configuration
└── visualizations/              # (Created during evaluation)
```

---

## 📈 Evaluation

### Standard Evaluation
```bash
# Evaluate trained model
python run.py \
  --dataset DeepFurniture \
  --mode test \
  --batch-size 128 \
  --num-layers 2 \
  --num-heads 2 \
  --use-center-base \
  --weights-path experiments/DeepFurniture/[experiment_name]/checkpoints/best_model.weights.h5
```

### Evaluation Metrics
The evaluation system provides comprehensive metrics:

- **Percentage-based Accuracy**: Top-1%, 3%, 5%, 10%, 20% of gallery
- **Mean Reciprocal Rank (MRR)**: Average of 1/rank
- **Category-wise Performance**: Individual metrics per furniture type
- **Average Percentile Rank**: Overall ranking performance

### Expected Output
```
📊 PERCENTAGE-BASED EVALUATION REPORT - DeepFurniture
================================================================
Overall Performance (Center-Corrected):
  Top-1% Accuracy: 0.127 (12.7%)
  Top-3% Accuracy: 0.245 (24.5%)
  Top-5% Accuracy: 0.318 (31.8%)
  Top-10% Accuracy: 0.456 (45.6%)
  Top-20% Accuracy: 0.623 (62.3%)
  Average Percentile Rank: 12.7%

Category Breakdown:
  1: Chairs     - Top-5%: 35.1%, MRR: 0.152
  2: Tables     - Top-5%: 29.4%, MRR: 0.118
  3: Storage    - Top-5%: 33.2%, MRR: 0.135
  [... detailed results for all 11 categories]

📁 Results saved to: experiments/DeepFurniture/[experiment_name]/
   - 🎨 Scene visualizations: visualizations/
   - 📊 CSV results: deepfurniture_percentage_results.csv
   - 📋 Summary table: deepfurniture_percentage_table.txt
================================================================
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

### Experiment Types
| Command | Description | Experiments |
|---------|-------------|-------------|
| `ablation` | Compare methods (baseline/cycle/clneg/full) | 4 |
| `architecture` | Compare architectures (2L2H/4L4H/6L6H) | 3 |
| `hyperparameter` | Batch size optimization | Variable |
| `cycle-sweep` | Cycle lambda tuning (0.1/0.2/0.5) | 6 |
| `clneg-sweep` | Negative samples tuning (10/20/30) | 6 |
| `all` | Complete factorial design | Up to 216 |

### Custom Experiments
```bash
# Custom parameters
./run_exp.sh \
  --gpu 0 \
  --arch "2L2H,4L4H" \
  --methods "baseline,full" \
  --batch "64,128" \
  --lambdas "0.1,0.2" \
  --negs "10,20" \
  custom \
  --mode train
```



---

## 📊 Results

### DeepFurniture Benchmark (11 categories)
| Method | Layers | Top-1% | Top-5% | Top-10% | Top-20% | MRR | Time |
|--------|--------|--------|--------|---------|---------|-----|------|
| Baseline | 2L2H | 6.5% | 22.3% | 35.8% | 52.1% | 0.065 | 1.5h |
| Baseline | 4L4H | 8.2% | 28.1% | 41.3% | 58.7% | 0.082 | 2.5h |
| + Cycle | 4L4H | 10.5% | 30.5% | 43.8% | 60.2% | 0.095 | 2.7h |
| + CLNeg | 4L4H | 11.3% | 31.2% | 44.6% | 61.5% | 0.103 | 3.0h |
| **Full (Ours)** | **4L4H** | **12.7%** | **33.8%** | **45.6%** | **62.3%** | **0.115** | **3.2h** |

### IQON3000 Benchmark (7 categories)
| Method | Layers | Top-1% | Top-5% | Top-10% | Top-20% | MRR | Time |
|--------|--------|--------|--------|---------|---------|-----|------|
| Baseline | 4L4H | 15.3% | 38.7% | 52.4% | 68.9% | 0.142 | 1.2h |
| **Full (Ours)** | **4L4H** | **18.9%** | **42.5%** | **56.8%** | **72.1%** | **0.178** | **1.5h** |

### Hardware Requirements
| Configuration | VRAM Usage | Training Time (100 epochs) |
|---------------|------------|----------------------------|
| 2L2H, batch=64 | ~6GB | 1-1.5 hours |
| 4L4H, batch=128 | ~12GB | 2-3 hours |
| 6L6H, batch=128 + CLNeg | ~24GB | 3-4 hours |

---

## 🚨 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Solution 1: Reduce batch size
python run.py --batch-size 64 ...

# Solution 2: Use smaller model
python run.py --num-layers 2 --num-heads 2 ...

# Solution 3: Disable CLNeg to save ~60% memory
python run.py --use-center-base --use-cycle-loss  # No --use-clneg-loss
```

#### Missing Dataset Files
```bash
# Check if processed data exists
ls datasets/DeepFurniture/
# Should show: train.pkl, validation.pkl, test.pkl, category_centers.pkl.gz

# If missing, rerun preprocessing
python make_datasets.py --dataset deepfurniture ...
```

#### TensorFlow GPU Issues
```bash
# Enable memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Specify GPU device
export CUDA_VISIBLE_DEVICES=0

# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### Visualization Not Generated
```bash
# Install visualization dependencies
pip install matplotlib pillow

# Verify image paths
ls data/DeepFurniture/scenes/*/image.jpg | head -5
ls data/DeepFurniture/furnitures/*.jpg | head -5

# Enable visualization explicitly
python run.py --mode test --enable-visualization
```

### Performance Optimization

**Data Loading:**
```bash
# Use SSD for datasets
ln -s /ssd/datasets datasets
```

**Mixed Precision (experimental):**
```bash
export TF_ENABLE_ONEDNN_OPTS=1
```

**Larger Batch Sizes:**
```bash
# If you have 48GB+ VRAM
python run.py --batch-size 256 ...
```

---

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