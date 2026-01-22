# HARMONY: Hierarchical Anchor Retrieval on Manifold for Oblivious-source Acoustic Anomaly Detection

This repository contains the official PyTorch implementation of the paper: **"HARMONY: Hierarchical Anchor Retrieval on Manifold for Oblivious-source Acoustic Anomaly Detection"**.

## 📝 Abstract

Universal unsupervised acoustic anomaly detection (ASD) aims to handle diverse machine types with a single unified model. However, existing methods suffer from **manifold entanglement** in mixed-source settings.

**HARMONY** addresses this by shifting from passive adaptation to **active geometric disentanglement**. It introduces an anchor-guided projection that separates the feature space into compact Voronoi regions, enabling a hierarchical retrieval mechanism with **sub-linear search complexity**.

## 📂 Project Structure

```bash
HARMONY/
├── conf/
│   └── config.yaml          # Main configuration file
├── scripts/
│   ├── 1_train_learner.py   # Stage 1: Anchor-Guided Disentanglement Training
│   ├── 2_build_experts.py   # Stage 2: Hierarchical Expert Index Construction
│   ├── 3_evaluate.py        # Stage 3: Inference and Evaluation
│   └── run.py               # One-click automation script
├── src/
│   ├── cluster.py           # Cluster Manager for Anchor Mining
│   ├── datasets.py          # Dataset Loader (DCASE/MIMII)
│   ├── ivf.py               # Inverted File Index (IVF) implementation
│   ├── models.py            # Backbone and Projection Head
│   └── utils.py             # Utility functions
├── requirements.txt         # Python dependencies
└── README.md

```

## 🚀 Getting Started

### 1. Prerequisites

Create a virtual environment and install the required packages. We recommend using **Python 3.8+**.

```bash
# 1. Create and activate environment
conda create -n harmony python=3.8
conda activate harmony

# 2. Install PyTorch (Recommended to install separately for correct CUDA support)
# Adjust the CUDA version (cu116) according to your driver
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# 3. Install other dependencies
pip install -r requirements.txt

```

### 2. Data Preparation

1. Download the **DCASE 2020 Task 2** or **MIMII** dataset.
2. Download the pre-trained **BYOL-A** weights (or AudioNTT2022).
3. Modify `conf/config.yaml` to point to your data and weight paths:

```yaml
paths:
  # Path to the dataset root directory
  data_root: "/path/to/dcase2020_dataset" 
  # Path to pre-trained backbone weights
  pretrained_weights: "/path/to/AudioNTT2022-BYOLA-64x96-1337.pth"
  # Output directory for logs and models
  output_root: "./outputs"

```

## ⚡ Usage

We provide a `run.py` script to automate the entire pipeline (Training  Indexing  Evaluation).

### One-Click Execution

Simply run the following command in the project root:

```bash
python scripts/run.py

```

This script will sequentially execute:

1. `1_train_learner.py`: Trains the projection head to disentangle the manifold.
2. `2_build_experts.py`: Builds the expert prototypes and IVF indices.
3. `3_evaluate.py`: Evaluates the model on the test set and reports AUC/pAUC/Latency.

### Step-by-Step Execution

If you prefer to run each step manually:

**Step 1: Train the Projection Head**
Learns the mapping  to separate acoustic modes.

```bash
python scripts/1_train_learner.py

```

**Step 2: Build Hierarchical Index**
Constructs the Voronoi tessellation (Experts) and local buckets.

```bash
python scripts/2_build_experts.py

```

**Step 3: Evaluate**
Performs inference using the hierarchical retrieval strategy.

```bash
python scripts/3_evaluate.py

```

## 📊 Performance

HARMONY achieves state-of-the-art performance on the DCASE 2020 dataset under the **Universal Unsupervised** setting (training on all machine types simultaneously without labels).

| Method | Mean AUC | Mean pAUC | Latency (CPU) |
| --- | --- | --- | --- |
| UniFlow | 0.754 | 0.664 | 19.6 ms |
| Audio-PC | 0.839 | 0.713 | 18.0 ms |
| **HARMONY** | **0.900** | **0.749** | **5.4 ms** |

## ⚙️ Configuration

Key hyperparameters in `conf/config.yaml`:

* `audio`:
* `n_mels`: 64
* `feature_d`: 3072 (Concatenated intermediate layers)


* `learner`:
* `n_anchors`: 8 (Number of spectral anchors)
* `output_dim`: 128 (Dimension of the intrinsic manifold)


* `cluster`:
* `ivf_centroids`: 64 (Number of buckets per expert)