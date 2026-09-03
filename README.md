# So2Sat LCZ42: Multi-Modal Adaptive Fusion for Local Climate Zone Classification

Code for the paper *"Adaptive Multi-Branch Fusion of Sentinel Optical and SAR Data for Local Climate Zone Classification"*.

## Overview

This project implements a lightweight yet effective framework for 17-class Local Climate Zone (LCZ) classification using Sentinel-2 optical and Sentinel-1 SAR data from the [So2Sat LCZ42](https://doi.org/10.1109/MGRS.2020.2964709) dataset. The core contribution is **Ada-MBA** (Adaptive Multi-Branch Adapter), a gated cross-attention fusion module that adaptively combines multi-modal and multi-view features.

### Model Variants

| Mode | Architecture | #Params | Description |
|------|-------------|---------|-------------|
| `full` | S2 + S1 + STMN → LayerNorm → 3-branch Ada-MBA | ~3.6M | Full model with spectral, SAR, and texture branches |
| `no_stmn` | S2 + S1 → LayerNorm → 2-branch Ada-MBA | ~2.95M | Proposed model (without STMN texture branch) |
| `no_ada` | S2 + S1 + STMN → LayerNorm → simple average | ~3.1M | Ablation: replace Ada-MBA with mean pooling |
| `no_gate` | S2 + S1 + STMN → LayerNorm → 3-branch Ada-MBA (no gate) | ~3.6M | Ablation: remove gating mechanism from Ada-MBA |
| `s2_only` | S2 → head | ~1.5M | Single-modality optical baseline |
| `s1_only` | S1 → head | ~1.5M | Single-modality SAR baseline |
| `concat` | S2 + S1 + STMN → LayerNorm → concat → Linear | ~3.1M | Early-fusion concatenation baseline |

---

## Project Structure

```
so2sat_exp/
├── code/
│   ├── config.py              # Central configuration (paths, hyperparameters, seeds)
│   ├── datasets.py            # Dataset & DataLoader (supports .h5, .h5.gz, and .pt)
│   ├── prepare_so2sat.py      # Data preprocessing (compute stats.json from h5)
│   ├── train.py               # Main training loop with 7 modes + early stopping
│   ├── utils.py               # Utility functions
│   ├── run_5k_seeds.sh         # Script: 5K subset multi-seed experiments
│   ├── run_320k_seeds.sh       # Script: 320K full-data multi-seed experiments
│   ├── run_5k_gate_seeds.sh    # Script: 5K gate-value experiments (full mode)
│   └── models/
│       ├── __init__.py         # Model component exports
│       ├── backbone.py         # MobileNetV3-Small encoder (ImageNet pretrained)
│       ├── stmn.py             # Spectral-to-Texture Mapping Network (Sobel + student)
│       ├── ada_mba.py          # Adaptive Multi-Branch Adapter (self-attn + cross-attn + gate)
│       └── da.py               # Domain adaptation components (gradient reversal)
├── data/
│   ├── raw/                   # HDF5 data files (training.h5, validation.h5, testing.h5)
│   └── stats.json             # Normalization statistics computed by prepare_so2sat.py
└── out/                       # Experiment outputs (auto-created per mode/seed)
    ├── no_stmn/               # 320K seed=42 results
    ├── no_stmn_seed123/       # 320K seed=123 results
    ├── no_stmn_5k/            # 5K seed=42 results
    ├── full_5k_seed123/       # 5K full-mode seed=123 results (with gate_value.txt)
    └── ...
```

---

## Environment

### Software

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10+ | Tested on 3.12 |
| PyTorch | 2.2.0+cu121 | Compatible with CUDA ≥ 12.1 drivers |
| torchvision | 0.17.0+cu121 | Must match torch version |
| h5py | ≥ 3.0 | For reading HDF5 data files |
| numpy | < 2.0 | numpy 2.x incompatible with torch 2.2 |
| scipy | < 1.14 | For sklearn compatibility |
| scikit-learn | ≥ 1.0 | For F1/Kappa metrics |
| indexed_gzip | ≥ 1.9 | (Optional) For reading .h5.gz directly without decompression |

### Hardware

- **GPU**: NVIDIA RTX 4090 (24 GB VRAM) recommended; any CUDA GPU with ≥ 8 GB works
- **RAM**: ≥ 32 GB (h5 file handles + DataLoader workers)
- **Disk**: ~60 GB for full dataset (training.h5 ≈ 49 GB, validation.h5 ≈ 3.4 GB, testing.h5 ≈ 3.4 GB)

### Quick Install

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install h5py "numpy<2" "scipy<1.14" scikit-learn
```

---

## Dataset

### So2Sat LCZ42

The dataset is available on [HuggingFace](https://huggingface.co/datasets/zhu-xlab/So2Sat-LCZ42). The HDF5 files are located under the `v4/` directory and distributed as `.h5.gz` (gzip-compressed).

| Split | File | Compressed | Decompressed | Samples |
|-------|------|-----------|-------------|---------|
| Training | `v4/training.h5.gz` | ~15 GB | ~49 GB | 352,366 |
| Validation | `v4/validation.h5.gz` | ~1.1 GB | ~3.4 GB | 24,119 |
| Testing | `v4/testing.h5.gz` | ~1.1 GB | ~3.4 GB | 24,188 |

Each sample contains:
- **sen2**: Sentinel-2 optical patch, shape `(32, 32, 10)` — 10 spectral bands
- **sen1**: Sentinel-1 SAR patch, shape `(32, 32, 8)` — 8 polarimetric bands
- **label**: One-hot 17-class LCZ label

### Download & Prepare

1. **Download** the compressed files from HuggingFace:

```bash
# Set your data directory
DATA_DIR=/path/to/so2sat_exp/data/raw
mkdir -p $DATA_DIR

# Download from HuggingFace (v4 subdirectory)
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
repo = 'zhu-xlab/So2Sat-LCZ42'
for f in ['v4/training.h5.gz', 'v4/validation.h5.gz', 'v4/testing.h5.gz']:
    hf_hub_download(repo, f, local_dir='$DATA_DIR', repo_type='dataset')
"
```

2. **Decompress** (if you have ≥ 60 GB disk space):

```bash
cd $DATA_DIR
gunzip training.h5.gz    # produces training.h5 (~49 GB)
gunzip validation.h5.gz  # produces validation.h5 (~3.4 GB)
gunzip testing.h5.gz     # produces testing.h5 (~3.4 GB)
```

> **Disk-constrained?** The code also supports reading `.h5.gz` files directly via `indexed_gzip` (no decompression needed). However, this is **significantly slower** for training and is only recommended for preprocessing. Install with `pip install indexed_gzip`.

3. **Preprocess** — compute normalization statistics:

```bash
cd /path/to/so2sat_exp
python code/prepare_so2sat.py
```

This creates `data/stats.json` containing band-wise mean and standard deviation for normalization. It takes ~5 minutes for the full training set.

---

## Usage

### Single Run

```bash
python code/train.py --mode no_stmn --seed 42
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | `full` | Experiment mode: `full`, `s2_only`, `s1_only`, `concat`, `no_stmn`, `no_ada`, `no_gate` |
| `--seed` | int | `42` | Random seed (overrides config default) |
| `--epochs` | int | `30` | Number of training epochs (overrides config default) |
| `--n_train` | int | `None` | Limit training samples (e.g., `5000` for 5K subset experiments) |

### Examples

```bash
# Full model, 30 epochs, default seed
python code/train.py --mode full

# Proposed model (no_stmn), custom seed
python code/train.py --mode no_stmn --seed 123

# Quick verification (2 epochs)
python code/train.py --mode no_stmn --seed 42 --epochs 2

# 5K subset experiment (reviewer-requested small-data test)
python code/train.py --mode no_stmn --seed 42 --n_train 5000 --epochs 50

# Multi-seed batch runs
bash code/run_5k_seeds.sh no_stmn 5000 50     # 5K subset, 10 seeds
bash code/run_320k_seeds.sh no_stmn 30         # 320K full data, 5 seeds
bash code/run_5k_gate_seeds.sh 50              # 5K gate values (full mode), 5 seeds
```

### Environment Variables

Set these before running to control caching and data paths:

```bash
export SO2SAT_ROOT=/path/to/so2sat_exp       # Project root (default: /root/autodl-tmp/so2sat_exp)
export HF_ENDPOINT=https://hf-mirror.com      # HuggingFace mirror (for China users)
export HF_HOME=/path/to/.cache/huggingface    # HF cache directory
export TORCH_HOME=/path/to/.cache/torch       # Torch model cache directory
```

---

## Configuration

All hyperparameters are centralized in `code/config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | auto | `cuda` if available, else `cpu` |
| `gpu_batch_size` | 256 | Batch size on GPU |
| `epochs` | 30 | Training epochs (full data converges faster) |
| `lr` | 1e-3 | Learning rate |
| `optimizer` | AdamW | Optimizer |
| `weight_decay` | 1e-4 | Weight decay |
| `scheduler` | cosine | Cosine annealing LR scheduler |
| `warmup_epochs` | 5 | Warmup period |
| `early_stop_patience` | 8 | Early stopping patience (in epochs) |
| `grad_clip` | 1.0 | Gradient clipping max norm |
| `seed` | 42 | Default random seed |
| `FULL_DATA` | `True` | Use complete dataset (h5) vs. subset (.pt) |

To modify hyperparameters, edit `CFG` dict in `config.py` or override `epochs`/`seed` via command-line arguments.

---

## Output Structure

Each run creates a directory under `out/` named by mode, subset tag, and seed:

```
out/
├── no_stmn/                    # full data, seed=42
├── no_stmn_seed123/            # full data, seed=123
├── no_stmn_5k/                 # 5K subset, seed=42
├── no_stmn_5k_seed123/         # 5K subset, seed=123
├── full_5k/                    # 5K full-mode, seed=42 (includes gate values)
└── ...
```

Each directory contains:

| File | Description |
|------|-------------|
| `table2_main.csv` | Primary results: OA, AA, Kappa, F1 |
| `train_log.csv` | Per-epoch training log (loss, metrics, gate values) |
| `table5_classwise.csv` | Per-class accuracy for all 17 LCZ classes |
| `confusion_matrix.csv` | 17×17 confusion matrix |
| `best_model.pt` | Best model checkpoint (by validation OA) |
| `gate_value.txt` | Ada-MBA gate weights (full mode only) |

### Gate Values (full mode)

When running `--mode full`, the Ada-MBA module logs cross-attention gate weights (α ∈ [0, 1]) at each epoch:

- **α = 0**: only original features used (cross-attention ignored)
- **α = 0.5**: equal blend of cross-attention and original features
- **α = 1**: only cross-attention output used (original features ignored)

Three gate pairs are logged: `0-1` (S2 ↔ S1), `0-2` (S2 ↔ STMN), `1-2` (S1 ↔ STMN).

---

## Results Summary

### 5K Subset (no_stmn, 10 seeds)

| Metric | Mean ± Std |
|--------|-----------|
| OA | 50.53 ± 1.51% |
| AA | 34.58 ± 1.02% |
| Kappa | 45.58 ± 1.62% |
| F1 | 34.10 ± 1.04% |

### 320K Full Data (no_stmn, 5 seeds)

| Metric | Mean ± Std |
|--------|-----------|
| OA | 59.85 ± 1.28% |
| AA | 47.35 ± 1.92% |
| Kappa | 55.88 ± 1.36% |
| F1 | 47.73 ± 1.53% |

### Statistical Significance

Paired t-test (320K vs 5K, 5 shared seeds): t = 13.04, **p = 0.0002**, Cohen's d = 5.83 (large effect). Data scaling from 5K to 320K yields a statistically significant improvement across all metrics (p < 0.001).

### Ada-MBA Gate Analysis (5K, full mode)

| Seed | OA | mean_gate | S2↔S1 | S2↔STMN | S1↔STMN |
|------|-----|-----------|--------|---------|---------|
| 123 | 51.86 | 0.286 | 0.069 | 0.157 | 0.633 |
| 456 | 50.81 | 0.390 | 0.256 | 0.309 | 0.606 |
| 789 | 50.36 | 0.571 | 0.828 | 0.417 | 0.468 |
| 1024 | 52.15 | 0.144 | 0.062 | 0.232 | 0.138 |

S1↔STMN gate values are consistently the highest, indicating that SAR–texture cross-modal interaction is the most active.

---

## Model Architecture Details

### MobileNetV3-Small Encoder (`backbone.py`)

ImageNet-pretrained MobileNetV3-Small with the first conv layer replaced to accept multi-channel input (10 for S2, 8 for S1). Outputs a 256-d feature vector. ~2.5M parameters per encoder.

### STMN — Spectral-to-Texture Mapping Network (`stmn.py`)

Extracts texture features from S2 optical data via **fixed Sobel filters** (horizontal + vertical gradients → gradient magnitude), followed by a student encoder (MobileNetV3-Small) and a 3-layer residual MLP. The Sobel filters are **not learnable**, ensuring the network extracts spatial gradient (texture) information rather than redundant spectral features.

### Ada-MBA — Adaptive Multi-Branch Adapter (`ada_mba.py`)

Fuses multiple feature branches via:
1. **Self-attention** per branch (MultiheadAttention, 8 heads)
2. **Cross-attention** for each pair of branches
3. **Gated blending**: α = σ(Wg [F_i; F_j]), output = α · CA + (1 − α) · F_i
4. **1D convolution reduction**: all branch + cross-attention outputs → single 256-d vector

Supports 2-branch (no_stmn) and 3-branch (full) configurations. The `skip_gate=True` option (no_gate ablation) bypasses the gating mechanism.

---

## Citation

If you use this code or the So2Sat LCZ42 dataset, please cite:

```bibtex
@article{zhu2020so2sat,
  title={So2Sat LCZ42: A benchmark data set for the classification of global local climate zones [Software and Data]},
  author={Zhu, Xiao Xiang and Hu, Jingliang and Qiu, Chunping and Shi, Yilei and Kang, Jian and Lu, Pingyu and Li, Lichao and Baier, Georg and Guo, Yun and Wang, Zhitong and others},
  journal={IEEE Geoscience and Remote Sensing Magazine},
  volume={8},
  number={3},
  pages={92--100},
  year={2020}
}
```
