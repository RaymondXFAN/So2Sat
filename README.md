# Scale-Aware Multi-Modal Fusion for So2Sat LCZ42

Reproducibility package for a controlled two-scale study of **seven Sentinel-1 / Sentinel-2 fusion strategies** on the So2Sat LCZ42 benchmark.

This repository contains the complete training pipeline, all 53 runs' metrics, the gate-statistics tooling, and the scripts that generate every figure in the paper. It does **not** contain the manuscript.

---

## What this study asks

Most multi-modal fusion papers report results at a single data scale, usually the full benchmark, and usually from a single random seed. Two questions follow: *does the ranking of fusion strategies survive a change of training-set size?* and *can a single-seed comparison distinguish a real effect from initialisation noise?*

We train seven fusion operators under an identical recipe at two scales — a 5,000-sample training subset and the full 352,366-sample training set — with three seeds per variant and scale (42 runs), plus 11 extended-seed runs as dispersion cross-checks (53 runs in total).

---

## Key results

### At 320K, strategy-level differences are not resolvable

Six non-degenerate variants occupy a **59.22–60.99% OA band**, a 1.77-point spread comparable to the cross-seed noise. The only pairwise difference that clears significance is `no_gate` vs `concat` (Δ = +1.06, t = 7.79, df = 2). A single-modality baseline matches the full fusion model within noise (60.65% vs 60.99%, t = 0.48).

| Variant | 5K OA | 320K OA | Δ (5K→320K) |
|---|---|---|---|
| `no_gate` | 51.35 ± 1.59 | 60.34 ± 0.74 | +8.99 |
| **`full`** | **42.48 ± 15.35** | **60.99 ± 0.46** | +18.51 |
| `concat` | 52.04 ± 0.77 | 59.28 ± 0.68 | +7.24 |
| `no_stmn` | 50.56 ± 1.60 | 59.28 ± 1.37 | +8.72 |
| `no_ada` | 51.62 ± 1.84 | 59.22 ± 1.30 | +7.60 |
| `s2_only` | 42.13 ± 12.24 | 60.65 ± 1.57 | +18.52 |
| `s1_only` | 23.48 ± 5.69 | 35.19 ± 0.40 | +11.71 |

*Three-seed mean ± std (seeds 42/123/456). All metrics computed on the full 24,188-sample test split at both scales.*

### Dispersion, not the mean, is what changes with scale

The STMN texture branch adds −8.08 points at 5K and +1.71 at 320K in mean terms, but the more informative statistic is variance:

| Variant | 5K σ | 320K σ | Contraction |
|---|---|---|---|
| `full` | **15.35** | **0.46** | **33×** |
| `no_stmn` | 1.60 | 1.37 | 1.2× |

At 5K the `full` model produced one collapsed run (24.76%, seed 42) among three; `s2_only` produced one as well (28.02%, seed 456). All four lower-variance variants stayed inside a 1.5-point band. The branch therefore behaves as a variance amplifier under data scarcity and as a stable gain once data are sufficient.

### The gate does not collapse at small scale

Mean Ada-MBA gate activations, measured on the complete 24,119-sample validation split:

| Scale | α(S2–S1) | α(S2–STMN) | α(S1–STMN) | Mean |
|---|---|---|---|---|
| 5K (5 seeds) | 0.274 ± 0.320 | 0.359 ± 0.219 | 0.495 ± 0.214 | **0.376 ± 0.166** |
| 320K (3 seeds) | 0.028 ± 0.022 | 0.211 ± 0.278 | 0.252 ± 0.181 | **0.164 ± 0.122** |

The gate is *more* active, and far less reproducible, when data are scarce. At 320K all seeds converge toward suppression, which is why the gated `full` and the gate-free `no_gate` are statistically indistinguishable (t = 1.16).

---

## Repository layout

```
.
├── README.md                 this file
├── DEPLOY.md                 full reproduction guide (environment, staged runs, FAQ)
├── CHANGELOG.md              version history
├── LICENSE
├── requirements.txt
├── code/                     training pipeline
│   ├── config.py             paths, hyper-parameters, cache redirection
│   ├── datasets.py           HDF5 / .pt dataset loaders
│   ├── train.py              main training loop, 7 fusion modes
│   ├── models/               MobileNetV3 encoders, STMN, Ada-MBA
│   └── prepare_so2sat.py     normalisation statistics
├── tools/
│   ├── dump_gates_full.py    gate statistics over the full validation split
│   ├── plot_figures.py       regenerates all four figures (400 dpi PNG + PDF)
│   └── audit_metrics.py      recomputes every table entry from the raw CSVs
├── scripts/
│   ├── run_main_grid.sh      42 runs (7 variants × 2 scales × 3 seeds)
│   └── run_extended_seeds.sh 11 extra runs for dispersion checks
├── results/
│   ├── all53_metrics.csv     per-run OA / AA / Kappa / macro-F1
│   ├── gate_full_val.csv     per-checkpoint gate values
│   └── README.md             data dictionary and measurement protocol
└── figures/                  fig1–fig4, 400 dpi PNG + vector PDF
```

---

## Quick start

```bash
git clone <this-repo>
cd so2sat-fusion-repro
pip install -r requirements.txt

# 1. Put the So2Sat LCZ42 HDF5 files under data/raw/ (see DEPLOY.md §3)
# 2. Check the environment
python -c "import torch; print(torch.cuda.is_available())"

# 3. Smoke test (~30 s, does not touch the main result directories)
python code/train.py --mode no_gate --seed 7777 --n_train 5000 --epochs 2

# 4. Full reproduction (~9 GPU-hours on one RTX 4090; see DEPLOY.md §4)
bash scripts/run_main_grid.sh
bash scripts/run_extended_seeds.sh

# 5. Gate statistics and figures
python tools/dump_gates_full.py
python tools/plot_figures.py
```

---

## Reproducibility checks

If your environment matches, the following values must reproduce **exactly** (bit-identical seeds, same data):

| Check | Expected |
|---|---|
| `no_stmn_5k`, 10-seed mean OA | 50.53 ± 1.51 |
| `no_stmn` @320K, 5-seed mean OA | 59.85 ± 1.28 |
| `full_5k_seed1024` OA | 52.15 |
| `no_gate_5k_seed123` OA | 52.88 |
| `full` @320K, 3-seed mean OA | 60.99 ± 0.46 |

Run `python tools/audit_metrics.py` to recompute every cell of the main table from `results/all53_metrics.csv` and confirm the numbers used in the paper.

Divergence larger than ±0.1 usually indicates a different PyTorch/CUDA build or a mismatched data file; see `DEPLOY.md` §6.

---

## Data

So2Sat LCZ42 is publicly available from the original authors (Zhu et al., 2020). This repository does not redistribute the imagery. You need:

```
data/
├── raw/
│   ├── training.h5      352,366 patches, 10 optical + 8 SAR channels, 32×32
│   ├── validation.h5     24,119 patches
│   └── testing.h5        24,188 patches
└── stats.json            per-channel mean/std for normalisation
```

`stats.json` is generated by `code/prepare_so2sat.py` (about 10 minutes; it only computes statistics, no data duplication).

Note that the 5K experiments restrict **only the training set**. Validation and test splits are always the full ones, so every number in `results/` is computed on identical evaluation data across scales.

---

## Hardware and cost

All runs were produced on a single NVIDIA RTX 4090 D (24 GB). Total compute is approximately 7 GPU-hours (≈4.6 h at 320K, ≈2.5 h at 5K). Peak memory is under 8 GB.

---

## Citation

If you use this code or the released metrics, please cite the accompanying manuscript (under review) and the original dataset paper:

```bibtex
@article{zhu2020so2sat,
  title   = {So2Sat LCZ42: A Benchmark Data Set for the Classification of Global Local Climate Zones},
  author  = {Zhu, Xiao Xiang and Hu, Jingliang and Qiu, Chunping and Shi, Yilei and Kang, Jian and others},
  journal = {IEEE Geoscience and Remote Sensing Magazine},
  volume  = {8},
  number  = {3},
  pages   = {76--89},
  year    = {2020}
}
```

---

## License

Code released under the MIT License (see `LICENSE`). The released metric tables and figures are provided under CC BY 4.0.

## Contact

Corresponding author: Weiwu Ma — 767266781@qq.com
