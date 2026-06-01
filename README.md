# CondI: Conditional Imputation for Within-Modality Missingness in Multi-Modal Federated Learning

Official code release for the paper *Conditional Imputation for Within-Modality Missingness in Multi-Modal Federated Learning*, accepted at the **CVPR 2026 FedVision Workshop**.

CondI is a federated framework that explicitly addresses **within-modality missingness** in multimodal time-series learning. Instead of representing unobserved data through implicit alignment or learnable missing embeddings, CondI restores the underlying signal via a conditional diffusion model guided by cross-modal context, and then trains downstream encoders on the explicitly imputed data.

<p align="center">
  <img src="./assets/structure.png" width="100%" alt="CondI framework overview"/>
</p>

## Overview

Real-world federated multimodal systems, especially in clinical scenarios such as MIMIC-IV, PTB-XL, and Sleep-EDF, face two simultaneous challenges. The first is modality-level missingness across clients, where each institution observes only a subset of sensors. The second is within-modality missingness, where irregular sampling or sensor detachment fractures the temporal dependencies of physiological signals.

Existing federated baselines handle these gaps implicitly through architectural alignment or learnable missing embeddings. These passive strategies often fail to recover the underlying data distribution and lose semantic coherence under severe missingness. CondI takes a different route by explicitly generating the missing components conditioned on the available multimodal context, and by exposing the imputed raw data to the downstream classifier.

The framework runs locally on each client in two phases per federated round.

**Phase A: Conditional Estimation.** A conditional diffusion module denoises the unobserved components given the observed signals and a learnable cross-modal conditional embedding $\mathcal{W}_{\text{cond}}$. The conditional embedding parameterizes the correlation between any source modality and any target modality, so it can be aggregated across clients via FedAvg without exchanging raw data.

**Phase B: Task Optimization.** Modality-specific encoders extract instance features $\mathcal{W}_{\text{ins}}$ and combine them with the modality identity embedding $\mathcal{W}_{\text{mod}}$ and the conditional embedding $\mathcal{W}_{\text{cond}}$ for classification. At inference, the imputed data produced in Phase A is fed back into the trained encoders, so the classifier receives a holistic and coherent representation rather than a fragmented one.

## Datasets

The paper evaluates CondI on three publicly available multimodal clinical datasets.

| Dataset | Modalities | Task | Reference |
|---|---|---|---|
| **PTB-XL** | 12 ECG leads | 5-class diagnostic classification | [PhysioNet](https://physionet.org/content/ptb-xl/) |
| **Sleep-EDF** | 2 EEG, EOG, Resp, EMG | 5-class sleep-stage scoring | [PhysioNet](https://www.physionet.org/content/sleep-edfx/) |
| **MIMIC-IV** | EHR vitals and labs | 48-hour in-hospital mortality, length-of-stay | [PhysioNet](https://physionet.org/content/mimiciv/) |

This initial release includes the full PTB-XL benchmark pipeline. Sleep-EDF and MIMIC-IV pipelines will be added in a follow-up release.

Each modality is treated as a separate stream that may be entirely missing for a client (modality-level), partially missing within a sample (within-modality), or both.

## Project Structure

```
CondI/
├── assets/                            Figures used in the README
├── docs/
│   ├── structure.md                   Detailed model architecture
│   └── central2fed.md                 Guide for adding a new benchmark
├── scripts/                           Top-level launch scripts
│   ├── train_ps02_pm02.sh             Low-missingness setting
│   └── train_ps08_pm08.sh             Severe-missingness setting
├── src/
│   ├── main.py                        Entry point
│   ├── generate_fedtask.py            Partition the dataset into a federated task
│   ├── requirements.txt
│   ├── algorithm/
│   │   ├── fedbase.py                 Base Server and Client classes
│   │   └── multimodal/ptbxl_classification_lm/
│   │       ├── CondI.py               Main federated algorithm
│   │       ├── baselines/             FedProx, MIFL, FedMSplit, FedInMM, FedNova
│   │       └── condi/                 Supporting modules
│   │           ├── fedavg_1enc_12enc.py
│   │           ├── fedprox_1enc_12enc.py
│   │           └── imputer/           CSDI-based diffusion imputer
│   ├── benchmark/
│   │   ├── ptbxl_classification_lm/   PTB-XL benchmark
│   │   └── RAW_DATA/                  Preprocessed dataset files
│   ├── fedtask/                       Generated federated task files (data.json)
│   ├── script/                        Per-experiment shell scripts
│   └── utils/
│       ├── fflow.py                   Federated training flow controller
│       ├── fmodule.py                 Model arithmetic for aggregation
│       └── logger/
└── README.md
```

For a deeper walkthrough of the model and tensor shapes see [`docs/structure.md`](docs/structure.md). For instructions on plugging in a new benchmark see [`docs/central2fed.md`](docs/central2fed.md).

## Requirements

```bash
conda create -n condi python=3.10
conda activate condi
pip install -r src/requirements.txt
```

Core dependencies are `torch>=2.0`, `numpy`, `scikit-learn`, `scipy`, `tqdm`, and optionally `wandb`.

## Dataset Preparation

### PTB-XL

1. Download PTB-XL version 1.0.1 from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.1/) and place the raw files where the preprocessing script can find them. The directory must contain `ptbxl_database.csv`, `scp_statements.csv`, and the `records100/` folder.
2. Run preprocessing once:
   ```bash
   cd src
   python -c "from benchmark.ptbxl_classification_lm.preprocess import preprocess_ptbxl; preprocess_ptbxl()"
   ```
   This produces `x_train.npy`, `y_train.npy`, `x_test.npy`, `y_test.npy`, and `standard_scaler.pkl` under `src/benchmark/RAW_DATA/PTBXL/`. See [`src/benchmark/RAW_DATA/PTBXL/README.md`](src/benchmark/RAW_DATA/PTBXL/README.md) for full details.

## Usage

### Step 1. Generate a federated task

```bash
cd src
python generate_fedtask.py \
    --benchmark ptbxl_classification_lm \
    --dist 1 --skew 0.5 \
    --num_clients 6 --seed 2026 \
    --missing \
    --sample_missing_ratio 0.5 \
    --client_visible_modalities 8
```

This writes `src/fedtask/ptbxl_classification_lm_cnum6_dist1_skew0.5_seed2026_full_modal_local_missing/data.json`. The file stores indices and per-client modality assignments, so the raw data must already be prepared.

### Step 2. Train CondI

The two convenience scripts in `scripts/` cover the canonical low-missingness and severe-missingness settings.

```bash
# 20% sample-level + 20% within-modality missingness
bash scripts/train_ps02_pm02.sh

# 80% sample-level + 80% within-modality missingness
bash scripts/train_ps08_pm08.sh
```

For full control, call `main.py` directly:

```bash
cd src
python main.py \
    --task      ptbxl_classification_lm_cnum6_dist1_skew0.5_seed2026_full_modal_local_missing \
    --model     condi_model \
    --algorithm multimodal.ptbxl_classification_lm.CondI \
    --sample    uniform \
    --aggregate other \
    --num_rounds 70  --proportion 0.5 --num_epochs 3 \
    --learning_rate 0.01 --batch_size 32 --test_batch_size 32 \
    --ps 0.2 --pm 0.2 --sample_missing_ratio 0.5 \
    --num_diff_steps 1 --diffusion_timesteps 50 --mask_ratio 0.2 \
    --num_outer_loops 5 --num_fedavg_rounds 0 --graph_quantile 0.75 \
    --gpu 0 --seed 0
```

Pre-built scripts for the full grid of $(p_s, p_w)$ configurations live in `src/script/ptbxl_classification_lm/`, covering CondI and all baselines under IID and Non-IID partitions.

### Quick smoke test

A minimal 2-round run on the included 6-client fedtask:

```bash
cd src
python main.py \
    --task ptbxl_classification_lm_cnum6_dist0_skew0_seed2026_full_modal_local_missing \
    --model condi_model \
    --algorithm multimodal.ptbxl_classification_lm.CondI \
    --sample uniform --aggregate other \
    --num_rounds 2 --proportion 1.0 --num_epochs 1 \
    --learning_rate 0.01 --batch_size 16 --test_batch_size 32 \
    --gpu 0 --seed 0 \
    --num_outer_loops 1 --num_fedavg_rounds 0 \
    --graph_quantile 0.75 --sample_missing_ratio 0.5 \
    --num_diff_steps 1 --diffusion_timesteps 10 --mask_ratio 0.2
```

## Key Hyperparameters

| Parameter | Description | Typical value |
|---|---|---|
| `--num_rounds` | Federated communication rounds | 70 (PTB-XL, Sleep-EDF), 30 (MIMIC-IV) |
| `--num_epochs` | Local epochs per round | 3 |
| `--proportion` | Fraction of clients sampled per round | 0.5 |
| `--ps` | Sample-level missing ratio | 0.2 or 0.8 |
| `--pm` | Within-modality missing probability | 0.2 or 0.8 |
| `--num_diff_steps` | Diffusion training steps per round | 1 |
| `--diffusion_timesteps` | Total DDPM timesteps $T$ | 50 |
| `--mask_ratio` | Fraction of observed values used as pseudo-imputation targets | 0.2 |
| `--num_outer_loops` | Nonparametric aggregation iterations | 5 |
| `--graph_quantile` | Edge pruning quantile in the client similarity graph | 0.75 |

## Citation

If you find this work useful, please cite the FedVision workshop paper:

```bibtex
@inproceedings{zheng2026condi,
  title     = {Conditional Imputation for Within-Modality Missingness in Multi-Modal Federated Learning},
  author    = {Zheng, Wugeng and Kan, Ziwen and Wang, Katie and Chen, Chen and Wang, Song},
  booktitle = {CVPR Workshop on Federated Learning for Computer Vision (FedVision)},
  year      = {2026}
}
```

## Acknowledgements

This codebase was developed on top of the released implementation of PEPSY. We thank the authors for making their work publicly available.

```bibtex
@inproceedings{nguyen2025pepsy,
  title     = {Learning Reconfigurable Representations for Multimodal Federated Learning with Missing Data},
  author    = {Nguyen, Duong M. and Hoang, Trong Nghia and Huynh, Thanh Trung and Nguyen, Quoc Viet Hung and Nguyen, Phi Le},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```

The federated learning skeleton is built on top of [easyFL](https://github.com/WwZzz/easyFL), and the conditional diffusion imputer is adapted from [CSDI](https://github.com/ermongroup/CSDI). We also thank the authors of PTB-XL, Sleep-EDF, and MIMIC-IV for releasing the datasets that made this study possible.
