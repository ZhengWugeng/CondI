# Algorithm Directory

This directory contains all federated learning algorithm implementations used in CondI experiments.

## Structure

```
algorithm/
├── fedbase.py                          # Base Server and Client classes
└── multimodal/
    └── ptbxl_classification_lm/
        ├── CondI.py                    # CondI (main algorithm)
        ├── fedavg.py                   # FedAvg baseline
        ├── mifl.py                     # MIFL baseline
        ├── baselines/                  # Other comparison methods
        │   ├── FedInMM.py
        │   ├── fedmsplit.py
        │   ├── fednova.py
        │   ├── fedprox.py
        │   └── mifl_contrastive2.py
        └── condi/                      # CondI supporting modules
            ├── fedavg_1enc_12enc.py            # CondI variant: FedAvg backbone
            ├── fedprox_1enc_12enc.py           # CondI variant: FedProx backbone
            └── imputer/                        # CSDI diffusion imputer modules
                ├── main_model.py
                ├── config_mm.py
                ├── csdi_mm_moe.py
                └── diff_models.py
```

## CondI Algorithm (`CondI.py`)

Each federated round runs two phases on every participating client:

**Phase A — Diffusion Imputation**
- Trains `per_modality_imputer` and `cond_encoder` using DDPM noise-prediction loss.
- All other model components are frozen.
- Uses AdamW optimizer with `diff_lr` learning rate.

**Phase B — Classification**
- Trains all feature extractors, modality embeddings, auxiliary gate, and classifier.
- The diffusion imputer is frozen; its output is used as a detached imputed signal.
- Uses Adam optimizer with `learning_rate`.

After local training, the server aggregates model weights using modality-aware FedAvg:
- Per-lead encoders are averaged only over clients that observed that lead.
- Shared components (imputer, classifier, etc.) are globally averaged.

### Running CondI

```bash
python main.py \
    --task ptbxl_classification_lm_cnum32_dist1_skew0.5_seed0_full_modal_local_missing \
    --model condi_model \
    --algorithm multimodal.ptbxl_classification_lm.CondI \
    --aggregate other \
    --num_diff_steps 1 \
    --diffusion_timesteps 50 \
    --mask_ratio 0.2 \
    --num_outer_loops 5
```

## Baseline Algorithms

| Algorithm | File | `--algorithm` flag |
|---|---|---|
| FedAvg | `fedavg.py` | `multimodal.ptbxl_classification_lm.fedavg` |
| FedProx | `baselines/fedprox.py` | `multimodal.ptbxl_classification_lm.baselines.fedprox` |
| MIFL | `mifl.py` | `multimodal.ptbxl_classification_lm.mifl` |
| FedMSplit | `baselines/fedmsplit.py` | `multimodal.ptbxl_classification_lm.baselines.fedmsplit` |
| FedInMM | `baselines/FedInMM.py` | `multimodal.ptbxl_classification_lm.baselines.FedInMM` |

## Adding a New Algorithm

1. Create `algorithm/multimodal/ptbxl_classification_lm/my_algorithm.py`.
2. Define `Server(BasicServer)` and `Client(BasicClient)` classes.
3. Run with `--algorithm multimodal.ptbxl_classification_lm.my_algorithm`.

The `BasicServer` and `BasicClient` base classes in `fedbase.py` provide standard FedAvg behavior. Override only the methods you need to change (e.g., `Client.train()`, `Server.aggregate()`).
