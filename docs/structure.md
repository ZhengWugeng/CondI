# CondI Data and Model Flow

## Dimension Notation

```
B  = batch_size
L  = sequence length after crop (default 250)
C  = channels / ECG leads (12)
D  = hidden dimension (128)
K  = num_classes (5)
M  = number of modalities (12)
T  = diffusion_timesteps (50)
```

## Full Training Flow (ASCII)

```
╔══════════════════════════════════════════════════════════════════════╗
║ Entry: main.py                                                       ║
║  option = fflow.read_option()                                        ║
║  server = fflow.initialize(option)   # loads data, model, clients   ║
║  server.run()                        # starts FL rounds             ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║ Step 1: Dataset Loading (PTBXLReduceDataset)                         ║
║  x_train.npy : (N_train, 1000, 12)  float64                         ║
║  y_train.npy : (N_train,)           int64  labels 0–4               ║
║  x_test.npy  : (N_test,  1000, 12)                                  ║
║  y_test.npy  : (N_test,)                                            ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║ Step 2: Preprocessing & Normalization                                ║
║  StandardScaler applied per-lead (fitted on train set)              ║
║  Random crop: start ~ Uniform[0, 1000 - L - 1]                      ║
║  x_crop: (12, L)  then normalize to [0,1] per lead                  ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║ Step 3: Sample-Level Missing Simulation                              ║
║  mask = ones(12, L)                                                  ║
║  miss = int(12 * L * sample_missing_ratio)  # default 50%           ║
║  randomly zero out `miss` positions                                  ║
║  x_masked = x * mask - (1 - mask)   # -1 = missing sentinel         ║
║  Test: deterministic mask (RandomState(index))                       ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║ Step 4: DataLoader  →  batch (B, 12, L), labels (B,)                ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║ Model Forward Pass  (condi_model.py)                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  obs_mask        = (x != -1)              (B, 12, L)                ║
║  x_obs           = x.clamp(min=0)         (B, 12, L)                ║
║  lead_obs_mask   = obs_mask.mean(-1) > 0  (B, 12)  bool             ║
║                                                                      ║
║  w_mod  : modality_embeddings[lead]        (B, 12, 128)             ║
║  w_ins  : ModalityEncoder + Inception1D    (B, 12, 128)  mask-aware ║
║  w_cond : CondEncoder(mean of others)      (B, 12, 128)             ║
║                                                                      ║
║  ─── CSDI Diffusion Imputer ──────────────────────────────────────  ║
║  cond = [w_cond ; w_ins]                   (B*12, 256)              ║
║  Training : DDPM noise-pred loss + x0_hat (1-step)                  ║
║  Inference: 50-step reverse diffusion → x0_hat                      ║
║                                                                      ║
║  w_imputed : encoder on x0_hat * miss_mask (B, 12, 128)             ║
║  x_completed = x_obs + x0_hat * miss_mask  (B, 12, L)               ║
║  f_main    : Inception1D on x_completed    (B, 12, 250)             ║
║                                                                      ║
║  aux = [w_mod, w_ins, w_imputed, w_cond]   (B, 12, 512)             ║
║  gated_aux = aux_gate(aux)                 (B, 12, 128)             ║
║  f_out = concat(f_main, gated_aux)         (B, 12, 378)             ║
║                                                                      ║
║  logits = Classifier(f_out.flatten)        (B, 5)                   ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║ Loss                                                                 ║
║  L = CrossEntropy(logits, y) + 0.1 * loss_imputation               ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║ Federated Round (CondI.py)                                           ║
║                                                                      ║
║  Phase A — Imputation (NUM_DIFF_STEPS steps, AdamW)                 ║
║    Trainable: per_modality_imputer, cond_encoder                    ║
║    Frozen:    everything else                                        ║
║                                                                      ║
║  Phase B — Classification (NUM_EPOCHS * batches, Adam)              ║
║    Trainable: all feature extractors, classifier, aux_gate          ║
║    Frozen:    per_modality_imputer                                   ║
║                                                                      ║
║  Aggregation (server):                                              ║
║    per-lead encoders   → weighted avg over clients owning that lead ║
║    shared components   → global weighted FedAvg                     ║
╚══════════════════════════════════════════════════════════════════════╝
```

## Tensor Size Summary

```
x_raw           (B, 12, 250)    input ECG (after crop), -1 = missing
obs_mask        (B, 12, 250)    1 = observed, 0 = missing
x_obs           (B, 12, 250)    missing replaced with 0
x_completed     (B, 12, 250)    x_obs + imputed signal

w_mod           (B, 12, 128)    learnable modality identity
w_ins           (B, 12, 128)    observed-signal features
w_cond          (B, 12, 128)    cross-modal context
w_imputed       (B, 12, 128)    imputed-signal features
imputer_cond    (B*12, 256)     [w_cond ; w_ins]
x0_hat          (B, 12, 250)    diffusion output

f_main          (B, 12, 250)    features from x_completed
gated_aux       (B, 12, 128)    aux_gate([w_mod,w_ins,w_imputed,w_cond])
f_out           (B, 12, 378)    concat(f_main, gated_aux)
logits          (B, 5)          classification output
```

## Key File Paths

```
src/main.py                                          entry point
src/utils/fflow.py                                   federated flow controller
src/algorithm/fedbase.py                             base Server / Client
src/benchmark/ptbxl_classification_lm/dataset.py    dataset
src/benchmark/ptbxl_classification_lm/core.py       TaskGen / TaskPipe
src/benchmark/ptbxl_classification_lm/model/
  condi_model.py                                     CondI model
  imputer/main_model.py                              CSDI diffusion model
  imputer/diff_models.py                             denoising network
src/algorithm/multimodal/ptbxl_classification_lm/
  CondI.py                                           federated algorithm
```
