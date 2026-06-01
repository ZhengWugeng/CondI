#!/usr/bin/env bash
# =============================================================================
# CondI Training Script — High Missing Rate  (ps=0.8, pm=0.8)
#
# Setting: 32 clients, Non-IID (label skew=0.5), PTB-XL 12-lead ECG
#   ps=0.8  →  80% of samples have a modality randomly erased
#   pm=0.8  →  80% of signal components masked within each affected sample
#
# This configuration corresponds to the severe data incompleteness setting
# reported in the paper.
#
# Prerequisites
#   1. Prepare the PTB-XL dataset:
#        See src/benchmark/RAW_DATA/PTBXL/README.md
#
#   2. Generate the federated task partition (run once):
#        cd src
#        python generate_fedtask.py \
#            --benchmark ptbxl_classification_lm \
#            --dist 1 --skew 0.5 \
#            --num_clients 32 --seed 0 \
#            --missing \
#            --sample_missing_ratio 0.5 \
#            --client_visible_modalities 8
#
# Usage (from project root):
#   bash scripts/train_ps08_pm08.sh
# =============================================================================

set -e
cd "$(dirname "$0")/../src"

python3 main.py \
    --task      ptbxl_classification_lm_cnum32_dist1_skew0.5_seed0_full_modal_local_missing \
    --model     condi_model \
    --algorithm multimodal.ptbxl_classification_lm.CondI \
    --sample    uniform \
    --aggregate other \
    \
    --num_rounds          70   \
    --proportion          0.33 \
    --num_epochs          3    \
    --learning_rate       0.01 \
    --lr_scheduler        0    \
    --learning_rate_decay 1.0  \
    --batch_size          32   \
    --test_batch_size     32   \
    --optimizer           SGD  \
    --gpu  0 \
    --seed 0 \
    \
    --ps  0.8 \
    --pm  0.8 \
    --sample_missing_ratio 0.5 \
    \
    --num_diff_steps      1    \
    --diffusion_timesteps 50   \
    --mask_ratio          0.2  \
    \
    --num_outer_loops   5    \
    --num_fedavg_rounds 0    \
    --graph_quantile    0.75 \
    --contrastive_weight 0.1
