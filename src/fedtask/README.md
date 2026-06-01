# fedtask

This directory stores generated federated task files. Each subdirectory is one experimental configuration and contains a `data.json` file that records:

- How the dataset is partitioned across clients
- Which modalities each client observes
- Train / validation / test index splits

## Generating a fedtask

Run `generate_fedtask.py` from `src/`:

```bash
cd src

# IID split, 6 clients, PTB-XL
python generate_fedtask.py \
    --benchmark ptbxl_classification_lm \
    --dist 0 \
    --skew 0 \
    --num_clients 6 \
    --seed 2026 \
    --missing \
    --sample_missing_ratio 0.2 \
    --client_visible_modalities 8

# Non-IID split, 32 clients, PTB-XL
python generate_fedtask.py \
    --benchmark ptbxl_classification_lm \
    --dist 1 \
    --skew 0.5 \
    --num_clients 32 \
    --seed 0 \
    --missing \
    --sample_missing_ratio 0.5 \
    --client_visible_modalities 8
```

### Key parameters

| Parameter | Description |
|---|---|
| `--benchmark` | Dataset benchmark (`ptbxl_classification_lm` or `edf_classification_lm`) |
| `--dist` | Distribution type: `0` = IID, `1` = non-IID label skew |
| `--skew` | Degree of non-IID skewness (0 = uniform, 1 = extreme) |
| `--num_clients` | Number of federated clients |
| `--seed` | Random seed for reproducibility |
| `--missing` | Enable heterogeneous modality assignment across clients |
| `--sample_missing_ratio` | Fraction of signal components masked per sample |
| `--client_visible_modalities` | Number of ECG leads each client can observe |

## Naming convention

Generated directories are named:

```
{benchmark}_cnum{N}_dist{D}_skew{S}_seed{seed}_full_modal_local_missing
```

Example: `ptbxl_classification_lm_cnum32_dist1_skew0.5_seed0_full_modal_local_missing`

## Notes

- The `.json` files only store **indices** into the original dataset, not the data itself. The `benchmark/RAW_DATA/` directory must be populated before training.
- Do **not** commit the generated `data.json` files to version control unless you want to fix the exact data split for reproducibility.
