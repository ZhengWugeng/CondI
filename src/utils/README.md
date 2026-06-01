# Utils

Core utilities for the CondI federated learning system.

## Modules

### `fflow.py` — Federated training flow

Controls the end-to-end execution: parses CLI options, initializes the federated system (server, clients, dataset), and starts training.

Key functions:
- `read_option()` — parses all command-line arguments including CondI-specific ones (`--num_diff_steps`, `--diffusion_timesteps`, `--mask_ratio`, etc.)
- `initialize(option)` — loads the fedtask, constructs the model, and instantiates Server and Client objects
- `setup_seed(seed)` — sets random seeds for reproducibility

### `fmodule.py` — Model arithmetic

Provides operator overloading for `torch.nn.Module` so that federated aggregation can be written as:

```python
aggregated = sum(w_k * model_k for w_k, model_k in zip(weights, models))
```

All CondI models inherit from `FModule` (defined here) rather than directly from `nn.Module`.

### `result_analysis.py` — Experiment analysis

Loads training records from `fedtask/<task>/record/` and produces plots and summary tables. Controlled by a YAML config file:

```yaml
task: ptbxl_classification_lm_cnum32_dist1_skew0.5_seed0_full_modal_local_missing
header:
  - CondI
  - fedavg
ploter:
  plot:
    - x: communication_round
      y: test_acc1
info:
  final_value:
    - test_acc1
```

Run:
```bash
cd utils
python result_analysis.py --config res_config.yml --save_figure
```

### `system_simulator.py` — Systemic heterogeneity

Simulates client availability, connectivity, completeness, and timeliness heterogeneity. Controlled by the `--availability`, `--connectivity`, `--completeness`, `--timeliness` flags (default: `IDL` = ideal, no heterogeneity).

### `logger/` — Training loggers

- `basic_logger.py` — default logger; records per-round test loss and accuracy, per-class accuracy, and timing
- `simple_logger.py` — lightweight logger for validation metrics only

To use a custom logger, create `utils/logger/my_logger.py` with a `Logger` class and pass `--logger my_logger` at runtime.
