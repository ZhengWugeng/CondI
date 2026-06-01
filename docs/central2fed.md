# Converting a Traditional ML Task to a Federated One

This guide explains how the CondI benchmark system partitions a standard dataset into a federated task and how the training system loads it back at runtime.

## Concepts

The benchmark system has two main components:

- **`TaskGen`** — downloads the dataset, partitions it across virtual clients, and saves the result to disk as a `data.json` file.
- **`TaskPipe`** — loads the saved `data.json` back into memory and returns typed dataset objects that the federated training loop can consume.

```
benchmark/
└── ptbxl_classification_lm/
    ├── core.py        # TaskGen, TaskPipe, TaskCalculator
    ├── dataset.py     # PTBXLReduceDataset (torch.utils.data.Dataset)
    ├── preprocess.py  # Raw WFDB → .npy files
    ├── subset.py      # Client-level dataset wrapper with missing simulation
    └── model/
        └── condi_model.py   # Model loaded by fflow.initialize()
```

## TaskGen

`TaskGen` (defined in `core.py`) handles:

1. **Loading data** — calls `dataset.py` to load the preprocessed `.npy` files into memory.
2. **Partitioning** — splits training indices across `num_clients` clients using the specified distribution (`dist`, `skew`).
3. **Modality assignment** — assigns a random subset of `client_visible_modalities` leads to each client.
4. **Saving** — calls `TaskPipe.save_task()` to write `data.json` to `fedtask/<task_name>/`.

Run from `src/`:
```bash
python generate_fedtask.py \
    --benchmark ptbxl_classification_lm \
    --dist 1 --skew 0.5 \
    --num_clients 32 --seed 0 \
    --missing --sample_missing_ratio 0.5 \
    --client_visible_modalities 8
```

## TaskPipe

`TaskPipe.load_task()` is called automatically by `fflow.initialize()` at training startup. It:

1. Reads `data.json` to get the class path and constructor arguments for the original dataset.
2. Instantiates `PTBXLReduceDataset` (train and test).
3. Wraps each client's index list in a `ClientSubset` (or `ImputedClientSubset`), which applies the sample-level missing mask at `__getitem__` time.
4. Returns `(train_datas, valid_datas, test_data, client_names, modalities_list)`.

## TaskCalculator

`TaskCalculator` (in `core.py`) decouples task-specific logic from the federated optimizer:

- `get_data_loader(dataset, batch_size)` — returns a `DataLoader`.
- `train_one_step(model, batch_data)` — runs a forward pass and returns `{'loss': ..., 'outputs': ...}`.
- `test(model, dataset)` — runs inference and returns per-class and overall accuracy.

The federated algorithm (`CondI.py`) calls these methods without knowing the internal structure of the dataset or model, which makes it straightforward to swap in a different benchmark.

## Adding a New Dataset

1. Create `benchmark/my_dataset/` with the same structure as `ptbxl_classification_lm/`.
2. Implement `TaskGen`, `TaskPipe`, and `TaskCalculator` in `core.py`.
3. Place your model in `benchmark/my_dataset/model/my_model.py` as a class named `Model`.
4. Run `generate_fedtask.py --benchmark my_dataset ...`.
5. Train with `--task my_dataset_... --model my_model --algorithm ...`.
