# PTB-XL Dataset

This directory holds the preprocessed PTB-XL data files required to run CondI experiments.

## Expected files

After preprocessing, this directory should contain:

```
PTBXL/
├── x_train.npy          # Training signals  (N_train, 1000, 12)  float64
├── y_train.npy          # Training labels   (N_train,)            int64  (classes 0-4)
├── x_test.npy           # Test signals      (N_test,  1000, 12)  float64
├── y_test.npy           # Test labels       (N_test,)             int64
└── standard_scaler.pkl  # Fitted StandardScaler for per-lead normalization
```

Labels map to five diagnostic superclasses:

| Label | Class |
|---|---|
| 0 | NORM (Normal) |
| 1 | MI (Myocardial Infarction) |
| 2 | STTC (ST/T-wave Change) |
| 3 | CD (Conduction Disturbance) |
| 4 | HYP (Hypertrophy) |

## How to prepare

### Option A. Automatic (recommended)

The dataset is preprocessed automatically the first time you run training, provided the raw PTB-XL files exist at the path configured in `dataset.py` (`_RAW_DATASET_PATH`). Update that variable to point at your local raw-data directory before launching training.

### Option B. Manual preprocessing

1. Download PTB-XL version 1.0.1 from PhysioNet at https://physionet.org/content/ptb-xl/1.0.1/.

2. Extract the archive into a directory of your choice. That directory must contain:
   - `ptbxl_database.csv`
   - `scp_statements.csv`
   - `records100/` (100 Hz recordings)

3. Run preprocessing from `src/`:
   ```bash
   python -c "
   from benchmark.ptbxl_classification_lm.preprocess import preprocess_ptbxl
   preprocess_ptbxl(
       source_dir='/path/to/raw/PTBXL',
       output_dir='./benchmark/RAW_DATA/PTBXL'
   )
   "
   ```

Preprocessing takes roughly 5 to 10 minutes and requires about 2 GB of disk space.

## Notes

- The raw `records100/` WFDB files are not required after preprocessing completes.
- Do not commit the `.npy` or `.pkl` files to version control. They are already listed in `.gitignore`.
- The `data.json` fedtask files reference this directory as `./benchmark/RAW_DATA/PTBXL` relative to `src/`.
