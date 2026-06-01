import os
import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import StandardScaler
import pickle

def preprocess_ptbxl(source_dir=None,
                     output_dir='./benchmark/RAW_DATA/PTBXL'):
    """
    Preprocess PTB-XL dataset into npy format expected by the project.

    Args:
        source_dir: path to the raw PTBXL dataset (read-only, not modified).
            If None, falls back to the PTBXL_RAW_PATH environment variable
            or the placeholder '/path/to/raw/PTBXL'.
        output_dir: path where preprocessed .npy / .pkl files are written.
    """
    if source_dir is None:
        source_dir = os.environ.get('PTBXL_RAW_PATH', '/path/to/raw/PTBXL')

    os.makedirs(output_dir, exist_ok=True)

    # Load database
    db_path = os.path.join(source_dir, 'ptbxl_database.csv')
    db = pd.read_csv(db_path, index_col='ecg_id')

    # Filter for records100 (500Hz -> 100Hz resampling)
    db = db[db['filename_lr'].str.contains('records100')]

    # Map SCP codes to classes (simplified 5-class classification)
    # NORM=0, MI=1, STTC=2, CD=3, HYP=4
    scp_to_class = {
        'NORM': 0,
        'MI': 1,
        'STTC': 2,
        'CD': 3,
        'HYP': 4
    }

    # Load SCP statements to map to diagnostic class
    agg_df = pd.read_csv(os.path.join(source_dir, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    # Get records with valid SCP codes
    valid_records = []
    labels = []

    for idx, row in db.iterrows():
        scp_codes = eval(row['scp_codes'])
        classes = set()
        for code in scp_codes.keys():
            if code in agg_df.index:
                diag_class = agg_df.loc[code].diagnostic_class
                if diag_class in scp_to_class:
                    classes.add(scp_to_class[diag_class])
        
        if len(classes) > 0:
            valid_records.append(row['filename_lr'])
            # Since some records might have multiple classes, we pick the first one for single-label classification
            labels.append(list(classes)[0])

    print(f"Found {len(valid_records)} valid records")

    # Load ECG signals
    signals = []
    final_labels = []
    
    # Shuffle valid records to ensure balanced classes if we take a subset
    import random
    combined = list(zip(valid_records, labels))
    random.seed(42)
    random.shuffle(combined)
    valid_records, labels = zip(*combined)
    valid_records = list(valid_records)
    labels = list(labels)

    # Use 3170 to match the expected dataset size for this project setup
    for i, filename in enumerate(valid_records[:3170]):  
        try:
            record = wfdb.rdrecord(os.path.join(source_dir, filename))
            sig = record.p_signal  # (1000, 12) - 10s at 100Hz, 12 leads
            if sig.shape[0] >= 1000:
                sig = sig[:1000]  # Crop to 1000 samples
            else:
                # Pad if shorter
                pad_len = 1000 - sig.shape[0]
                sig = np.pad(sig, ((0, pad_len), (0, 0)), mode='constant')
            signals.append(sig)
            final_labels.append(labels[i])
        except Exception as e:
            # print(f"Error loading {filename}: {e}")
            continue

    signals = np.array(signals)
    labels = np.array(final_labels)

    print(f"Loaded {len(signals)} signals with shape {signals.shape}")

    # Split train/test (80/20)
    n_train = int(0.8 * len(signals))
    x_train = signals[:n_train]
    y_train = labels[:n_train]
    x_test = signals[n_train:]
    y_test = labels[n_train:]

    # Standardize
    ss = StandardScaler()
    x_train_flat = x_train.reshape(-1, x_train.shape[-1])  # (n_samples * 1000, 12)
    ss.fit(x_train_flat)

    # Save to output_dir (project-local, does not touch source_dir)
    np.save(os.path.join(output_dir, 'x_train.npy'), x_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'x_test.npy'), x_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    with open(os.path.join(output_dir, 'standard_scaler.pkl'), 'wb') as f:
        pickle.dump(ss, f)

    print("Preprocessing complete!")

if __name__ == '__main__':
    preprocess_ptbxl()