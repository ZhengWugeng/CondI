from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import random

# Path to the raw (read-only) PTB-XL dataset.
# Set the PTBXL_RAW_PATH environment variable, or edit the default below,
# to point at the directory that holds ptbxl_database.csv, scp_statements.csv
# and the records100/ folder.
_RAW_DATASET_PATH = os.environ.get('PTBXL_RAW_PATH', '/path/to/raw/PTBXL')

class PTBXLReduceDataset(Dataset):
    def __init__(self, root, download=True,
                 standard_scaler=True, train=True,
                 crop_length=250, valid=False, pm=0.0, ps=0.0,
                 impute_path=None, sample_missing_ratio=0.5):
        self.root = root
        self.standard_scaler = standard_scaler
        self.train = train
        self.crop_length = crop_length
        self.pm=pm
        self.ps=ps
        self.imputation = (impute_path != None)
        # New setting: missingness is generated inside each sample (time points/components),
        # not by dropping whole modalities per client.
        self.sample_level_missing = True
        self.sample_missing_ratio = float(max(0.0, min(1.0, sample_missing_ratio)))

        # Check if preprocessed files exist at the project-local root
        npy_check = os.path.join(self.root, 'x_train.npy')
        if not os.path.exists(npy_check):
            if download:
                print(f'Preprocessed files not found at {self.root}. '
                      f'Generating from raw dataset at {_RAW_DATASET_PATH} ...')
                from .preprocess import preprocess_ptbxl
                preprocess_ptbxl(source_dir=_RAW_DATASET_PATH, output_dir=self.root)
                print('Preprocessing done!')
            else:
                raise RuntimeError(
                    f'Preprocessed dataset not found at {self.root}. '
                    f'Set download=True to generate from {_RAW_DATASET_PATH}'
                )
        if self.train:
            self.x = np.load(os.path.join(self.root, 'x_train.npy'), allow_pickle=True, fix_imports=True)
            self.y = np.load(os.path.join(self.root, 'y_train.npy'), allow_pickle=True, fix_imports=True)
        else:
            self.x = np.load(os.path.join(self.root, 'x_test.npy'), allow_pickle=True, fix_imports=True)
            self.y = np.load(os.path.join(self.root, 'y_test.npy'), allow_pickle=True, fix_imports=True)
        if self.standard_scaler:
            self.ss = pickle.load(open(os.path.join(self.root, 'standard_scaler.pkl'), 'rb'))
            x_tmp = list()
            for x in self.x:
                x_shape = x.shape
                x_flat = x.reshape(-1, x.shape[-1])  # (1000, 12)
                x_transformed = self.ss.transform(x_flat)
                x_tmp.append(x_transformed.reshape(x_shape))
            self.x = np.array(x_tmp)
            
        self.impute = None
        impute_path = None
        if impute_path is not None:
            if self.train:
                self.impute = np.load(os.path.join(impute_path, 'x_train.npy'), allow_pickle=True, fix_imports=True)
            else:
                self.impute = np.load(os.path.join(impute_path, 'x_test.npy'), allow_pickle=True, fix_imports=True)
            
    def standardize(self, x):
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True)
        return (x - mean) / std
    
    def adjust_crop_length(self, crop_length):
        print("Reset crop length to: ", crop_length)
        self.crop_length = crop_length
        return
            
    def normalize(self, x): 
        pmax = np.max(x, axis=1, keepdims=True) 
        pmin = np.min(x, axis=1, keepdims=True) 
        return (x - pmin) / (pmax - pmin + 1e-9)
    
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        start_idx = random.randint(0, x.shape[0] - self.crop_length - 1)
        # start_idx=0
        x = x[start_idx:start_idx + self.crop_length].transpose().astype(np.float32)
        
        x = self.normalize(x).astype(np.float32)
        
        # apply sample-level partial missing
        # if train: randomly missing 50%
        # if test: fix missing 50%
        
        # in the current dataset x is of shape (12, crop_length)
        if self.train:
            # Train: each sample gets a fresh random component-level mask.
            mask = np.ones(x.shape, dtype=np.float32)
            total = mask.size
            miss = int(total * self.sample_missing_ratio)
            missing_flat = np.random.choice(total, miss, replace=False)
            mask.ravel()[missing_flat] = 0.0
            x = x * mask - (1 - mask)  # -1 sentinel for missing entries
        else:
            # Test: each sample uses deterministic component-level mask.
            mask = np.ones(x.shape, dtype=np.float32)
            total = mask.size
            miss = int(total * self.sample_missing_ratio)
            rng = np.random.RandomState(index)
            missing_flat = rng.choice(total, miss, replace=False)
            mask.ravel()[missing_flat] = 0.0
            x = x * mask - (1 - mask)

        x = x.astype(np.float32)
        y = y.astype(np.float32)
        if self.impute is None:
            return x, y
        else:
            imp = self.impute[index]    # LxC
            imp = imp[:self.crop_length].transpose().astype(np.float32)
            imp = self.normalize(imp).astype(np.float32)
            
            return x, imp, y
    
if __name__ == '__main__':
    dataset = PTBXLReduceDataset(root='./benchmark/RAW_DATA/PTBXL', standard_scaler=True, train=True)
