import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import random
import copy

class ClientSubset(Subset):
    """Custom Subset for local missing setting
    """
    def __init__(self, dataset, indices, seed=0):
        super(ClientSubset, self).__init__(dataset, indices)
        self.missing_map = dict()
        self.seed=seed
        self.crop_length = self.dataset.crop_length
        
    def reset_missing_dict(self):
        self.missing_map = dict()
    
    def local_missing_setup(self, leads, ps, pm, local_missing_leads=None):
        """
        Local missing setup for client dataset. Will be called after clients have their own datasets
        args:
            leads: client leads
        """
        self.reset_missing_dict()
        # CondI sample-level missing setting uses within-sample partial missingness from the base dataset.
        # Keep client-level modality masking disabled in this mode.
        if getattr(self.dataset, "sample_level_missing", False):
            return

        random.seed(self.seed)
        if local_missing_leads is None:
            local_missing_leads = sorted(leads, key=lambda x: random.random())[:int(len(leads)*pm)]
            local_missing_leads.sort()
        
        missing_indices_leads = list()
        for i, ml in enumerate(local_missing_leads):
            random.seed(self.seed+i)
            tmp_indices = sorted(self.indices, key=lambda x: random.random())
            
            if i < len(local_missing_leads)-1:
                missing_indices = tmp_indices[:int(len(tmp_indices)*ps)]
                missing_indices_leads.append(missing_indices)   # list of list
            else:
                have_most_miss = set()
                for j, indices_lead in enumerate(missing_indices_leads):
                    if j==0: have_most_miss = set(indices_lead)
                    else:
                        have_most_miss = have_most_miss.intersection(set(indices_lead))
                valid_indices = list(set(tmp_indices) - have_most_miss)
                valid_indices = sorted(valid_indices, key=lambda x: random.random())
                
                missing_indices = valid_indices[:int(len(valid_indices)*ps)]
                missing_indices_leads.append(missing_indices)
                
            
        missing_indices = list()
        for ml_indices in missing_indices_leads:
            missing_indices += ml_indices
        missing_indices = set(missing_indices)
        print(f"Missing leads {local_missing_leads} from {leads} with {ps*100}%")
        
        missing_map = {}
        for i, ml in enumerate(local_missing_leads):
            for idx in missing_indices_leads[i]:
                if idx not in missing_map:
                    missing_map[idx] = []
                missing_map[idx].append(ml)
        self.missing_map = missing_map
        
    def global_missing_setup(self, leads, ps, pm):
        # only use for motivation
        self.reset_missing_dict()
        random.seed(self.seed)
        
        full_leads = list(range(12))
        missing_map = {}
        
        for idx in self.indices:
            for fl in full_leads:
                if fl not in leads: # missing
                    if idx not in missing_map:
                        missing_map[idx] = []
                    missing_map[idx].append(fl)
        self.missing_map = missing_map
        
        
    def save_data_dist(self, pm, ps, fn=None):
        
        cnt = [0 for i in range(12)]
        dist = np.zeros([12*40, self.__len__(), 3])
        # loop over dataset
        for i in range(self.__len__()):
            x, y = self.__getitem__(i)
            nleads = x.shape[0]
            for lid in range(nleads):
                if x[lid, 0] != -1:  
                    cnt[lid] += 1
                    dist[lid*40: (lid+1)*40, i, :] = np.array((255, 165, 0)) # orange
                else: 
                    dist[lid*40: (lid+1)*40, i, :] = (0, 0, 255)   # blue
                    continue
                
        # visualize
        fig = plt.figure(figsize=(8, 6))
        x_draw = [f'{i+1}' for i in range(12)]
        y1 = [c / self.__len__() for c in cnt]
        y2 = [1 for i in range(12)]
        plt.bar(x_draw, y2, color='blue')
        plt.bar(x_draw, y1, color='orange')
        plt.rc('xtick', labelsize=18)
        plt.rc('ytick', labelsize=18)
        plt.xlabel('Modality index', fontsize=20)
        plt.ylabel('Propotion in the dataset', fontsize=20)
        plt.tight_layout()
        fn = f"config_{pm}_{ps}.png" if fn is None else "config_"+fn
        fn_dist = f"dist_{pm}_{ps}.png" if fn is None else "dist_"+fn
        fig.savefig(
            os.path.join(f"./visualize_notebooks/plots/", fn), )
        plt.close(fig)
        
        if "client" not in fn:
            fig = plt.figure(figsize=(8, 6))
            plt.imshow(dist.astype(np.uint8)[:, 0:600, :])
            plt.yticks(ticks=np.arange(0, 40 * 12 + 1, 40),
                    labels=np.arange(0, 13))
            plt.xticks(ticks=np.arange(0, 600, 200).tolist() + [600],
                    labels=np.arange(0, 600, 200).tolist() + [600])
            plt.rc('xtick', labelsize=18)
            plt.rc('ytick', labelsize=18)
            plt.xlabel('Sample index', fontsize=20)
            plt.ylabel('Modality index', fontsize=20)
            fig.savefig(
                os.path.join("./visualize_notebooks/plots/", fn_dist), )
        
        
    def adjust_crop_length(self, crop_length):
        self.dataset.adjust_crop_length(crop_length)
        self.crop_length = crop_length
        return
    
    def _apply_missing(self, x, orig_idx):
        if orig_idx in self.missing_map:
            x = x.copy()
            for lead in self.missing_map[orig_idx]:
                x[lead, :] = -1
        return x

    def __getitem__(self, idx):
        if isinstance(idx, list):
            outs = list()
            for i in idx:
                orig_idx = self.indices[i]
                res = self.dataset[orig_idx]
                if isinstance(res, tuple) and len(res) == 2:
                    x, y = res
                    x = self._apply_missing(x, orig_idx)
                    outs.append((x, y))
                else:
                    x, imp, y = res
                    x = self._apply_missing(x, orig_idx)
                    outs.append((x, imp, y))
            return outs
        orig_idx = self.indices[idx]
        res = self.dataset[orig_idx]
        if isinstance(res, tuple) and len(res) == 2:
            x, y = res
            x = self._apply_missing(x, orig_idx)
            return (x, y)
        else:
            x, imp, y = res
            x = self._apply_missing(x, orig_idx)
            return (x, imp, y)
    
    def __getitems__(self, indices):
        outs = list()
        for idx in indices:
            orig_idx = self.indices[idx]
            res = self.dataset[orig_idx]
            if isinstance(res, tuple) and len(res) == 2:
                x, y = res
                x = self._apply_missing(x, orig_idx)
                outs.append((x, y))
            else:
                x, imp, y = res
                x = self._apply_missing(x, orig_idx)
                outs.append((x, imp, y))
        return outs
    
    def __len__(self):
        return len(self.indices)
    
class ImputedClientSubset(ClientSubset):
    def __init__(self, dataset, indices, seed=0):
        super().__init__(dataset, indices, seed)
        
    def local_missing_setup(self, leads, ps, pm, local_missing_leads=None):
        """
        Local missing setup for client dataset. Will be called after clients have their own datasets.
        In Imputed version, the batch contains 3 elements: x, imputed, y.
        args:
            leads: client leads
        """
        super().local_missing_setup(leads, ps, pm, local_missing_leads)