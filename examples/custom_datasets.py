import numpy as np
import pickle

import torch
from torch.utils import data

# For now, CustomDataset is a wrapper class that implements additional
# features (subsampling, logits parsing, etc.) on top.
class CustomDataset(data.Dataset):
    def __init__(self, base_dataset, subsample=None, logits=None):
        self.base_dataset = base_dataset
        self.valid_idx = np.arange(len(base_dataset))
        self.subsample = subsample
        if subsample: 
            if subsample['mode'] == 'uniform':
                num_samples = int(len(base_dataset) * subsample['sample_frac'])
                idx = np.linspace(0, len(base_dataset), num_samples,
                                  endpoint=False, dtype=np.int32)
                self.valid_idx = self.valid_idx[idx]
                print(self.valid_idx)
        # Assume that logits are never provided directly by the base_dataset
        # but in an auxilliary pkl path.
        self.logits = logits
        self.logits_data = [None for _ in range(len(base_dataset))]
        if logits:
            with open(logits['path'], 'rb') as f:
                self.logits_data = pickle.load(f)
            self.logits_T = logits['temperature']

    def __getitem__(self, idx):
        target_idx = self.valid_idx[idx]
        data, label = self.base_dataset[target_idx]
        if self.logits:
            logits = self.logits_data[target_idx] / self.logits_T
        else:
            logits = label
        return data, label, logits

    def __len__(self):
        return len(self.valid_idx)


class RepeatDataset(object):
    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        return self.times * self._ori_len
