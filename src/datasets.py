import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from sklearn.preprocessing import StandardScaler


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", scaler=None) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.scaler = scaler

        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt")).numpy()
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))

        if self.scaler:
            self.X = self.scaler.fit_transform(self.X.reshape(-1, self.X.shape[-1])).reshape(self.X.shape)

        self.X = torch.tensor(self.X, dtype=torch.float32)

        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
            
            
            
    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]