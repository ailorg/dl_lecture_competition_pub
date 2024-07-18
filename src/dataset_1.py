import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from scipy.signal import resample, butter, filtfilt
from sklearn.preprocessing import StandardScaler

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", target_fs: int = 128) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.target_fs = target_fs
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
        # Initialize scaler
        self.scaler = StandardScaler()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i]
        
        # Original sampling rate (this needs to be known or provided)
        original_fs = 1000  # Example: replace with your actual original sampling rate
        
        # Number of samples in the resampled data
        num_samples = int(X.shape[1] * self.target_fs / original_fs)
        
        # Resample the data
        X_resampled = resample(X, num_samples, axis=1)
        
        # Apply bandpass filter
        lowcut = 1.0  # Example value, set according to your requirements
        highcut = 40.0  # Example value, set according to your requirements
        # X_filtered = self.bandpass_filter(X_resampled, lowcut, highcut, self.target_fs)
        X_filtered=X_resampled
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X_filtered.T).T  # Transpose to fit the scaler and then transpose back
        
        if hasattr(self, "y"):
            return torch.tensor(X_scaled, dtype=torch.float32), self.y[i], self.subject_idxs[i]
        else:
            return torch.tensor(X_scaled, dtype=torch.float32), self.subject_idxs[i]

    @staticmethod
    def bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data, axis=1)
        return y
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
