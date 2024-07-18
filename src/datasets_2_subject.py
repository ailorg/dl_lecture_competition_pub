import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from glob import glob
from scipy.signal import resample


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.resample_rate = 2  # リサンプリングレート
        self.baseline_correct = True # ベースライン補正のフラグ
        self.scale_data = True  # スケーリングのフラグ
        
        # データ数を取得
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = np.load(X_path)  # Numpy配列としてデータを読み込む
        
        # ベースライン補正を適用
        if self.baseline_correct:
            X = self.apply_baseline_correction(X)
        
        # リサンプリングを適用
        if self.resample_rate > 1:
            X = self.resample_data(X)
                
        # スケーリングを適用
        if self.scale_data:
            X = self.scale(X)
        
        X = torch.from_numpy(X)
        
        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path)).long()
        
        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path))
            
            return X, y, subject_idx
        else:
            return X, subject_idx
    
    def resample_data(self, X):
        # データのリサンプリングを行う関数
        resampled_data = []
        for channel_data in X:
            resampled_channel_data = resample(channel_data, int(len(channel_data) / self.resample_rate))
            resampled_data.append(resampled_channel_data)
        return np.array(resampled_data)
    
    def apply_baseline_correction(self, X):
        # ベースライン補正を適用する関数
        baseline = np.mean(X[:, :100], axis=1, keepdims=True)  # 最初の100時間点の平均をベースラインとする例
        X_corrected = X - baseline
        return X_corrected
    
    def scale(self, X):
        # スケーリングを適用する関数
        X_mean = np.mean(X, axis=1, keepdims=True)
        X_std = np.std(X, axis=1, keepdims=True)
        X_scaled = (X - X_mean) / X_std
        return X_scaled
    
    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]
    
    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]
