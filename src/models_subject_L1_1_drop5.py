import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        num_subjects: int = 4,
        hid_dim: int = 128
    ) -> None:
        super().__init__()
        self.subject_embedding = nn.Embedding(num_subjects, hid_dim)  # 被験者の埋め込み層を追加

        self.blocks = nn.Sequential(
            ConvBlock(in_dim=in_channels, out_dim=hid_dim, kernel_size=3),
            ConvBlock(in_dim=hid_dim, out_dim=hid_dim, kernel_size=15),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        return self.head(X)
        subject_embedding = self.subject_embedding(subject_idx)
        subject_embedding = subject_embedding.unsqueeze(2).expand_as(X)
        X = X + subject_embedding  # 埋め込みを特徴マップに追加
        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.5,
        l1_strength: float = 0.1  # L1正則化の強さ
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)
        
        # L1正則化を重みに適用
        self.conv0.weight_regularizer = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.conv1.weight_regularizer = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        return self.dropout(X)
