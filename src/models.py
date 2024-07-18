import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


# class BasicConvClassifier(nn.Module):
#     def __init__(
#         self,
#         num_classes: int,
#         seq_len: int,
#         in_channels: int,
#         hid_dim: int = 128
#     ) -> None:
#         super().__init__()

#         self.blocks = nn.Sequential(
#             ConvBlock(in_channels, hid_dim),
#             ConvBlock(hid_dim, hid_dim),
#         )

#         self.head = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             Rearrange("b d 1 -> b d"),
#             nn.Linear(hid_dim, num_classes),
#         )

#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         """_summary_
#         Args:
#             X ( b, c, t ): _description_
#         Returns:
#             X ( b, num_classes ): _description_
#         """
#         X = self.blocks(X)

#         return self.head(X)
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConvClassifier(nn.Module):
    def __init__(self, num_classes, seq_len, num_channels, num_subjects, embedding_dim=50):
        super(BasicConvClassifier, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.embedding_dim = embedding_dim
        
        # Calculate the size after convolutions to define the fully connected layer
        self.seq_len = seq_len
        
        # Assuming the input size (seq_len) does not change after conv layers with padding=1
        conv_output_size = 128 * 35
        
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256 + embedding_dim, 128)  # Concatenate subject embeddings
        self.fc3 = nn.Linear(128, num_classes)
        self.subject_embedding = nn.Embedding(num_subjects, embedding_dim)  # Embedding layer for subject indices

    def forward(self, x, subject_idxs):
        # print(f"Input shape: {x.shape}")
        x = F.relu(self.conv1(x))
        # print(f"After conv1: {x.shape}")
        x = F.relu(self.conv2(x))
        # print(f"After conv2: {x.shape}")
        x = x.view(x.size(0), -1)  # Flatten
        # print(f"After flatten: {x.shape}")
        x = F.relu(self.fc1(x))
        # print(f"After fc1: {x.shape}")
        subject_embeds = self.subject_embedding(subject_idxs)
        # print(f"Subject embeddings: {subject_embeds.shape}")
        x = torch.cat((x, subject_embeds), dim=1)
        # print(f"After concatenation: {x.shape}")
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)