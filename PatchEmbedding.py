import torch
import torch.nn as nn
import numpy as np

class TimeSeriesPatchEncoder(nn.Module):
    def __init__(self, input_dim, d_model, kernel_size, stride, padding):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.d_model = d_model

    def forward(self, x, timestamps):
        # x: (B, L, M), timestamps: (B, L)
        B, L, M = x.shape

        # 1. Patch embedding via Conv1D
        x = x.permute(0, 2, 1)  # -> (B, M, L)
        patches = self.conv1d(x)  # -> (B, d_model, num_patches)
        patches = patches.permute(0, 2, 1)  # -> (B, num_patches, d_model)

        # 2. Real-time positional encoding
        num_patches = patches.size(1)
        patch_times = []
        for b in range(B):
            sample_t = []
            for j in range(num_patches):
                b_idx = j * self.stride - self.padding
                l_idx = b_idx + self.kernel_size
                idx_range = range(max(b_idx, 0), min(l_idx, L))
                t_patch = timestamps[b, list(idx_range)]
                median_t = torch.median(t_patch).item()
                sample_t.append(median_t)
            patch_times.append(sample_t)

        patch_times = torch.tensor(patch_times)  # (B, num_patches)

        # 3. Sinusoidal encoding
        pe = self._sinusoidal_encoding(patch_times, self.d_model)  # (B, num_patches, d_model)

        return patches + pe

    def _sinusoidal_encoding(self, times, d_model):
        B, L = times.shape
        pe = torch.zeros((B, L, d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        for i in range(B):
            for pos in range(L):
                t = times[i, pos]
                pe[i, pos, 0::2] = torch.sin(t * div_term)
                pe[i, pos, 1::2] = torch.cos(t * div_term)
        return pe

# 假数据参数
batch_size = 2
time_steps = 100
num_features = 4  # M
d_model = 32  # embedding维度（简化版）

# 模拟时间戳（不规则）
timestamps = torch.sort(torch.rand(batch_size, time_steps) * 100, dim=1).values  # (B, L)

# 模拟多特征时间序列数据
x = torch.randn(batch_size, time_steps, num_features)  # (B, L, M)

encoder = TimeSeriesPatchEncoder(
    input_dim=num_features,
    d_model=d_model,
    kernel_size=8,
    stride=4,
    padding=2
)

out = encoder(x, timestamps)
print("Final embedding shape:", out.shape)  # (B, num_patches, d_model)
