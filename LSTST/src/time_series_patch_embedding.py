import torch.nn as nn


class TimeSeriesPatchEmbedding(nn.Module):
    def __init__(self, input_dim, d_model, kernel_size, stride):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,

        )
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        # x: (B, L, M)
        x = x.permute(0, 2, 1)  # -> (B, M, L)
        x = self.conv1d(x)      # -> (B, d_model, num_patches)
        x = x.permute(0, 2, 1)  # -> (B, num_patches, d_model)
        return x