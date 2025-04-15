import torch.nn as nn

from LSTST.src.RealTimePositionalEncoding import RealTimePositionalEncoding
from LSTST.src.TimeSeriesPatchEmbedding import TimeSeriesPatchEmbedding


class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim, d_model, kernel_size, stride, padding):
        super().__init__()
        self.patch_embedder = TimeSeriesPatchEmbedding(input_dim, d_model, kernel_size, stride, padding)
        self.pe_encoder = RealTimePositionalEncoding(d_model)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x, timestamps):
        # x: (B, L, M), timestamps: (B, L)
        B, L, M = x.shape
        patches = self.patch_embedder(x)  # (B, num_patches, d_model)
        pe = self.pe_encoder.compute(timestamps, L, self.kernel_size, self.stride, self.padding)
        return patches + pe