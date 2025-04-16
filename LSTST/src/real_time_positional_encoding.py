import numpy as np
import torch


class RealTimePositionalEncoding:
    def __init__(self, d_model):
        self.d_model = d_model

    def compute(self, timestamps, L, kernel_size, stride, padding):
        # timestamps: (B, L)
        B = timestamps.shape[0]
        num_patches = (L + 2 * padding - kernel_size) // stride + 1
        patch_times = []

        for b in range(B):
            sample_t = []
            for j in range(num_patches):
                b_idx = j * stride - padding
                l_idx = b_idx + kernel_size
                idx_range = range(max(b_idx, 0), min(l_idx, L))
                t_patch = timestamps[b, list(idx_range)]
                median_t = torch.median(t_patch).item()
                sample_t.append(median_t)
            patch_times.append(sample_t)

        patch_times = torch.tensor(patch_times, dtype=torch.float32)  # (B, num_patches)
        return self._sinusoidal_encoding(patch_times)

    def _sinusoidal_encoding(self, times):
        B, L = times.shape
        pe = torch.zeros((B, L, self.d_model))
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model))
        for i in range(B):
            for pos in range(L):
                t = times[i, pos]
                pe[i, pos, 0::2] = torch.sin(t * div_term)
                pe[i, pos, 1::2] = torch.cos(t * div_term)
        return pe
