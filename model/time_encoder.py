import torch
import torch.nn as nn
import numpy as np

class FourierTimeEncoder(nn.Module):
    def __init__(self, out_dim, scale=10.0):
        super().__init__()
        # Random frequencies for capturing temporal variations at different scales
        self.register_buffer("freqs", torch.randn(out_dim // 2) * scale)

    def forward(self, x):
        """
        x: (B, 243, 1) - Continuous time Delta-T
        returns: (B, 243, out_dim)
        """
        # Mathematical core: map scalar to sine-cosine pairs
        x_proj = 2 * np.pi * x @ self.freqs.view(1, 1, -1)
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)