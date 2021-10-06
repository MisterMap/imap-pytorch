import torch
import torch.nn as nn


class GaussianPositionalEncoding(nn.Module):
    def __init__(self, sigma=25, encoding_dimension=93, use_sin=False):
        super().__init__()
        self._use_sin = use_sin
        if use_sin:
            encoding_dimension = encoding_dimension // 2
        self._b_encoding_matrix = nn.Linear(3, encoding_dimension, bias=False)
        self._b_encoding_matrix.weight = nn.Parameter(torch.randn((encoding_dimension, 3)) * sigma)

    def forward(self, x):
        encodings = self._b_encoding_matrix(x)
        if not self._use_sin:
            return torch.cos(encodings)
        cos_encodings = torch.cos(encodings)
        sin_encodings = torch.sin(encodings)
        return torch.cat([cos_encodings, sin_encodings], dim=1)
