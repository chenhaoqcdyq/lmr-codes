"""
Quantization module for VQ-VAE (inference only)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizeEMAReset(nn.Module):
    """Quantizer with EMA and reset mechanism"""
    def __init__(self, nb_code, code_dim, args):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = args.mu
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim))

    def quantize(self, x):
        """Quantize input to codebook indices"""
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0, keepdim=True)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        """Dequantize codebook indices to vectors"""
        x = F.embedding(code_idx, self.codebook)
        return x

    def preprocess(self, x):
        """NCT -> NTC -> [NT, C]"""
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def forward(self, x):
        """Forward pass for quantization"""
        N, width, T = x.shape
        x = self.preprocess(x)
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Passthrough gradients
        x_d = x + (x_d - x).detach()
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()

        # Dummy loss and perplexity for inference
        commit_loss = torch.tensor(0.0, device=x_d.device)
        perplexity = torch.tensor(0.0, device=x_d.device)

        return x_d, commit_loss, perplexity
