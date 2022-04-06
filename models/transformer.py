import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import math

from .attention import MultiHeadAttention

class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_seq, n_dim, n_head, d_ff):
        super().__init__()
        self.mha = MultiHeadAttention(n_seq, n_dim, n_head)
        self.ff = nn.Sequential(nn.Linear(n_dim, d_ff),
                                nn.ReLU(),
                                nn.Linear(d_ff, n_dim))
        self.ln1 = nn.LayerNorm(n_dim)
        self.ln2 = nn.LayerNorm(n_dim)
    
    def forward(self, x):
        x = self.ln1(x + self.mha(x, x, x))
        x = self.ln2(x + self.ff(x))
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, n_seq, n_dim, n_head, d_ff):
        super().__init__()
        self.mha1 = MultiHeadAttention(n_seq, n_dim, n_head)
        self.mha2 = MultiHeadAttention(n_seq, n_dim, n_head)
        self.ff = nn.Sequential(nn.Linear(n_dim, d_ff),
                                nn.ReLU(),
                                nn.Linear(d_ff, n_dim))
        self.ln1 = nn.LayerNorm(n_dim)
        self.ln2 = nn.LayerNorm(n_dim)
        self.ln3 = nn.LayerNorm(n_dim)
    
    def forward(self, x, z):
        x = self.ln1(x + self.mha1(x, x, x))
        x = self.ln2(x + self.mha2(x, z, z)) # Q from decoder, K-V pair from encoder
        x = self.ln3(x + self.ff(x))
        return x

    
class PositionalEncoding(nn.Module):
    def __init__(self, n_seq, n_dim):
        super().__init__()
        self.pe = torch.zeros(n_seq, n_dim)
        pos = torch.arange(0, n_seq)
        # i = torch.arange(0, n_dim).unsqueeze(0) # row vector

        for i in range(n_dim):
            if i % 2 == 0:
                self.pe[:, i] = torch.sin(pos / (10000**(2*i/n_dim)))
            else:
                self.pe[:, i] = torch.cos(pos / (10000**(2*i/n_dim)))

    def forward(self, x):
        # x: (..., n_seq, n_dim)
        return self.pe + x
