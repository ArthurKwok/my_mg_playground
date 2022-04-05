import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import math

from .attention import MultiHeadAttention

class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_seq, n_dim, n_head):
        super().__init__()
        self.mha = MultiHeadAttention(n_seq, n_dim, n_head)
        self.linear = nn.Linear(n_dim, n_dim)
        self.ln1 = nn.LayerNorm(n_dim)
        self.ln2 = nn.LayerNorm(n_dim)
    
    def forward(self, x):
        x = self.ln1(x + self.mha(x, x, x))
        x = self.ln2(x + self.linear(x))
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, n_seq, n_dim, n_head):
        super().__init__()
        self.mha1 = MultiHeadAttention(n_seq, n_dim, n_head)
        self.mha2 = MultiHeadAttention(n_seq, n_dim, n_head)
        self.linear = nn.Linear(n_dim, n_dim)
        self.ln1 = nn.LayerNorm(n_dim)
        self.ln2 = nn.LayerNorm(n_dim)
        self.ln3 = nn.LayerNorm(n_dim)
    
    def forward(self, x, z):
        x = self.ln1(x + self.mha1(x, x, x))
        x = self.ln2(x + self.mha2(x, z, z))
        x = self.ln3(x + self.linear(x))
        return x
