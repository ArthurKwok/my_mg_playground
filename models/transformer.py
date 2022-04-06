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
        for i in range(n_dim):
            if i % 2 == 0:
                self.pe[:, i] = torch.sin(pos / (10000**(2*i/n_dim)))
            else:
                self.pe[:, i] = torch.cos(pos / (10000**(2*i/n_dim)))
        self.pe = nn.Parameter(self.pe)

    def forward(self, x):
        # x: (..., n_seq, n_dim)
        return self.pe + x


class TransformerEncoder(nn.Module):
    def __init__(self, n_seq, n_dim, n_head, n_layer, d_ff=2048):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(n_seq, n_dim, n_head, d_ff)
                       for i in range(n_layer)])

    def forward(self, x):
        # x: (n_batch, n_seq, n_dim)
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, n_seq, n_dim, n_head, n_layer, d_ff=2048):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(n_seq, n_dim, n_head, d_ff)
                       for i in range(n_layer)])

    def forward(self, x, z):
        # x: (n_batch, n_seq, n_dim)
        # z: (n_batch, n_seq, n_dim)
        for layer in self.layers:
            x = layer(x, z)
        return x


class Transformer(nn.Module):
    def __init__(self, n_seq, n_dim, n_head, n_layer, d_ff=2048):
        super().__init__()
        self.pe = PositionalEncoding(n_seq, n_dim)
        self.encoder = TransformerEncoder(n_seq, n_dim, n_head, n_layer, d_ff)
        self.decoder = TransformerDecoder(n_seq, n_dim, n_head, n_layer, d_ff)
        self.ff = nn.Sequential(
            nn.Linear(n_dim, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, n_dim)
        )

    def forward(self, x, x_dec):
        # x, x_dec: (n_batch, n_seq, n_dim)
        # x_dec is the shifted version of x
        x = self.pe(x)
        h = self.encoder(x)
        output = self.decoder(self.pe(x_dec), h)

        return output


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_batch = 512
    n_seq = 64
    n_dim = 128
    n_head = 8
    n_layer = 2
    input = torch.randn(n_batch, n_seq, n_dim).to(device)
    input_shifted = torch.roll(input, -1, -2)
    input_shifted[:, -1, :] = torch.zeros_like(input_shifted[:, -1, :])

    tfm = Transformer(n_seq, n_dim, n_head, n_layer).to(device)
    output = tfm(input, input_shifted)
    print(output.shape)
