import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, n_seq, n_dim, n_head):
        super().__init__()

        self.n_dim = n_dim
        self.n_head = n_head
        self.n_seq = n_seq
        self.d_k = int(self.n_dim/self.n_head)

        self.w_q = nn.Linear(n_dim, n_dim)
        self.w_k = nn.Linear(n_dim, n_dim)
        self.w_v = nn.Linear(n_dim, n_dim)
        self.w_o = nn.Linear(n_dim, n_dim)
    
    def forward(self, q, k, v):
        # q, k, v: (n_batch, n_seq, n_dim)
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        Q = Q.view(-1, self.n_seq, self.d_k, self.n_head)
        K = K.view(-1, self.n_seq, self.d_k, self.n_head)
        K = K.transpose(-2, -1)
        V = V.view(-1, self.n_seq, self.d_k, self.n_head)

        score = torch.matmul(Q, K) / math.sqrt(self.d_k)
        score = F.softmax(score, dim=-1)
        # score: (n_batch, n_seq, d_k, d_k)
        attention = torch.matmul(score, V)
        # attention: (n_batch, n_seq, d_k, n_head)
        attention = attention.view(-1, n_seq, n_dim)

        return self.w_o(attention)


if __name__ == "__main__":
    n_batch = 256
    n_seq = 128
    n_dim = 512
    n_head = 8
    input = torch.randn(n_batch, n_seq, n_dim)
    mha = MultiHeadAttention(n_batch, n_seq, n_dim, n_head)
    output = mha(input, input, input)

    print(output.shape)
