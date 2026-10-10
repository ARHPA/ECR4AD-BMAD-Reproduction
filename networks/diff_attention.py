#adopted from https://github.com/aykutcayir34/DifferentialTransformer/blob/main/main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len: int, *, device: torch.device):
        seq = torch.arange(max_seq_len, device=device)
        freqs = torch.einsum("i,j->ij", seq, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))

class DifferentialAttention(nn.Module):
    def __init__(self, dim, num_heads = 8, qkv_bias = False, qk_scale = None, attn_drop=0, proj_drop=0, depth = 0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 6, bias=qkv_bias)

        self.depth = depth
        self.lambda_q1 = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.rotary_emb = RotaryEmbedding(head_dim * 2)
        self.proj = nn.Linear(2 * dim, dim, bias=False)

    def forward(self, x):
        lambda_init = lambda_init_fn(self.depth)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 6, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = torch.concat((qkv[0], qkv[1]), dim = -1)
        k = torch.concat((qkv[2], qkv[3]), dim = -1)
        v = torch.concat((qkv[4], qkv[5]), dim = -1)
        seq_len = x.shape[1]
        cos, sin = self.rotary_emb(seq_len, device=x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)

        attn1 = q1 @ k1.transpose(-2, -1) * self.scale
        attn2 = q2 @ k2.transpose(-2, -1) * self.scale

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q2)
        lambda_ = lambda_1 - lambda_2 + lambda_init

        x = (F.softmax(attn1, dim=-1)  - lambda_ * F.softmax(attn2, dim=-1)) @ v

        #x = (attn1 / (torch.sum(attn1, dim=-1, keepdim=True)) - lambda_ * attn2 / (torch.sum(attn2, dim=-1, keepdim=True))) @ v

        x = x.transpose(1, 2).reshape(B, N, C * 2)
        x = self.proj(x)

        return x, None
