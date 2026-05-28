"""DTS-GSSF model architecture — GraphSSM backbone with LoRA adaptation.

Extracted from main.py for integration with FastAPI backend.
"""
import math
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x, beta=1.0, threshold=20.0)


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: float = 16.0, bias: bool = True):
        super().__init__()
        self.r = r
        self.scale = alpha / max(1, r)
        self.base = nn.Linear(in_features, out_features, bias=bias)
        if r > 0:
            self.A = nn.Parameter(torch.zeros(r, in_features))
            self.B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r > 0:
            y = y + self.scale * ((x @ self.A.T) @ self.B.T)
        return y

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        if self.r <= 0:
            return []
        return [self.A, self.B]


class GatedSSMBlock(nn.Module):
    def __init__(self, d_in: int, d_model: int, dropout: float = 0.1, lora_r: int = 8):
        super().__init__()
        self.d_model = d_model
        self.in_proj = LoRALinear(d_in, d_model, r=lora_r, alpha=16.0)
        self.gate_a = nn.Linear(d_model, d_model)
        self.gate_b = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        nn.init.constant_(self.gate_a.bias, -1.0)
        nn.init.zeros_(self.gate_b.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, N, _ = x.shape
        u = self.drop(F.gelu(self.in_proj(x)))
        u2 = u.reshape(B * N, L, self.d_model)
        s = torch.zeros((B * N, self.d_model), device=u.device, dtype=u.dtype)
        for t in range(L):
            ut = u2[:, t, :]
            a = torch.sigmoid(self.gate_a(ut))
            b = torch.tanh(self.gate_b(ut))
            s = a * s + (1.0 - a) * b
        s = self.norm(s).reshape(B, N, self.d_model)
        return s


class GraphPropagation(nn.Module):
    def __init__(self, N: int, d: int, A_phys: np.ndarray, K: int = 2, alpha_phys: float = 0.6, d_emb: int = 16):
        super().__init__()
        self.K = K
        self.alpha_phys = alpha_phys
        self.register_buffer("A_phys", torch.from_numpy(A_phys).float())
        self.E1 = nn.Parameter(torch.randn(N, d_emb) * 0.05)
        self.E2 = nn.Parameter(torch.randn(N, d_emb) * 0.05)
        self.Wg = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    def adaptive_adj(self) -> torch.Tensor:
        logits = F.relu(self.E1 @ self.E2.T)
        return F.softmax(logits, dim=-1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        A_adp = self.adaptive_adj()
        A = self.alpha_phys * self.A_phys + (1.0 - self.alpha_phys) * A_adp
        out = h
        for _ in range(self.K):
            out = torch.einsum("ij,bjd->bid", A, out)
            out = F.gelu(self.Wg(out))
        return self.norm(out + h)


class TemporalAttention(nn.Module):
    """Multi-head self-attention over the SSM's per-timestep hidden states."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.norm(x + self.drop(self.out(out)))


class DTSGSSF(nn.Module):
    def __init__(self, N: int, F_in: int, n_series: int, n_agg: int, A_phys: np.ndarray,
                 d_model: int = 64, horizon: int = 12, K: int = 2, lora_r: int = 8, dropout: float = 0.1,
                 n_heads: int = 4):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model
        self.ssm = GatedSSMBlock(F_in, d_model, dropout=dropout, lora_r=lora_r)
        self.graph = GraphPropagation(N, d_model, A_phys=A_phys, K=K, alpha_phys=0.6, d_emb=16)
        self.attn = TemporalAttention(d_model, n_heads=n_heads, dropout=dropout)
        self.head_bottom = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )
        self.pool = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        self.head_agg = LoRALinear(d_model, horizon * n_agg, r=lora_r, alpha=16.0, bias=True)
        self.log_kappa = nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.N = N
        self.n_series = n_series
        self.n_agg = n_agg

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, N, _ = x.shape
        # SSM: process each station's temporal sequence
        h_ssm = self.ssm(x)  # (B, N, d_model)
        # Graph propagation: spatial diffusion
        h_graph = self.graph(h_ssm)  # (B, N, d_model)
        # Temporal attention over per-timestep SSM projections
        u = self.ssm.drop(F.gelu(self.ssm.in_proj(x)))  # (B, L, N, d_model)
        u = u.permute(0, 2, 1, 3).reshape(B * N, L, self.d_model)
        h_temp = self.attn(u).reshape(B, N, L, self.d_model).mean(dim=2)  # (B, N, d_model)
        # Combine graph + temporal
        h = h_graph + h_temp
        # Prediction heads
        eta_bottom = self.head_bottom(h)
        mu_bottom = torch.exp(eta_bottom).permute(0, 2, 1)
        pooled = self.pool(h).mean(dim=1)
        eta_agg = self.head_agg(pooled).view(B, self.horizon, self.n_agg)
        mu_agg = torch.exp(eta_agg)
        mu_all = torch.cat([mu_bottom, mu_agg], dim=-1)
        kappa = softplus(self.log_kappa) + 1e-4
        return mu_all, kappa

    def freeze_base_for_adaptation(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        def unfreeze_lora(m: nn.Module) -> None:
            if isinstance(m, LoRALinear):
                for p in m.lora_parameters():
                    p.requires_grad = True
        self.apply(unfreeze_lora)
        self.log_kappa.requires_grad = True

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


def nb_nll(y: torch.Tensor, mu: torch.Tensor, kappa: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y = torch.clamp(y, min=0.0)
    mu = torch.clamp(mu, min=eps)
    k = torch.clamp(kappa, min=eps)
    loglik = (torch.lgamma(y + k) - torch.lgamma(k) - torch.lgamma(y + 1.0)
              + k * (torch.log(k) - torch.log(k + mu))
              + y * (torch.log(mu) - torch.log(k + mu)))
    return -loglik
