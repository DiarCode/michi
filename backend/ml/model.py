"""DTS-GSSF model architecture — GraphSSM backbone with LoRA adaptation.

Canonical implementation matching the paper specification:
  - 3-layer MLP prediction head (d_model -> 2*d_model -> d_model -> horizon)
  - Learnable graph adjacency mixing (alpha initialised at 0.6)
  - Multi-head self-attention via nn.MultiheadAttention
  - Concatenation fusion [h_graph; h_temp] -> projection
  - Negative Binomial likelihood for count data

Hyperparameter defaults align with Table 8 of the paper:
  d_model=192, horizon=4, K=3, lora_r=16, n_heads=6, dropout=0.1
"""

import math
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x, beta=1.0, threshold=20.0)


class LoRALinear(nn.Module):
    """Low-Rank Adaptation of a linear layer (Hu et al., 2022).

    Computes  y = W_base x + (alpha/r) B A x  where
      A ∈ R^{r × in_features},  B ∈ R^{out_features × r}.
    Setting r=0 falls back to a plain linear layer.
    """

    def __init__(self, in_features: int, out_features: int, r: int = 16, alpha: float = 16.0, bias: bool = True):
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
    """Gated Recurrent Encoder: minimal gated unit with LoRA-adapted input projection.

    Computes per-timestep update:
        s_t = a_t ⊙ s_{t-1} + (1 - a_t) ⊙ b_t
    where a_t = σ(gate_a(u_t)), b_t = tanh(gate_b(u_t)).
    """

    def __init__(self, d_in: int, d_model: int, dropout: float = 0.1, lora_r: int = 16):
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
    """Dual-adjacency graph propagation with learnable mixing.

    Computes  A = σ(log_α) · A_phys + (1 − σ(log_α)) · A_adp
    where A_adp = softmax(ReLU(E₁ E₂ᵀ))  and  σ(log_α) is initialised at alpha_phys.
    K propagation steps:  h^{(k)} = GELU(W_g · einsum(A, h^{(k-1)}))
    with residual connection and LayerNorm.
    """

    def __init__(
        self,
        N: int,
        d: int,
        A_phys: np.ndarray,
        K: int = 3,
        alpha_phys: float = 0.6,
        d_emb: int = 16,
        learnable_alpha: bool = True,
    ):
        super().__init__()
        self.K = K
        self.learnable_alpha = learnable_alpha
        if learnable_alpha:
            self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha_phys), dtype=torch.float32))
        else:
            self.register_buffer("log_alpha", torch.tensor(math.log(alpha_phys), dtype=torch.float32))
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
        alpha = torch.sigmoid(self.log_alpha)
        A = alpha * self.A_phys + (1.0 - alpha) * A_adp
        out = h
        for _ in range(self.K):
            out = torch.einsum("ij,bjd->bid", A, out)
            out = F.gelu(self.Wg(out))
        return self.norm(out + h)


class TemporalAttention(nn.Module):
    """Multi-head self-attention over the time dimension of per-station sequences.

    Applied independently per station.  The per-timestep GRE outputs U ∈ R^{T×d}
    are processed by standard multi-head attention with residual connection and
    LayerNorm.  After attention, mean pooling over the time dimension produces a
    fixed-size station representation h_temp ∈ R^d.
    """

    def __init__(self, d_model: int, n_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*N, L, d_model)
        attn_out, _ = self.attn(x, x, x)
        return self.norm(attn_out + x)


class DTSGSSF(nn.Module):
    """Dual-Timescale Graph State-Space Forecasting model.

    Paper defaults (Table 8):
      d_model=192, horizon=4, K=3, lora_r=16, n_heads=6, dropout=0.1
    """

    def __init__(
        self,
        N: int,
        F_in: int,
        n_series: int,
        n_agg: int,
        A_phys: np.ndarray,
        d_model: int = 192,
        horizon: int = 4,
        K: int = 3,
        lora_r: int = 16,
        dropout: float = 0.1,
        n_heads: int = 6,
        alpha_phys: float = 0.6,
    ):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model

        # Encoder: Gated SSM + Graph propagation + Temporal attention
        self.ssm = GatedSSMBlock(F_in, d_model, dropout=dropout, lora_r=lora_r)
        self.graph = GraphPropagation(
            N, d_model, A_phys=A_phys, K=K, alpha_phys=alpha_phys, d_emb=16, learnable_alpha=True
        )
        self.attn = TemporalAttention(d_model, n_heads=n_heads, dropout=dropout)

        # Fusion: concatenation [h_graph; h_temp] -> projection -> d_model
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        # Prediction heads
        self.head_bottom = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )
        self.pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.head_agg = LoRALinear(d_model, horizon * n_agg, r=lora_r, alpha=16.0, bias=True)

        # Negative Binomial dispersion (global)
        self.log_kappa = nn.Parameter(torch.tensor(math.log(8.0), dtype=torch.float32))
        self.N = N
        self.n_series = n_series
        self.n_agg = n_agg

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, N, _ = x.shape
        # SSM: process each station's temporal sequence
        h_ssm = self.ssm(x)  # (B, N, d_model)
        # Graph propagation: spatial diffusion
        h_graph = self.graph(h_ssm)  # (B, N, d_model)
        # Temporal attention over per-timestep SSM projections
        u = self.ssm.drop(F.gelu(self.ssm.in_proj(x)))  # (B, L, N, d_model)
        u = u.permute(0, 2, 1, 3).reshape(B * N, L, self.d_model)
        h_temp = self.attn(u)  # (B*N, L, d_model)
        h_temp = h_temp.reshape(B, N, L, self.d_model).mean(dim=2)  # (B, N, d_model)
        # Concatenation fusion
        h = self.fusion_proj(torch.cat([h_graph, h_temp], dim=-1))  # (B, N, d_model)
        # Prediction heads
        eta_bottom = self.head_bottom(h)  # (B, N, horizon)
        mu_bottom = torch.exp(eta_bottom).permute(0, 2, 1)  # (B, horizon, N)
        pooled = self.pool(h).mean(dim=1)  # (B, d_model)
        eta_agg = self.head_agg(pooled).view(B, self.horizon, self.n_agg)
        mu_agg = torch.exp(eta_agg)  # (B, horizon, n_agg)
        mu_all = torch.cat([mu_bottom, mu_agg], dim=-1)  # (B, horizon, n_series)
        kappa = softplus(self.log_kappa) + 1e-4
        return mu_all, kappa

    def freeze_base_for_adaptation(self) -> None:
        """Freeze all parameters except LoRA matrices and dispersion."""
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
    """Negative log-likelihood of the Negative Binomial distribution.

    Parameters
    ----------
    y : Tensor, non-negative integer counts
    mu : Tensor, predicted mean (> 0)
    kappa : Tensor, dispersion parameter (> 0)
    eps : float, numerical guard

    Returns
    -------
    Tensor, scalar mean NLL
    """
    y = torch.clamp(y, min=0.0)
    mu = torch.clamp(mu, min=eps)
    k = torch.clamp(kappa, min=eps)
    k_plus_mu = torch.clamp(k + mu, min=eps)
    loglik = (
        torch.lgamma(y + k)
        - torch.lgamma(k)
        - torch.lgamma(y + 1.0)
        + k * (torch.log(k) - torch.log(k_plus_mu))
        + y * (torch.log(mu) - torch.log(k_plus_mu))
    )
    return -loglik
