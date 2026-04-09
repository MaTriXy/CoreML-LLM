"""ANE-optimized ops that stay in NCHW format throughout.

Key insight: hidden_states = (1, C, 1, 1) NCHW format, never transposed to NHC.
This eliminates ~144 transposes per chunk that were forcing GPU selection.

Based on Apple ml-ane-transformers reference + MLComputePlan cost analysis.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_DTYPE = torch.float16


class ANERMSNormNCHW(nn.Module):
    """RMSNorm for NCHW tensors (1, C, 1, 1).

    Uses manual reduce_mean/rsqrt/mul instead of cat-trick because these
    ops all support ANE natively and produce fewer layout changes than
    cat+layernorm on an NCHW tensor.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=MODEL_DTYPE))
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, C, 1, 1) fp16. Force fp16 through all intermediate ops
        # (torch.mean and torch.rsqrt can auto-upcast to fp32).
        sq = (x * x).to(MODEL_DTYPE)
        mean_sq = sq.mean(dim=1, keepdim=True).to(MODEL_DTYPE)
        inv = torch.rsqrt(mean_sq + self.eps).to(MODEL_DTYPE)
        x_normed = (x * inv).to(MODEL_DTYPE)
        w = self.weight.view(1, -1, 1, 1)
        return (x_normed * w).to(MODEL_DTYPE)


class ANERMSNormPerHead(nn.Module):
    """RMSNorm on per-head tensor (1, num_heads, head_dim, 1)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=MODEL_DTYPE))
        self.eps = eps
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sq = (x * x).to(MODEL_DTYPE)
        mean_sq = sq.mean(dim=2, keepdim=True).to(MODEL_DTYPE)
        inv = torch.rsqrt(mean_sq + self.eps).to(MODEL_DTYPE)
        x_normed = (x * inv).to(MODEL_DTYPE)
        w = self.weight.view(1, 1, -1, 1)
        return (x_normed * w).to(MODEL_DTYPE)


def rotate_half_nchw(x: torch.Tensor) -> torch.Tensor:
    """Rotate half: [x1, x2] → [-x2, x1] on a NCHW-style (1, H, D, 1) tensor.
    D dim is head_dim which is split in half.
    """
    # x: (1, H, D, 1)
    x1, x2 = torch.chunk(x, 2, dim=2)
    return torch.cat((-x2, x1), dim=2)


def apply_rope_nchw(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding on NCHW tensor.

    x: (1, H, D, 1)
    cos, sin: (1, 1, D, 1)
    """
    return x * cos + rotate_half_nchw(x) * sin


def ane_softmax_nchw(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Manual softmax preserving fp16."""
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = (x - x_max).to(MODEL_DTYPE)
    exp_x = torch.exp(x_shifted).to(MODEL_DTYPE)
    return (exp_x / exp_x.sum(dim=dim, keepdim=True)).to(MODEL_DTYPE)
