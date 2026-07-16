"""
Lift3D-VLA: Temporal Action Heads Ablation (Table IV) — Illustrative Re-implementation

This script implements a *pedagogical* PyTorch module that mirrors the spirit of
Table IV in the paper: given a stack of visual + proprioceptive tokens, a
"temporal action head" predicts action chunks at multiple time scales, and the
number of layers in the head controls which scales are reachable.

It is NOT the official code. It is meant to make the design choice
("1 / 2 / 4 layers ↔ which time scales get decoded") tangible for a reader.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration: which temporal scales exist and how many layers "reach" each.
# ---------------------------------------------------------------------------
@dataclass
class TemporalHeadConfig:
    # Time scales, in action-step units, ordered coarse → fine.
    # E.g. predict chunks at {8, 4, 2, 1}-step granularity.
    scales: List[int]      # e.g. [8, 4, 2, 1]
    # How many transformer layers are stacked in the action head.
    n_layers: int          # 1, 2, or 4 (the three columns of Table IV)
    d_model: int = 256
    n_heads: int = 8
    action_dim: int = 7    # e.g. 6-DoF + gripper
    chunk_len: int = 8     # the longest horizon we ever emit

    def reachable_scales(self) -> List[int]:
        """
        Layer budget -> which scales are decoded.
        Rule of thumb (matches the paper's ablation pattern):
          1 layer  -> only the coarsest scale
          2 layers -> coarsest + middle
          4 layers -> all scales
        """
        s = sorted(self.scales, reverse=True)  # coarse first
        if self.n_layers <= 1:
            return [s[0]]
        if self.n_layers == 2:
            return [s[0], s[len(s) // 2]]
        return s  # 4+ layers => full multi-scale


# ---------------------------------------------------------------------------
# A tiny transformer block, kept small for clarity (not the paper's backbone).
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + h)
        x = self.norm2(x + self.mlp(x))
        return x


# ---------------------------------------------------------------------------
# The multi-scale temporal action head.
# ---------------------------------------------------------------------------
class TemporalActionHead(nn.Module):
    def __init__(self, cfg: TemporalHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_layers)
        )
        # One action-projection head per reachable scale.
        # The key insight from Table IV: more layers -> more scales are "decodable".
        self.scale_heads = nn.ModuleDict(
            {
                f"scale_{s}": nn.Linear(cfg.d_model, cfg.action_dim)
                for s in cfg.reachable_scales()
            }
        )
        # A learned query token per scale (one per reachable scale).
        self.scale_queries = nn.ParameterDict(
            {
                f"scale_{s}": nn.Parameter(torch.randn(1, 1, cfg.d_model) / math.sqrt(cfg.d_model))
                for s in cfg.reachable_scales()
            }
        )

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        tokens: (B, N, D) — fused visual + proprio + 3D-geometry tokens.
        Returns a dict {scale_name -> (B, T, action_dim)} for every reachable scale.
        """
        x = self.in_proj(tokens)
        for blk in self.blocks:
            x = blk(x)

        # Pool the token sequence to a single context vector (mean-pool, simple).
        ctx = x.mean(dim=1, keepdim=True)  # (B, 1, D)

        out: dict[str, torch.Tensor] = {}
        for name, head in self.scale_heads.items():
            scale = int(name.split("_")[1])
            # Repeat the context to match this scale's horizon.
            horizon = max(1, self.cfg.chunk_len // scale)
            q = self.scale_queries[name].expand(ctx.size(0), horizon, -1)
            # Cross-attend queries to the pooled context (simplified: add + MLP).
            act = head(ctx.expand_as(q) + q)  # (B, horizon, action_dim)
            out[name] = act
        return out


# ---------------------------------------------------------------------------
# Demo / smoke test — runs without any external assets.
# ---------------------------------------------------------------------------
def ablation_table_iv_demo() -> None:
    torch.manual_seed(0)
    B, N, D = 2, 64, 256
    tokens = torch.randn(B, N, D)

    for n_layers in (1, 2, 4):
        cfg = TemporalHeadConfig(scales=[8, 4, 2, 1], n_layers=n_layers)
        head = TemporalActionHead(cfg)
        out = head(tokens)
        print(f"--- n_layers={n_layers} -> reachable_scales={cfg.reachable_scales()} ---")
        for k, v in out.items():
            print(f"  {k}: shape={tuple(v.shape)}")
        # Sanity: every action is finite.
        assert all(torch.isfinite(v).all() for v in out.values())


if __name__ == "__main__":
    ablation_table_iv_demo()