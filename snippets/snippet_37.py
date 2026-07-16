"""
Lift3D-VLA — architectural intuition, in ~120 lines.

Goal of this snippet (didactic, not the official code):
    Show *why* adding temporal dynamics on top of 3D lifting gives a
    second accuracy jump in the paper's ablation table
    (3D-only ≈ 70.0 %  →  3D + temporal dynamics ≈ 82.8 %).

We implement two minimal branches and a tiny ablation harness:
    (A) Geometry-only      : per-frame point cloud tokens  → action
    (B) Geometry + dynamics: per-frame point cloud tokens
                             + inter-frame velocity tokens → action

The "score" we print is *illustrative*: the variance of the predicted
action distribution under a fixed synthetic input stream. A higher
variance for the dynamics branch means the policy's action is
conditioned on richer context, which is exactly the mechanism behind
the +12.8 pp jump reported in the paper. We are NOT reproducing the
paper's metric — we are illustrating the architectural asymmetry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Toy inputs: a short video of an object being pushed across a plane.
# ---------------------------------------------------------------------------
@dataclass
class ClipConfig:
    T: int = 4          # number of frames (temporal dynamics needs T >= 2)
    H: int = 16         # spatial height of the feature map
    W: int = 16         # spatial width  of the feature map
    D_feat: int = 32    # per-pixel feature dim produced by the 2D encoder
    P: int = 64         # number of points sampled per frame


def make_synthetic_clip(cfg: ClipConfig, seed: int = 0) -> dict:
    """Synthetic multi-frame clip: features + a learned camera + depth."""
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(cfg.T, cfg.H, cfg.W, cfg.D_feat, generator=g)
    # Simulated depth (positive) that drifts slightly across frames so the
    # dynamics module has something real to model.
    depth = 1.0 + 0.2 * torch.arange(cfg.T).float().view(-1, 1, 1, 1)
    depth = depth.expand(cfg.T, cfg.H, cfg.W)
    # Camera intrinsics-ish (focal + principal point) as identity-like.
    K = torch.tensor([[[[cfg.W / 2, 0.0, cfg.W / 2],
                        [0.0, cfg.H / 2, cfg.H / 2],
                        [0.0, 0.0, 1.0]]]])
    return {"feats": feats, "depth": depth, "K": K}


# ---------------------------------------------------------------------------
# 2. Encoders.
# ---------------------------------------------------------------------------
class FrameEncoder(nn.Module):
    """2D image → per-pixel feature (cheap conv stack)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (T, H, W, D_feat)  →  (T, D_feat, H, W) for Conv2d
        x = feats.permute(0, 3, 1, 2)
        return self.net(x)  # (T, D, H, W)


class LiftTo3D(nn.Module):
    """Per-frame lift: (feature map, depth, K) → (P, 3+D) point tokens.

    This is the *first* claim of the paper: reconstructing 3D point clouds
    directly is already enough to outperform a 2D-only baseline.
    """

    def __init__(self, feat_dim: int, K_sample: int):
        super().__init__()
        self.K_sample = K_sample
        # Tiny "score head" that picks which pixels become point tokens.
        self.score = nn.Conv2d(feat_dim, 1, 1)

    @staticmethod
    def backproject(uv: torch.Tensor, depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """uv: (P, 2), depth: (P,), K: (1, 1, 3, 3)  →  xyz: (P, 3)."""
        fx, fy = K[0, 0, 0, 0], K[0, 0, 1, 1]
        cx, cy = K[0, 0, 0, 2], K[0, 0, 1, 2]
        x = (uv[:, 0] - cx) * depth / fx
        y = (uv[:, 1] - cy) * depth / fy
        z = depth
        return torch.stack([x, y, z], dim=-1)

    def forward(self, feat_map: torch.Tensor, depth: torch.Tensor,
                K: torch.Tensor) -> torch.Tensor:
        T, D, H, W = feat_map.shape
        # (a) pixel grid (T, H*W, 2)
        ys, xs = torch.meshgrid(
            torch.arange(H, device=feat_map.device),
            torch.arange(W, device=feat_map.device), indexing="ij")
        uv = torch.stack([xs, ys], dim=-1).float().view(T, -1, 2)
        d_flat = depth.view(T, -1)
        f_flat = feat_map.permute(0, 2, 3, 1).reshape(T, -1, D)

        out_tokens = []
        for t in range(T):
            scores = self.score(feat_map[t:t + 1]).view(-1)
            idx = torch.topk(scores, self.K_sample).indices
            xyz = self.backproject(uv[t][idx], d_flat[t][idx], K)
            f = f_flat[t][idx]
            out_tokens.append(torch.cat([xyz, f], dim=-1))  # (P, 3+D)
        return torch.stack(out_tokens, dim=0)  # (T, P, 3+D)


class TemporalDynamics(nn.Module):
    """Per-point displacement / velocity tokens across consecutive frames.

    This is the *second* claim of the paper: modelling the geometric
    *evolution* of the scene gives an additional accuracy jump because
    contact, slip and object articulation are all encoded in motion.
    """

    def __init__(self, point_dim: int, dyn_dim: int):
        super().__init__()
        # Operates on the *xyz* portion to keep geometry-grounded motion.
        self.mlp = nn.Sequential(
            nn.Linear(point_dim * 2, dyn_dim),  # concat(prev, curr)
            nn.GELU(),
            nn.Linear(dyn_dim, dyn_dim),
        )

    def forward(self, pc_tokens: torch.Tensor) -> torch.Tensor:
        # pc_tokens: (T, P, 3+D). We only use xyz = pc_tokens[..., :3].
        xyz = pc_tokens[..., :3]
        # First-order motion: delta_t = xyz[t] - xyz[t-1].
        delta = xyz[1:] - xyz[:-1]                 # (T-1, P, 3)
        prev = xyz[:-1]                            # (T-1, P, 3)
        motion = self.mlp(torch.cat([prev, delta], dim=-1))  # (T-1, P, dyn_dim)
        # Pad a zero token at t=0 so downstream shapes align with pc_tokens.
        pad = torch.zeros(1, *motion.shape[1:], device=motion.device,
                          dtype=motion.dtype)
        return torch.cat([pad, motion], dim=0)     # (T, P, dyn_dim)


# ---------------------------------------------------------------------------
# 3. The policy head. It is shared between the two ablation variants so the
#    only thing that changes is *what* tokens it sees.
# ---------------------------------------------------------------------------
class VLAPolicy(nn.Module):
    def __init__(self, token_dim: int, action_dim: int = 7):
        super().__init__()
        self.attn = nn.MultiheadAttention(token_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)
        self.head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, action_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (T*P, token_dim) → mean-pool → action.
        if tokens.dim() == 2:
            x = tokens.unsqueeze(0)               # (1, N, D)
        else:
            B = tokens.shape[0]
            x = tokens.reshape(B, -1, tokens.shape[-1])
        a, _ = self.attn(x, x, x)
        a = self.norm(a).mean(dim=1)
        return self.head(a)


# ---------------------------------------------------------------------------
# 4. Ablation harness.
# ---------------------------------------------------------------------------
def run_ablation(seed: int = 0) -> dict:
    cfg = ClipConfig()
    clip = make_synthetic_clip(cfg, seed)

    enc = FrameEncoder(cfg.D_feat, cfg.D_feat)
    lift = LiftTo3D(cfg.D_feat, cfg.P)
    dyn = TemporalDynamics(point_dim=3, dyn_dim=cfg.D_feat)
    policy_geo = VLAPolicy(token_dim=3 + cfg.D_feat)        # geometry-only
    policy_full = VLAPolicy(token_dim=3 + cfg.D_feat * 2)   # + dynamics

    pc_tokens = lift(enc(clip["feats"]), clip["depth"], clip["K"])  # (T,P,3+D)
    motion_tokens = dyn(pc_tokens)                                  # (T,P,D)

    # Geometry-only path: pool pc_tokens across time.
    geo_flat = pc_tokens.reshape(-1, pc_tokens.shape[-1])
    act_geo = policy_geo(geo_flat)

    # Geometry + dynamics path: concat per token.
    full_tokens = torch.cat([pc_tokens, motion_tokens], dim=-1)
    act_full = policy_full(full_tokens)

    return {
        "pc_tokens_shape": tuple(pc_tokens.shape),
        "motion_tokens_shape": tuple(motion_tokens.shape),
        "action_geo_std": act_geo.std().item(),
        "action_full_std": act_full.std().item(),
    }


if __name__ == "__main__":
    out = run_ablation()
    for k, v in out.items():
        if isinstance(v, float):
            print(f"{k:24s} = {v:.6f}")
        else:
            print(f"{k:24s} = {v}")

    # Quick mathematical sanity check on what the second branch really
    # contributes: it is the inter-frame xyz displacement projected through
    # an MLP. That projection is the architectural locus of the +12.8 pp
    # jump — without it the policy is frame-blind; with it the policy can
    # condition on contact, slip, and articulation signals.
    print("\nInterpretation:")
    print("  • pc_tokens      = per-frame (xyz, rgb-feat)        → 3D branch")
    print("  • motion_tokens  = per-frame MLP( xyz_{t-1}, Δxyz ) → dynamics branch")
    print("  • The two are concatenated, so the head sees BOTH")
    print("    'what is there' (geometry) AND 'how it moves' (dynamics).")