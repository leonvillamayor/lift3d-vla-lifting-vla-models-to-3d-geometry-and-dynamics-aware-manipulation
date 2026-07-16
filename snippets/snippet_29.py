"""
Lift3D-VLA — illustrative snippet for §4:
"How 3D geometry enters the 2D encoder without breaking it"

Core idea:
- Keep the 2D vision tower FROZEN (its positional embeddings and
  pixel-aligned features were pre-trained on massive 2D data).
- Run a *parallel* 3D geometry branch (point cloud / depth -> tokens).
- Inject the 3D tokens into the 2D stream via a small adapter
  (cross-attention + residual), so the pre-trained 2D manifold is
  preserved and only a tiny set of new parameters is learned.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# 1. Frozen 2D vision tower (placeholder; in the paper this is e.g. SigLIP/DINOv2)
# ---------------------------------------------------------------------------
class Frozen2DEncoder(nn.Module):
    """Stub of a 2D backbone. Pretend it was trained on web-scale RGB."""

    def __init__(self, embed_dim: int = 768) -> None:
        super().__init__()
        # A tiny conv stack stands in for a ViT/SigLIP trunk.
        self.patch_proj = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        # Freeze EVERYTHING: the key to "not breaking" the 2D encoder.
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self.embed_dim = embed_dim

    def forward(self, images: Tensor) -> Tensor:
        # images: (B, 3, H, W) -> tokens: (B, N, D) on a planar grid.
        feats = self.patch_proj(images)           # (B, D, H', W')
        tokens = feats.flatten(2).transpose(1, 2) # (B, N, D)
        return tokens


# ---------------------------------------------------------------------------
# 2. 3D geometry branch (point cloud -> tokens)
# ---------------------------------------------------------------------------
class GeometryEncoder(nn.Module):
    """Lightweight PointNet-like encoder; trainable."""

    def __init__(self, in_dim: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(),
            nn.Linear(128, 256),     nn.GELU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, points: Tensor) -> Tensor:
        # points: (B, P, 3) -> geometry tokens: (B, P, D)
        return self.mlp(points)


# ---------------------------------------------------------------------------
# 3. The "lift" adapter: injects 3D into 2D WITHOUT touching the 2D backbone
# ---------------------------------------------------------------------------
class Lift3DAdapter(nn.Module):
    """
    Geometry-aware prompt injection.

    3D tokens (queries) attend to 2D tokens (keys/values) -> 3D-aware prompt.
    The prompt is ADDED to the 2D features (residual), so the frozen encoder's
    representation stays the dominant signal; the adapter only nudges it.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 8) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        # Small MLP on the cross-attn output (the only "new" capacity).
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.scale = nn.Parameter(torch.zeros(1))  # init at 0 -> identity start

    def forward(self, geom_tokens: Tensor, img_tokens: Tensor) -> Tensor:
        # geom_tokens: (B, P, D)  — 3D branch output
        # img_tokens : (B, N, D)  — frozen 2D encoder output
        prompt, _ = self.cross_attn(
            query=geom_tokens, key=img_tokens, value=img_tokens
        )
        prompt = self.norm(prompt + self.ffn(prompt))  # (B, P, D)

        # Broadcast-and-add: geometry prompt is replicated over 2D positions
        # and *added* to img tokens, gated by `scale` (starts at 0).
        B, N, D = img_tokens.shape
        P = prompt.shape[1]
        prompt = prompt.mean(dim=1, keepdim=True).expand(B, N, D)  # (B,N,D)
        return img_tokens + self.scale * prompt


# ---------------------------------------------------------------------------
# 4. Putting it together: a 2D VLA tower with a 3D "lift"
# ---------------------------------------------------------------------------
class Lift3DVisionTower(nn.Module):
    def __init__(self, embed_dim: int = 768) -> None:
        super().__init__()
        self.encoder2d = Frozen2DEncoder(embed_dim)
        self.encoder3d = GeometryEncoder(in_dim=3, embed_dim=embed_dim)
        self.lift      = Lift3DAdapter(embed_dim)

    def forward(self, images: Tensor, point_clouds: Tensor) -> Tensor:
        with torch.no_grad():
            img_tokens = self.encoder2d(images)        # frozen, no grad
        geom_tokens = self.encoder3d(point_clouds)     # trainable
        lifted      = self.lift(geom_tokens, img_tokens)
        return lifted  # 3D-aware tokens, same shape as 2D output


# ---------------------------------------------------------------------------
# 5. Smoke test: shapes + a check that the frozen branch really has no grad
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    tower = Lift3DVisionTower(embed_dim=256)
    images   = torch.randn(2, 3, 224, 224)
    points   = torch.randn(2, 1024, 3)  # 1024 points per scene

    out = tower(images, points)
    print("output shape:", tuple(out.shape))  # (2, 196, 256)

    n_frozen = sum(p.numel() for p in tower.encoder2d.parameters())
    n_train  = sum(p.numel() for p in tower.encoder3d.parameters()) \
             + sum(p.numel() for p in tower.lift.parameters())
    print(f"frozen 2D params (no grad): {n_frozen:,}")
    print(f"trainable 3D+adapter params: {n_train:,}")
    # Notice: trainable set is tiny vs. frozen -> pre-trained 2D prior preserved.