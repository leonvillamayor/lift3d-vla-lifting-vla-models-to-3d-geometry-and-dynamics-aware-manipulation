"""
2D semantic branch of the Lift3D-VLA dual-branch design.

Paper context (arXiv:2607.06564):
  - The 2D branch is the *semantic* half: a frozen ViT (DINOv2 / SigLIP in
    the paper) that provides appearance / "what is it" tokens.
  - On top of it sits the "Dynamics-Aware" module of the title: it predicts
    *where things are going* (motion / flow / future direction) at the same
    patch grid, so the 3D branch can query motion cues via cross-attention.

This file is a didactic, self-contained re-implementation:
  * `FrozenViT2DEncoder`   -> frozen ViT, returns per-patch tokens (B, N, D)
  * `DynamicsAwareHead`    -> fuses tokens + optional flow -> motion features
  * `Lift3D2DBranch`       -> end-to-end wrapper used by the article
  * `_TinyViT`             -> minimal pure-torch ViT used as offline fallback

Requires (optional):  pip install timm torch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Optional backbone: timm ViT (e.g. DINOv2 / SigLIP variants).
# We keep the import inside a try/except so the snippet stays runnable
# even without timm installed (falls back to _TinyViT).
# ---------------------------------------------------------------------------
try:
    import timm

    _HAS_TIMM = True
except Exception:  # pragma: no cover
    _HAS_TIMM = False


# ---------------------------------------------------------------------------
# Minimal ViT used as a deterministic, network-free fallback. It mirrors
# the interface we care about: patch embed -> N transformer blocks -> tokens.
# ---------------------------------------------------------------------------
class _TinyViT(nn.Module):
    def __init__(self, img_size: int = 224, patch: int = 16, dim: int = 384,
                 depth: int = 4, n_heads: int = 6) -> None:
        super().__init__()
        self.patch = patch
        self.grid = img_size // patch
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> tokens (B, N, D)
        tok = self.proj(x)                              # (B, D, g, g)
        tok = tok.flatten(2).transpose(1, 2)            # (B, N, D)
        tok = self.blocks(tok)
        return self.norm(tok)


# ---------------------------------------------------------------------------
# Frozen 2D encoder. The paper keeps this branch frozen; we mirror that by
# disabling gradients on its parameters.
# ---------------------------------------------------------------------------
class FrozenViT2DEncoder(nn.Module):
    """Frozen ViT-style encoder returning per-patch semantic tokens.

    In the paper this is instantiated with DINOv2 or SigLIP; here we either
    use a real `timm` model or a tiny deterministic stand-in.
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224.dino",
        img_size: int = 224,
        use_timm: bool = True,
    ) -> None:
        super().__init__()
        if use_timm and _HAS_TIMM:
            self.backbone = timm.create_model(
                model_name, pretrained=True, num_classes=0,
                img_size=img_size,
            )
            # timm ViTs expose feature dim via `.num_features`
            self.dim: int = self.backbone.num_features
        else:
            self.backbone = _TinyViT(img_size=img_size)
            self.dim = self.backbone.dim

        # Freeze: this branch is the "appearance / what is it" half and
        # must not be updated during VLA training, per the paper.
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W)  ->  tokens: (B, N, D)."""
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(
                f"Expected (B,3,H,W), got {tuple(images.shape)}"
            )
        out = self.backbone(images)
        # timm ViTs w/ num_classes=0 already return (B, N, D) (patch tokens).
        # The tiny fallback also returns (B, N, D). Normalize either way:
        if out.dim() == 2:  # safety: pooled features -> unsqueeze grid
            n = out.shape[-1]
            out = out.unsqueeze(1)
        return out


# ---------------------------------------------------------------------------
# Dynamics-Aware head: predicts per-patch motion / "where it is going"
# features from appearance tokens and an optional flow / track input.
# In the paper this is what makes the 2D branch temporally informative;
# the 3D branch then queries these dynamics tokens via cross-attention.
# ---------------------------------------------------------------------------
class DynamicsAwareHead(nn.Module):
    """Per-patch dynamics predictor.

    Inputs
    ------
    sem_tokens : (B, N, D)   semantic tokens from the frozen 2D encoder
    flow       : (B, 2, H, W) optional optical flow / track field, in px

    Output
    ------
    dyn_tokens : (B, N, D_dyn) per-patch dynamics / direction features
    flow_pred  : (B, 2, H, W)  predicted dense flow (regression aux loss)
    """

    def __init__(self, sem_dim: int, dyn_dim: int = 256,
                 patch: int = 16, img_size: int = 224) -> None:
        super().__init__()
        self.patch = patch
        self.grid = img_size // patch

        # Project semantic tokens into a dynamics-aware space.
        self.proj_sem = nn.Linear(sem_dim, dyn_dim)

        # Project flow (per-pixel 2D vector) into a per-patch summary by
        # average-pooling inside each patch cell.
        self.flow_proj = nn.Conv2d(2, dyn_dim, kernel_size=1)
        self.flow_pool = nn.AvgPool2d(kernel_size=patch, stride=patch)

        # Fuse + a small temporal / directional transformer.
        self.fuse = nn.Linear(dyn_dim * 2, dyn_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dyn_dim, nhead=4, dim_feedforward=dyn_dim * 4,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=2)

        # Dense flow decoder (transposed conv back to image resolution).
        self.flow_dec = nn.Sequential(
            nn.Conv2d(dyn_dim, dyn_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(dyn_dim // 2, 2, kernel_size=patch,
                               stride=patch),
        )
        self.dyn_dim = dyn_dim

    def forward(
        self,
        sem_tokens: torch.Tensor,
        flow: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = sem_tokens.shape
        g = self.grid
        if N != g * g:
            raise ValueError(
                f"token count {N} != grid^2 {g*g}; check patch/img_size"
            )

        sem = self.proj_sem(sem_tokens)                 # (B, N, D_dyn)

        if flow is None:
            flow = torch.zeros(B, 2, g * self.patch, g * self.patch,
                               device=sem_tokens.device,
                               dtype=sem_tokens.dtype)
        if flow.shape[-2:] != (g * self.patch, g * self.patch):
            flow = F.interpolate(flow, size=(g * self.patch, g * self.patch),
                                 mode="bilinear", align_corners=False)
        flow_patch = self.flow_pool(flow)               # (B, 2, g, g)
        flow_emb = self.flow_proj(flow_patch)           # (B, D_dyn, g, g)
        flow_emb = flow_emb.flatten(2).transpose(1, 2)  # (B, N, D_dyn)

        fused = torch.cat([sem, flow_emb], dim=-1)
        dyn = self.fuse(fused)                          # (B, N, D_dyn)
        dyn = self.temporal(dyn)                        # direction-aware

        # Decode dense flow for auxiliary supervision (L1 vs. GT flow).
        flow_pred = self.flow_dec(
            dyn.transpose(1, 2).reshape(B, self.dyn_dim, g, g)
        )
        return dyn, flow_pred


# ---------------------------------------------------------------------------
# End-to-end 2D branch used in the article.
# ---------------------------------------------------------------------------
@dataclass
class Branch2DConfig:
    img_size: int = 224
    patch: int = 16
    dyn_dim: int = 256
    use_timm: bool = True
    timm_model: str = "vit_small_patch16_224.dino"


class Lift3D2DBranch(nn.Module):
    """The 2D semantic + dynamics-aware half of Lift3D-VLA."""

    def __init__(self, cfg: Branch2DConfig = Branch2DConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = FrozenViT2DEncoder(
            model_name=cfg.timm_model,
            img_size=cfg.img_size,
            use_timm=cfg.use_timm,
        )
        # If a timm model has a different patch size, trust it.
        # (We still keep cfg.patch for the dynamics head's pooling math.)
        self.head = DynamicsAwareHead(
            sem_dim=self.encoder.dim,
            dyn_dim=cfg.dyn_dim,
            patch=cfg.patch,
            img_size=cfg.img_size,
        )

    def forward(
        self,
        images: torch.Tensor,
        flow: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        sem_tokens = self.encoder(images)              # frozen -> no grad
        dyn_tokens, flow_pred = self.head(sem_tokens, flow)
        return {
            "sem_tokens": sem_tokens,    # (B, N, D_sem)   appearance
            "dyn_tokens": dyn_tokens,    # (B, N, D_dyn)   "where it goes"
            "flow_pred":  flow_pred,     # (B, 2, H, W)    dense aux output
        }


# ---------------------------------------------------------------------------
# Demo / smoke test.
# ---------------------------------------------------------------------------
def _demo() -> None:
    torch.manual_seed(0)
    cfg = Branch2DConfig(use_timm=_HAS_TIMM)  # auto-fallback if no timm
    branch = Lift3D2DBranch(cfg).eval()

    B, H, W = 2, cfg.img_size, cfg.img_size
    images = torch.randn(B, 3, H, W)

    # Fake optical flow in pixels (e.g. from a tracker).
    flow = torch.randn(B, 2, H, W) * 2.0

    with torch.no_grad():
        out = branch(images, flow=flow)

    print("sem_tokens:", tuple(out["sem_tokens"].shape))
    print("dyn_tokens:", tuple(out["dyn_tokens"].shape))
    print("flow_pred :", tuple(out["flow_pred"].shape))

    # Sanity checks
    assert out["sem_tokens"].ndim == 3
    assert out["dyn_tokens"].ndim == 3
    assert out["flow_pred"].shape == (B, 2, H, W)
    # Frozen check: encoder params must have no grad.
    assert not any(p.requires_grad for p in branch.encoder.parameters())
    print("OK: 2D branch forward pass + frozen-encoder invariants hold.")


if __name__ == "__main__":
    _demo()