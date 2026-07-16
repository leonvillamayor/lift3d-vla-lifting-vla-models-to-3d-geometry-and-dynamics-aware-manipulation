"""
Lift3D-VLA style ablation: 2D vs 2D+RGB+Depth lifting.

This is an illustrative, self-contained snippet showing how one might
structure an ablation that *isolates* the contribution of adding RGB and
depth to a 2D backbone for a Vision-Language-Action policy.

It does NOT reproduce numbers from any specific paper. It uses a tiny
synthetic point-cloud lifting module so the example runs on CPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor, nn


# ----------------------------- configuration ----------------------------- #


@dataclass(frozen=True)
class LiftConfig:
    """Hyper-parameters for the lifting + action head."""

    image_size: int = 64           # H == W
    patch_size: int = 8            # ViT-like patch size
    d_vision: int = 128            # visual token dim
    d_action: int = 7              # action dim (e.g., 6-DoF + gripper)
    fx: float = 120.0              # focal length (pixels), x
    fy: float = 120.0              # focal length (pixels), y
    cx: float = 32.0               # principal point, x
    cy: float = 32.0               # principal point, y
    max_depth: float = 3.0         # clamp range (meters)


# ----------------------------- modules ----------------------------- #


class VisionBackbone2D(nn.Module):
    """Patch-token encoder producing (B, N, d_vision) features."""

    def __init__(self, cfg: LiftConfig) -> None:
        super().__init__()
        self.patch = nn.Conv2d(
            in_channels=3,                 # RGB only
            out_channels=cfg.d_vision,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.norm = nn.LayerNorm(cfg.d_vision)

    def forward(self, rgb: Tensor) -> Tensor:
        # rgb: (B, 3, H, W) in [0, 1]
        feats = self.patch(rgb)                       # (B, D, H/P, W/P)
        feats = feats.flatten(2).transpose(1, 2)      # (B, N, D)
        return self.norm(feats)


class GeometryLifter(nn.Module):
    """Lift 2D RGB features into 3D using per-pixel depth and camera intrinsics.

    For each patch center (u, v) we back-project to a 3D point using depth z:
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        z = z
    The 3D point is concatenated with the 2D feature to form a "geometry-aware"
    token that downstream heads can consume.
    """

    def __init__(self, cfg: LiftConfig, use_depth: bool, use_rgb: bool) -> None:
        super().__init__()
        self.cfg = cfg
        self.use_depth = use_depth
        self.use_rgb = use_rgb

        in_dim = cfg.d_vision + (3 if use_depth else 0)
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, cfg.d_vision),
            nn.GELU(),
            nn.Linear(cfg.d_vision, cfg.d_vision),
        )

    def _patch_centers(self, b: int, h_p: int, w_p: int, device: torch.device) -> Tensor:
        ys = (torch.arange(h_p, device=device) + 0.5) * self.cfg.patch_size
        xs = (torch.arange(w_p, device=device) + 0.5) * self.cfg.patch_size
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        # (h_p, w_p, 2) -> (1, h_p*w_p, 2)
        centers = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)
        return centers.expand(b, -1, -1)

    def forward(self, rgb_feats: Tensor, depth: Tensor | None) -> Tensor:
        """rgb_feats: (B, N, D). depth: (B, 1, H, W) meters or None."""
        b, n, _ = rgb_feats.shape
        device = rgb_feats.device
        side = int(math.sqrt(n))
        centers = self._patch_centers(b, side, side, device)           # (B, N, 2)

        if self.use_depth and depth is not None:
            # Average depth inside each patch via avg-pool aligned with patch stride.
            patch_depth = nn.functional.avg_pool2d(
                depth, kernel_size=self.cfg.patch_size, stride=self.cfg.patch_size
            )                                                         # (B, 1, h_p, w_p)
            patch_depth = patch_depth.flatten(1).transpose(1, 2)       # (B, N, 1)
            patch_depth = patch_depth.clamp(max=self.cfg.max_depth)

            u = centers[..., 0]
            v = centers[..., 1]
            z = patch_depth.squeeze(-1)
            x = (u - self.cfg.cx) * z / self.cfg.fx
            y = (v - self.cfg.cy) * z / self.cfg.fy
            xyz = torch.stack([x, y, z], dim=-1)                       # (B, N, 3)

            tokens = torch.cat([rgb_feats, xyz], dim=-1)
        else:
            tokens = rgb_feats

        return self.fuse(tokens)


class ActionHead(nn.Module):
    """Lightweight MLP head: mean-pool tokens -> action."""

    def __init__(self, cfg: LiftConfig) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_vision, cfg.d_vision),
            nn.GELU(),
            nn.Linear(cfg.d_vision, cfg.d_action),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        pooled = tokens.mean(dim=1)            # (B, D)
        return torch.tanh(self.mlp(pooled))    # actions in [-1, 1]


class LiftPolicy(nn.Module):
    """End-to-end policy used by the ablation."""

    def __init__(self, cfg: LiftConfig, use_depth: bool, use_rgb: bool) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = VisionBackbone2D(cfg) if use_rgb else nn.Identity()
        self.lifter = GeometryLifter(cfg, use_depth=use_depth, use_rgb=use_rgb)
        self.head = ActionHead(cfg)

    def forward(
        self,
        rgb: Tensor,                 # (B, 3, H, W)
        depth: Tensor | None = None, # (B, 1, H, W) or None when use_depth=False
    ) -> Tensor:
        feats = self.backbone(rgb) if not isinstance(self.backbone, nn.Identity) else rgb
        if feats.dim() == 4:                       # (B, D, h, w) -> tokens
            h, w = feats.shape[-2:]
            feats = feats.flatten(2).transpose(1, 2)
        tokens = self.lifter(feats, depth)
        return self.head(tokens)


# ----------------------------- ablation harness ----------------------------- #


def synthetic_batch(cfg: LiftConfig, b: int = 4) -> Tuple[Tensor, Tensor, Tensor]:
    """Generate (rgb, depth, target_actions) for a toy regression task."""
    g = torch.Generator().manual_seed(0)
    rgb = torch.rand(b, 3, cfg.image_size, cfg.image_size, generator=g)
    # Depth correlates with the red channel so the model can actually exploit it.
    depth = (rgb[:, :1] * cfg.max_depth).clone()
    # Target is a deterministic function of (mean_rgb, mean_depth).
    mean_rgb = rgb.mean(dim=(2, 3))
    mean_depth = depth.mean(dim=(2, 3))
    target = torch.cat([mean_rgb[:, :3], mean_depth], dim=1)
    target = torch.tanh(target - 0.5)
    return rgb, depth, target


def train_and_eval(
    cfg: LiftConfig,
    *,
    use_rgb: bool,
    use_depth: bool,
    steps: int = 200,
    lr: float = 3e-3,
) -> dict:
    """Quick train/eval returning final MSE — small enough to run on CPU."""
    device = torch.device("cpu")
    rgb, depth, target = synthetic_batch(cfg)
    val_rgb, val_depth, val_target = synthetic_batch(cfg, b=8)

    model = LiftPolicy(cfg, use_depth=use_depth, use_rgb=use_rgb).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(steps):
        opt.zero_grad()
        pred = model(rgb, depth if use_depth else None)
        loss = loss_fn(pred, target)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(val_rgb, val_depth if use_depth else None)
        val_mse = float(loss_fn(val_pred, val_target))
    return {"use_rgb": use_rgb, "use_depth": use_depth, "val_mse": val_mse}


def run_ablation(cfg: LiftConfig | None = None) -> list[dict]:
    cfg = cfg or LiftConfig()
    runs: list[dict] = []

    # 1) 2D baseline: RGB only, no depth lifting.
    runs.append(train_and_eval(cfg, use_rgb=True, use_depth=False))

    # 2) Same backbone + RGB, plus depth-aware 3D lifting.
    runs.append(train_and_eval(cfg, use_rgb=True, use_depth=True))

    # 3) No-RGB control: only depth (forces the model to rely on geometry).
    runs.append(train_and_eval(cfg, use_rgb=False, use_depth=True))

    return runs


if __name__ == "__main__":
    results = run_ablation()
    print(f"{'config':<24} {'val_mse':>10}")
    for r in results:
        tag = f"rgb={int(r['use_rgb'])} depth={int(r['use_depth'])}"
        print(f"{tag:<24} {r['val_mse']:>10.4f}")