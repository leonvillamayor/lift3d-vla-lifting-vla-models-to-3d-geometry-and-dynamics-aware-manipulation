"""
3D Tokenizer para Lift3D-VLA (arXiv:2607.06564)
================================================
Contrato del módulo:
    Input : N vistas RGB (típicamente N=4 en dual-arm: 2 por brazo)
            + depth/geometría opcional por vista
    Output: volumen de tokens 3D (B, T_3d, D_3d) que se prefija al VLA

Regla "2D semantic + freeze + lift":
    - Encoder 2D preentrenado (DINOv2 / SAM-style) con pesos CONGELADOS.
    - Solo se entrenan adaptadores LoRA sobre las proyecciones lineales.
    - Un módulo de lifting epipolar / depth-aware funde las N vistas
      en una representación voxel-grid y luego la "aplana" en tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# 1. LoRA: adaptadores de bajo rango sobre proyecciones lineales
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Linear con LoRA: base congelada + adaptador entrenable de rango r."""

    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=True)
        for p in self.base.parameters():
            p.requires_grad_(False)  # <-- FREEZE del encoder 2D

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.scale = alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(x)) * self.scale


# ---------------------------------------------------------------------------
# 2. Encoder 2D semántico (sustituto limpio de DINOv2/SAM para el ejemplo)
# ---------------------------------------------------------------------------
@dataclass
class Encoder2DConfig:
    in_channels: int = 3
    embed_dim: int = 768          # p.ej. ViT-B/14 de DINOv2
    patch_size: int = 14
    depth: int = 2                # pocas capas para que el ejemplo sea ligero
    lora_r: int = 8


class Semantic2DEncoder(nn.Module):
    """
    Patch-encoder tipo ViT simplificado. En el paper real se reemplaza por
    DINOv2 / SAM congelado; aquí mostramos la MISMA INTERFAZ para que el
    lector entienda qué entra y qué sale.
    """

    def __init__(self, cfg: Encoder2DConfig):
        super().__init__()
        self.cfg = cfg
        self.patchify = nn.Conv2d(cfg.in_channels, cfg.embed_dim,
                                  kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.embed_dim, nhead=12, dim_feedforward=cfg.embed_dim * 4,
                batch_first=True, activation="gelu"
            )
            for _ in range(cfg.depth)
        ])
        # Reemplazamos las proyecciones lineales de cada bloque por LoRA.
        for blk in self.blocks:
            lin1: nn.Linear = blk.linear1  # type: ignore[assignment]
            lin2: nn.Linear = blk.linear2  # type: ignore[assignment]
            blk.linear1 = LoRALinear(lin1.in_features, lin1.out_features, r=cfg.lora_r)
            blk.linear2 = LoRALinear(lin2.in_features, lin2.out_features, r=cfg.lora_r)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B*N, 3, H, W) -> (B*N, P, D) con P = (H/ps)*(W/ps)."""
        feats = self.patchify(images)                     # (B*N, D, H/ps, W/ps)
        feats = rearrange(feats, "b d h w -> b (h w) d")
        for blk in self.blocks:
            feats = blk(feats)
        return feats                                      # tokens 2D por parche


# ---------------------------------------------------------------------------
# 3. Lifting epipolar / depth-aware: N vistas -> voxel-grid -> tokens 3D
# ---------------------------------------------------------------------------
@dataclass
class Lift3DConfig:
    embed_dim: int = 768
    grid_size: int = 16              # resolución voxel-grid (16^3 por defecto)
    num_cameras: int = 4             # dual-arm típico
    fusion_layers: int = 2


def unproject_to_voxels(
    features_2d: torch.Tensor,       # (B*N, P, D)
    depths: torch.Tensor,            # (B*N, P)   profundidad por parche
    intrinsics: torch.Tensor,        # (B, N, 3, 3)
    extrinsics: torch.Tensor,        # (B, N, 4, 4) world<-cam
    grid_size: int,
) -> torch.Tensor:
    """
    Reproyecta tokens 2D a coordenadas voxel-grid usando depth + calibración.
    Implementación vectorizada con scatter_add para el ejemplo.
    """
    B = intrinsics.shape[0]
    N = intrinsics.shape[1]
    D = features_2d.shape[-1]

    # Separamos batch y cámara
    f2d = rearrange(features_2d, "(b n) p d -> b n p d", b=B, n=N)
    dep = rearrange(depths, "(b n) p -> b n p", b=B, n=N)
    P = f2d.shape[2]

    # Píxeles centrados en cada parche (esquina -> centro)
    # Para el ejemplo usamos un grid regular; en producción se indexa por (h,w)
    h = w = int(P ** 0.5)
    ys, xs = torch.meshgrid(
        torch.arange(h, device=f2d.device),
        torch.arange(w, device=f2d.device), indexing="ij")
    pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).float()  # (h,w,3)
    pix = pix.reshape(1, 1, P, 3).expand(B, N, P, 3)

    # Ray directions en cam-frame
    K_inv = torch.linalg.inv(intrinsics)                             # (B,N,3,3)
    rays_cam = torch.einsum("bnij,bnpj->bnpi", K_inv, pix)           # (B,N,P,3)
    rays_world = torch.einsum(
        "bnij,bnpj->bnpi", extrinsics[..., :3, :3], rays_cam
    )                                                                  # (B,N,P,3)
    origins = extrinsics[..., :3, 3].unsqueeze(-2).expand_as(rays_world)

    pts_world = origins + rays_world * dep.unsqueeze(-1)              # (B,N,P,3)

    # Voxelización simple a [-1,1]^3
    pts_norm = pts_world.clamp(-1, 1)                                # asumimos escena normalizada
    voxel_idx = ((pts_norm + 1.0) * 0.5 * grid_size).long()
    voxel_idx = voxel_idx.clamp(0, grid_size - 1)

    # Scatter-add en un tensor (B, G, G, G, D)
    voxels = torch.zeros(B, grid_size, grid_size, grid_size, D,
                         device=f2d.device, dtype=f2d.dtype)
    counts = torch.zeros(B, grid_size, grid_size, grid_size, 1,
                         device=f2d.device, dtype=f2d.dtype)

    # Flat indexes para scatter_add_
    flat_idx = (
        voxel_idx[..., 0] * grid_size * grid_size
        + voxel_idx[..., 1] * grid_size
        + voxel_idx[..., 2]
    )                                                                 # (B,N,P)

    f2d_flat = rearrange(f2d, "b n p d -> b (n p) d")
    idx_flat = rearrange(flat_idx, "b n p -> b (n p)").unsqueeze(-1).expand(-1, -1, D)

    voxels.scatter_add_(1, idx_flat, f2d_flat)
    counts.scatter_add_(
        1,
        rearrange(flat_idx, "b n p -> b (n p) 1"),
        torch.ones_like(f2d_flat[..., :1]),
    )
    voxels = voxels / counts.clamp_min(1.0)                           # promedio por celda
    return voxels                                                     # (B, G, G, G, D)


class Lift3DHead(nn.Module):
    """Fusiona el voxel-grid en tokens 3D listos para el VLA."""

    def __init__(self, cfg: Lift3DConfig):
        super().__init__()
        self.cfg = cfg
        self.spatial_mix = nn.Sequential(
            nn.Conv3d(cfg.embed_dim, cfg.embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(cfg.embed_dim, cfg.embed_dim, kernel_size=3, padding=1),
        )
        # Reducimos G^3 -> T_3d tokens via pooling jerárquico
        self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        # voxels: (B, G, G, G, D) -> (B, D, G, G, G)
        x = rearrange(voxels, "b gx gy gz d -> b d gx gy gz")
        x = self.spatial_mix(x)                                       # mezcla local 3D
        x = rearrange(x, "b d gx gy gz -> b (gx gy gz) d")
        x = self.proj(x)
        return self.norm(x)                                           # (B, T_3d, D)


# ---------------------------------------------------------------------------
# 4. 3D Tokenizer (módulo completo)
# ---------------------------------------------------------------------------
class Lift3DTokenizer(nn.Module):
    """
    Implementación de referencia del 3D Tokenizer de Lift3D-VLA.
    Solo se entrenan: adaptadores LoRA + Lift3DHead.
    El encoder 2D queda CONGELADO.
    """

    def __init__(
        self,
        enc_cfg: Encoder2DConfig | None = None,
        lift_cfg: Lift3DConfig | None = None,
    ):
        super().__init__()
        self.enc_cfg = enc_cfg or Encoder2DConfig()
        self.lift_cfg = lift_cfg or Lift3DConfig(embed_dim=self.enc_cfg.embed_dim)

        self.encoder_2d = Semantic2DEncoder(self.enc_cfg)
        self.lift_head = Lift3DHead(self.lift_cfg)

        # Congelamos TODO salvo LoRA + lift head
        for n, p in self.encoder_2d.named_parameters():
            if "lora_" not in n:
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,          # (B, N, 3, H, W)
        depths: torch.Tensor,          # (B, N, H, W) o (B, N, P) ya reducido
        intrinsics: torch.Tensor,      # (B, N, 3, 3)
        extrinsics: torch.Tensor,      # (B, N, 4, 4)
    ) -> torch.Tensor:
        B, N, C, H, W = images.shape

        # 1) Encode 2D (con LoRA, base congelada)
        x = rearrange(images, "b n c h w -> (b n) c h w")
        feat_2d = self.encoder_2d(x)                                # (B*N, P, D)

        # 2) Reducir depth por parche si viene a resolución píxel
        if depths.shape[-1] == W:
            ps = self.enc_cfg.patch_size
            depths_patch = F.avg_pool2d(depths.view(B * N, 1, H, W),
                                        kernel_size=ps, stride=ps)
            depths_patch = rearrange(depths_patch, "bn 1 h w -> (bn) (h w)")
        else:
            depths_patch = depths.reshape(B * N, -1)

        # 3) Lifting a voxel-grid
        voxels = unproject_to_voxels(
            feat_2d, depths_patch, intrinsics, extrinsics, self.lift_cfg.grid_size
        )                                                             # (B,G,G,G,D)

        # 4) Cabeza 3D -> tokens
        tokens_3d = self.lift_head(voxels)                            # (B, T_3d, D)
        return tokens_3d


# ---------------------------------------------------------------------------
# 5. Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, N = 2, 4            # dual-arm: 2 batch, 4 cámaras
    H = W = 224
    ps = 14
    G = 16

    images = torch.randn(B, N, 3, H, W)
    depths = torch.rand(B, N, H, W)                 # depth en [0,1] aprox
    K = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()
    K[..., 0, 0] = K[..., 1, 1] = H                 # focal ~ H
    K[..., 0, 2] = K[..., 1, 2] = H / 2
    E = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).clone()

    tok = Lift3DTokenizer()
    out = tok(images, depths, K, E)
    print("tokens 3d:", out.shape)                  # (2, G^3, 768) = (2, 4096, 768)

    trainable = sum(p.numel() for p in tok.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in tok.parameters())
    print(f"trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M  "
          f"(≈ {100*trainable/total:.1f}% entrenable)")