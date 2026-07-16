"""
Lift3D-VLA — Ingeniería del `pe_table` como puente 2D→3D (mecánica a nivel de código).

Idea: el encoder 2D preentrenado (DINOv2) tiene un positional encoding aprendible
de tamaño (H_p, W_p, D_pe). Para que "hable 3D" sin reentrenar, Lift3D indexa ese
mismo tensor usando un mapeo geométrico (proyección pinhole + discretización)
desde coordenadas 3D del mundo a celdas (i, j) del plano imagen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. PE table: codebook posicional 2D preentrenado, FROZEN.
# --------------------------------------------------------------------------- #
class FrozenPETable(nn.Module):
    """Wrapper frozen sobre un `nn.Embedding` 2D indexado por (i, j).

    En el paper, este tensor es **idéntico** al que usó DINOv2 durante su
    pre-training; aquí lo inicializamos aleatoriamente pero con `requires_grad=False`
    para ilustrar el contrato "no se reentrena".
    """

    def __init__(self, h: int, w: int, d_pe: int) -> None:
        super().__init__()
        # Almacenamos como buffer (no parámetro entrenable) para que state_dict
        # lo mueva, pero torch.optim no lo toque.
        pe = torch.randn(h, w, d_pe) / math.sqrt(d_pe)
        self.register_buffer("pe", pe, persistent=True)
        self.h, self.w, self.d_pe = h, w, d_pe

    @torch.no_grad()
    def lookup(self, ij: torch.Tensor) -> torch.Tensor:
        """`ij`: (..., 2) long con coordenadas (i, j) en [0,H) x [0,W) → (..., D_pe)."""
        i = ij[..., 0].clamp(0, self.h - 1)
        j = ij[..., 1].clamp(0, self.w - 1)
        return self.pe[i, j]


# --------------------------------------------------------------------------- #
# 2. Proyección pinhole 3D → 2D + discretización.
# --------------------------------------------------------------------------- #
@dataclass
class Intrinsics:
    """Intrinsics mínimos (pinhole). `cx, cy` ya en coords de celda (no píxel)."""
    fx: float
    fy: float
    cx: float
    cy: float


def project_xyz_to_ij(
    xyz: torch.Tensor, K: Intrinsics, H: int, W: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Proyecta puntos 3D (N,3) a índices de celda (i,j) y máscara de validez.

    `xyz` está en el frame de la cámara (z hacia adelante). Devuelve:
        ij     : (N, 2) long, coordenadas cuantizadas con `floor`.
        valid  : (N,)  bool, True si z>0 y dentro de [0,H) x [0,W).
    """
    x, y, z = xyz.unbind(dim=-1)
    safe_z = torch.where(z.abs() < 1e-6, torch.full_like(z, 1e-6), z)
    u = K.fx * (x / safe_z) + K.cx
    v = K.fy * (y / safe_z) + K.cy
    i = u.floor().long()
    j = v.floor().long()
    valid = (z > 0) & (i >= 0) & (i < H) & (j >= 0) & (j < W)
    ij = torch.stack([i, j], dim=-1)
    return ij, valid


# --------------------------------------------------------------------------- #
# 3. GeometryLift: el puente "3D coords → PE vector".
# --------------------------------------------------------------------------- #
class GeometryLift(nn.Module):
    """Convierte coordenadas 3D en positional encodings 2D preentrenados.

    Flujo:
        xyz (B, N, 3) ─► project_xyz_to_ij ─► FrozenPETable.lookup ─► pe (B, N, D_pe)
                                                          │
                                                          └─► zero-pad donde !valid
    """

    def __init__(self, h: int, w: int, d_pe: int, intrinsics: Intrinsics) -> None:
        super().__init__()
        self.pe_table = FrozenPETable(h, w, d_pe)
        self.H, self.W, self.D = h, w, d_pe
        self.K = intrinsics

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # xyz: (B, N, 3) en frame cámara
        B, N, _ = xyz.shape
        ij_list, valid_list, pe_list = [], [], []
        for b in range(B):
            ij, valid = project_xyz_to_ij(xyz[b], self.K, self.H, self.W)
            pe = self.pe_table.lookup(ij)              # (N, D_pe)
            pe = pe * valid.unsqueeze(-1).float()      # zero-pad fuera de frustum
            ij_list.append(ij)
            valid_list.append(valid)
            pe_list.append(pe)
        pe_out = torch.stack(pe_list, dim=0)           # (B, N, D_pe)
        valid_out = torch.stack(valid_list, dim=0)     # (B, N)
        return pe_out, valid_out


# --------------------------------------------------------------------------- #
# 4. Micro-test: confirma el flujo y la "frozenness".
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    torch.manual_seed(0)
    H, W, D_PE = 16, 22, 64          # tamaño típico del PE de un ViT-S DINOv2 patcheado
    K = Intrinsics(fx=300.0, fy=300.0, cx=W / 2, cy=H / 2)

    lift = GeometryLift(H, W, D_PE, K)

    # Asegurar que nada es entrenable (contrato frozen).
    n_trainable = sum(p.numel() for p in lift.parameters() if p.requires_grad)
    assert n_trainable == 0, f"pe_table debe ser frozen, hay {n_trainable} params"

    # Batch de puntos 3D: algunos delante de la cámara, otros detrás / fuera.
    xyz = torch.tensor([
        # (B=1, N=5, 3)
        [[0.0, 0.0, 1.0],   # centro → ~ (cx, cy)
         [0.1, 0.1, 1.0],   # desplazado
         [0.0, 0.0, -1.0],  # detrás (z<0) → debe quedar enmascarado
         [5.0, 5.0, 0.1],   # fuera de imagen → enmascarado
         [0.05, -0.05, 2.0]]
    ])
    pe, valid = lift(xyz)
    assert pe.shape == (1, 5, D_PE)
    assert valid.shape == (1, 5)
    assert valid[0, 0].item() is True or valid[0, 0].item() == 1
    assert valid[0, 2].item() is False or valid[0, 2].item() == 0   # z<0
    assert torch.all(pe[0, ~valid[0]] == 0.0), "puntos inválidos deben ser zero-pad"
    print("pe shape:", tuple(pe.shape), "| valids:", valid[0].tolist())


if __name__ == "__main__":
    _self_test()