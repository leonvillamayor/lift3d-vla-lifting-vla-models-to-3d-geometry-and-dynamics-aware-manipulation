"""
GC-MAE mask-ratio sweep (mini).
Replica simplificada del análisis del paper Lift3D-VLA:
   - Tokenización de la nube de puntos en patches.
   - Enmascarado aleatorio con ratio r ∈ {0.3, ..., 0.9}.
   - Encoder-decoder tipo MAE con dos cabezas: geom (Chamfer proxy) + futura.
   - Curva de pérdida total para localizar el "sweet spot".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# 1. Datos sintéticos: una nube de puntos "tipo objeto" en movimiento
# ---------------------------------------------------------------------------
@dataclass
class PointSample:
    points: Tensor      # (N, 3)
    future: Tensor      # (N, 3)  desplazamiento futuro por punto


def make_synthetic_batch(
    batch_size: int = 8,
    n_points: int = 512,
    seed: int = 0,
) -> PointSample:
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(batch_size, n_points, 3, generator=g)
    # Movimiento futuro suave: campo de velocidades lineal.
    flow = 0.05 * pts + 0.01 * torch.randn_like(pts)
    return PointSample(points=pts, future=flow)


# ---------------------------------------------------------------------------
# 2. Tokenización tipo Point-MAE: FPS + KNN grouping (aprox. con voxel hash)
# ---------------------------------------------------------------------------
def farthest_point_sample(points: Tensor, k: int) -> Tensor:
    """FPS simple (suficiente para experimentar con ratios)."""
    b, n, _ = points.shape
    centroids = torch.zeros(b, k, dtype=torch.long, device=points.device)
    dist = torch.full((b, n), 1e10, device=points.device)
    farthest = torch.zeros(b, dtype=torch.long, device=points.device)
    batch_arange = torch.arange(b, device=points.device)
    for i in range(k):
        centroids[:, i] = farthest
        centroid = points[batch_arange, farthest].unsqueeze(1)  # (B,1,3)
        d = ((points - centroid) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        farthest = dist.argmax(-1)
    return centroids


def patchify(points: Tensor, future: Tensor, n_patches: int = 64, k: int = 8) -> tuple[Tensor, Tensor]:
    """Agrupa la nube en n_patches centroides con k vecinos cada uno."""
    idx = farthest_point_sample(points, n_patches)            # (B, P)
    # Recolecta vecinos por radio (aquí: top-k por distancia).
    # Para mantenerlo rápido, usamos gather sobre las P centroides más cercanas
    # a partir de la matriz de distancias aleatoria del FPS.
    centroids = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    # Distancias (B, N, P) -> top-k vecinos por centroide.
    d = torch.cdist(points, centroids)                         # (B, N, P)
    _, knn_idx = d.topk(k, dim=1, largest=False)               # (B, k, P)
    # Reorganiza puntos y futuro en patches: (B, P, k, 3)
    gather_idx = knn_idx.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, 3)
    pts_patches = torch.gather(points.unsqueeze(2).expand(-1, -1, k, -1), 1,
                                gather_idx.permute(0, 2, 1, 3))
    fut_patches = torch.gather(future.unsqueeze(2).expand(-1, -1, k, -1), 1,
                                gather_idx.permute(0, 2, 1, 3))
    # Token del patch = centroide + xyz medio de los vecinos.
    tokens_geom = torch.cat([centroids, pts_patches.mean(1)], dim=-1)  # (B, P, 6)
    tokens_future = fut_patches.mean(1)                                  # (B, P, 3)
    return tokens_geom, tokens_future


# ---------------------------------------------------------------------------
# 3. Enmascarado estilo MAE con ratio configurable
# ---------------------------------------------------------------------------
def random_masking(tokens: Tensor, ratio: float) -> tuple[Tensor, Tensor, Tensor]:
    """Devuelve (tokens_visibles, mask, ids_restore)."""
    b, p, d = tokens.shape
    n_keep = int(p * (1 - ratio))
    noise = torch.rand(b, p, device=tokens.device)
    ids_shuffle = noise.argsort(1)
    ids_restore = ids_shuffle.argsort(1)
    ids_keep = ids_shuffle[:, :n_keep]
    tokens_vis = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))
    mask = torch.ones(b, p, device=tokens.device)
    mask[:, :n_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return tokens_vis, mask, ids_restore


# ---------------------------------------------------------------------------
# 4. Encoder/Decoder con cabeza geométrica + cabeza futura (GC-MAE)
# ---------------------------------------------------------------------------
class GCMAE(nn.Module):
    def __init__(self, d_in_geom: int = 6, d_in_fut: int = 3, dim: int = 128,
                 depth: int = 4, n_patches: int = 64) -> None:
        super().__init__()
        self.n_patches = n_patches
        self.proj_geom = nn.Linear(d_in_geom, dim)
        self.proj_fut = nn.Linear(d_in_fut, dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=dim * 2,
                                        batch_first=True, activation="gelu"),
            num_layers=depth,
        )
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=dim * 2,
                                        batch_first=True, activation="gelu"),
            num_layers=2,
        )
        self.mask_token = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.head_geom = nn.Linear(dim, d_in_geom)
        self.head_fut = nn.Linear(dim, d_in_fut)

    def forward(self, geom: Tensor, fut: Tensor, ratio: float) -> dict[str, Tensor]:
        vis_g, mask, ids_restore = random_masking(self.proj_geom(geom), ratio)
        vis_f = torch.gather(self.proj_fut(fut), 1,
                              ids_restore.argsort(1)[:, : vis_g.size(1)])
        z = self.encoder(vis_g + vis_f)
        # Decodificar todos los patches
        full = torch.cat([z, self.mask_token[None, None, :].expand(z.size(0),
                          self.n_patches - z.size(1), -1)], dim=1)
        full = torch.gather(full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, z.size(-1)))
        dec = self.decoder(full)
        pred_geom = self.head_geom(dec)
        pred_fut = self.head_fut(dec)
        return {"pred_geom": pred_geom, "pred_fut": pred_fut, "mask": mask,
                "target_geom": geom, "target_fut": fut}


# ---------------------------------------------------------------------------
# 5. Pérdida y bucle de evaluación para el sweep de mask ratio
# ---------------------------------------------------------------------------
def chamfer_proxy(a: Tensor, b: Tensor) -> Tensor:
    """Proxy barato: MSE entre patches reconstruidos y originales."""
    return ((a - b) ** 2).sum(-1).mean()


def gcmae_loss(out: dict[str, Tensor], lam: float = 0.5) -> Tensor:
    mask = out["mask"].unsqueeze(-1)
    loss_g = ((out["pred_geom"] - out["target_geom"]) ** 2 * mask).sum() / (mask.sum() * out["target_geom"].size(-1) + 1e-6)
    loss_f = ((out["pred_fut"] - out["target_fut"]) ** 2 * mask).sum() / (mask.sum() * out["target_fut"].size(-1) + 1e-6)
    return loss_g + lam * loss_f


def sweep_mask_ratios(model: GCMAE, batch: PointSample,
                       ratios: Iterable[float], steps: int = 50) -> dict[float, float]:
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    results: dict[float, float] = {}
    geom, fut = patchify(batch.points, batch.future)
    for r in ratios:
        torch.manual_seed(int(r * 1000))
        last = 0.0
        for _ in range(steps):
            opt.zero_grad()
            out = model(geom, fut, ratio=r)
            loss = gcmae_loss(out)
            loss.backward()
            opt.step()
            last = float(loss.detach())
        results[round(r, 2)] = last
    return results


# ---------------------------------------------------------------------------
# 6. Ejecutar el experimento
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(42)
    batch = make_synthetic_batch()
    model = GCMAE()
    ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    losses = sweep_mask_ratios(model, batch, ratios)

    best_r = min(losses, key=losses.get)
    print("\nMask-ratio sweep (GC-MAE)")
    print("-" * 30)
    for r, l in losses.items():
        bar = "█" * int(l * 50)
        print(f"  r={r:.1f}  loss={l:.4f}  {bar}")
    print(f"\n  -> sweet spot ≈ {best_r}  (paper Lift3D-VLA reporta ~0.6)")
    print(f"  -> Reconstrucción proxy (Chamfer≈MSE) mínima en r*={best_r}")


if __name__ == "__main__":
    main()