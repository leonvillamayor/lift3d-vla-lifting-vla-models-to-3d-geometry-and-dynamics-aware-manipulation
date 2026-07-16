import torch

def aggregate_mean(voxel_feats_per_view: list[torch.Tensor]) -> torch.Tensor:
    """
    voxel_feats_per_view[i].shape = (V, d)  # V voxels, d feature dim
    Devuelve (V, d) con la agregación Mean invariante al orden.
    """
    stacked = torch.stack(voxel_feats_per_view, dim=0)  # (N, V, d)
    return stacked.mean(dim=0)                          # F_voxel = (1/N) Σ F_proj

def project_features_to_voxels(F_proj: torch.Tensor, depth: torch.Tensor,
                               K: torch.Tensor, T: torch.Tensor,
                               voxel_res: float = 0.02) -> torch.Tensor:
    """Unexpect + indexa en una voxel-grid regular de lado voxel_res (metros)."""
    # Placeholder: en la práctica usa ops. diferenciables (gsplat/torch-scatter)
    raise NotImplementedError