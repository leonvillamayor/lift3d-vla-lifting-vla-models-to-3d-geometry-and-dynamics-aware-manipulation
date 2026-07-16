import torch
import torch.nn as nn

class FuturePointFlowHead(nn.Module):
    """Predicts per-point displacement Δ at t+1 from geometry tokens at t.
    Supervision: MSE against GT flow from Rectified Point Flow pre-training.
    """
    def __init__(self, dim: int = 256, horizon: int = 1):
        super().__init__()
        self.horizon = horizon
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, 3 * horizon),  # dx, dy, dz per future step
        )

    def forward(self, point_tokens: torch.Tensor) -> torch.Tensor:
        # point_tokens: [B, N, dim]   (N points, already lifted by GC-MAE)
        delta = self.mlp(point_tokens)                # [B, N, 3*H]
        return delta.view(*delta.shape[:2], self.horizon, 3)

    def loss(self, pred: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        # gt_flow from RPF: [B, N, H, 3]
        return torch.nn.functional.mse_loss(pred, gt_flow)