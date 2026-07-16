class CubeFaceAggregator(nn.Module):
    def __init__(self, n_faces: int = 6, pe_dim: int = 256, learnable_null: bool = True):
        super().__init__()
        self.n_faces = n_faces
        self.null_token = nn.Parameter(torch.zeros(pe_dim)) if learnable_null else None

    def forward(self, pe_2d: Tensor, face_idx: Tensor, valid: Tensor) -> Tensor:
        """
        pe_2d:  (N_pc, n_faces, pe_dim)  — preentrenado, ||·||=1
        face_idx: (N_pc, n_faces)        — índice celda voxel por cara
        valid:    (N_pc, n_faces) bool   — cara dentro del frustum
        returns:  (N_pc, pe_dim)
        """
        gathered = pe_2d.gather(1, face_idx.unsqueeze(-1).expand(-1, -1, pe_2d.size(-1)))
        n_valid = valid.float().sum(1, keepdim=True).clamp(min=1.0)
        pe_3d = (gathered * valid.unsqueeze(-1)).sum(1) / n_valid
        # Normalización opcional para invariancia a oclusión
        pe_3d = F.normalize(pe_3d, dim=-1)
        # Fallback: punto sin geometría visible
        if self.null_token is not None:
            pe_3d = torch.where(valid.any(1, keepdim=True), pe_3d, self.null_token)
        return pe_3d