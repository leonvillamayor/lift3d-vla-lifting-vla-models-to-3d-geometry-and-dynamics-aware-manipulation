def virtual_project_pe(F_2d, depth, K_inv, R, t, voxel_res, pe_table):
    # F_2d: (H, W, D) features + PE embedding del backbone 2D
    # depth: (H, W) mapa de profundidad en metros
    # pe_table: nn.Embedding(H_pe * W_pe, D) — frozen del ViT/SigLIP

    # 1) Backproject pixel grid a mundo
    u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    rays = K_inv @ torch.stack([u, v, torch.ones_like(u)], dim=-1)  # (H, W, 3)
    pts_cam = rays * depth[..., None]                                # (H, W, 3)
    pts_world = (R.T @ (pts_cam - t).reshape(-1, 3).T).T            # (N_pts, 3)

    # 2) Voxelize
    vox_idx = torch.floor(pts_world / voxel_res).long()              # (N_pts, 3)

    # 3) Acumula por vóxel usando el PE 2D original (no uno nuevo)
    pe_2d = pe_table(pixel_to_pe_idx(u, v))                          # (H, W, D)
    scatter_mean(vox_idx, pe_2d.reshape(-1, D), out=F_pe_voxel)      # → (Vx, Vy, Vz, D)
    return F_pe_voxel