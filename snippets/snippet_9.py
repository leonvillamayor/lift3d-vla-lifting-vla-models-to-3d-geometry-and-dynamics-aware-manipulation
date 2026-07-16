# Pseudo-PyTorch del aggregation block (estilo del paper)
def lift_3d_tokenizer(views_2d, point_cloud, cam_extrinsics):
    # views_2d: list[N] of [B, C, H, W] features from frozen 2D encoder
    # point_cloud: [B, V, 3] voxel centers
    # cam_extrinsics: [B, N, 4, 4]

    voxel_feats = []
    for i, F_view in enumerate(views_2d):
        # Project 2D features into 3D voxel grid via extrinsics + intrinsics
        F_proj = project_features_to_voxels(F_view, point_cloud, cam_extrinsics[:, i])
        voxel_feats.append(F_proj)

    # Mean aggregation (default)
    F_3d = torch.stack(voxel_feats, dim=1).mean(dim=1)  # [B, V, C]
    return F_3d  # -> GC-MAE encoder