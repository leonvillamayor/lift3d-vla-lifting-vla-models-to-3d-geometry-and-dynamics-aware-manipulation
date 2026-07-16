pixels = [(u1, v1), (u2, v2), (u3, v3)]   # 3 vistas
depths = [z1, z2, z3]                       # depth métrico (m)
K, R, t = load_camera_intrinsics_extrinsics()

# 1. Unproject a mundo
X_world = [(R[c].T @ (np.linalg.inv(K) @ np.array([u, v, 1]) * z - t[c]))
           for c, ((u, v), z) in enumerate(zip(pixels, depths))]

# 2. Voxelizar
voxel_idx = [tuple(np.floor(X / 0.02).astype(int)) for X in X_world]

# 3. Gather + scatter_mean de PEs
pe_voxel = scatter_mean(
    src=torch.stack([pe_table[u, v] for (u, v) in pixels]),
    index=voxel_idx, dim=0
)  # shape: (N_voxels, D_pe)