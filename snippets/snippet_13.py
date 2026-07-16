# 1) Unproject pixel → mundo (K, R, t fijos por cámara)
pts_cam = Z * (K_inv @ [u, v, 1])          # Z = depth métrico
pts_world = R.T @ (pts_cam - t)

# 2) Voxelización con voxel_res ≈ 2 cm
i, j, k = floor(pts_world / voxel_res)     # índices enteros

# 3) Scatter-mean del pe_table FROZEN
F_pe_voxel[i, j, k] += pe_table[u, v]
F_pe_voxel /= count[i, j, k]               # media sobre N vistas