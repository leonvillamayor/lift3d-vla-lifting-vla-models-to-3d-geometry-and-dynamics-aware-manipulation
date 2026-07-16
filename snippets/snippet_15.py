# Conceptual: forward 3D (fine-tuning / inferencia)
pe_3d = scatter_mean(pe_table[u, v], voxel_idx)   # (B, N_voxels, D_pe)
x_voxel = voxel_features + pe_3d