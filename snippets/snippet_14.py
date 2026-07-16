# Conceptual: forward 2D (pre-training)
pe_2d = pe_table[i, j]          # (B, N_patches, D_pe)
x = patch_features + pe_2d