# Inputs
points_xyz   # (B, N_pc=128, 3)
pe_tables    # list[6] de (H_p, W_p, d_h=4096)

# Internos
proj_uv      # (B, N_pc, 6, 2)
pe_indexed   # (B, N_pc, 6, d_h)
pe_3d        # (B, N_pc, d_h)        # mean sobre eje 6

# Fusión
geo_tokens   # (B, N_pc, d_h) = kNN_agg + Linear(3,d_h)
geo_tokens   = geo_tokens + pe_3d

# Concat con visuales y texto
encoder_in   # (B, 256+128+text_len, d_h)