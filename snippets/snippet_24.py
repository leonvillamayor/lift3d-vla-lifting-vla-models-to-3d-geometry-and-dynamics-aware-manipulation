# pseudo-shapes
pe_3d = torch.zeros(B, N_pc, d_h)
for j in range(6):
    u, v = project(p_xyz, plane=j)            # (B, N_pc)
    pe_3d += pe_table[j][:, u, v, :]          # (B, N_pc, d_h)
pe_3d /= 6.0                                  # mean over views