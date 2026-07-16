pe_3d = (pe_2d[cube_face_idx] * visible_mask).sum(1) / visible_mask.sum(1).clamp(min=1)
pe_3d = F.normalize(pe_3d, dim=-1)  # ← invariante a nº de caras visibles