# Para cada píxel (u,v):
p_cam = (u - cx) * depth / fx, (v - cy) * depth / fy, depth
p_world = R^T @ (p_cam - t)              # extrínseca rígida
semantic_feat = vit_encoder_grid(u, v)   # (D_enc,)
# → punto 3D con payload semántico
points_curr = torch.cat([p_world, semantic_feat], dim=-1)  # (N_pts, 3 + D_enc)