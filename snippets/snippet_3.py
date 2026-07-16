# Pseudo-código del lifting: punto 3D -> token 2D con geometría
def lift_point(p_xyz, pos_embed_2d):
    # p_xyz: (N, 3) nube de puntos
    # pos_embed_2d: (H, W, D) embeddings posicionales del ViT congelado
    uv = project_to_image_plane(p_xyz)           # (N, 2)
    pe = bilinear_sample(pos_embed_2d, uv)       # (N, D)
    token = pe + geometry_mlp(p_xyz)             # fusiona geometría 3D
    return token  # input al ViT, sin reproyección destructiva