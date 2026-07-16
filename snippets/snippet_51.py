# Pseudo-código: el latente "promedia" cuando no hay instance embedding
def gc_mae_encode(point_cloud):
    # patches de 2cm, sin identidad de instancia
    tokens = voxelize(point_cloud, patch_size=0.02)
    latent = transformer(tokens)  # invariante a permutación
    return latent  # => colapsa al centroide si hay N similares