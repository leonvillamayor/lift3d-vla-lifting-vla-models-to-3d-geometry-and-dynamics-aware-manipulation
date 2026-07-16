# Lo que un VLA 2D "ve" vs lo que necesita "razonar"
vision_2d_features = vla_encoder(rgb_image)      # shape: (B, N_patches, D)
action = vla_head(vision_2d_features, instruction)  # shape: (B, T_chunk, action_dim)
# Problema: vision_2d_features NO codifica volumen, profundidad métrica,
# ni dinámica de fluidos. La acción sale por extrapolación estadística.