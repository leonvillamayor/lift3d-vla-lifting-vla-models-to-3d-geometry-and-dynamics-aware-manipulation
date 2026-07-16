# Conceptual: cómo se ve el lifting
points_3d = sample_point_cloud(depth_image, K, T)  # K=intrinsics, T=extrinsics
virtual_planes = project_to_virtual_planes(points_3d, anchor='camera_view')
# Cada plano respeta los PEs 2D que el ViT ya aprendió
tokens_3d = align_with_2d_pe(virtual_planes, pretrained_pe_grid)
features = vit_2d_encoder(tokens_3d)  # pesos congelados o fine-tuned