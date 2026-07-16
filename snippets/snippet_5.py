# Pseudocódigo del alineamiento
def align_3d_to_2d_pe(point_features, xyz_3d, grid_size, llm_pe_table):
    # 1. Proyectar xyz 3D → (u, v) en grid 2D
    uv = project_to_2d_grid(xyz_3d, grid_size)  # (N, 2)

    # 2. Indexar PEs 2D preentrenados del LLM congelado
    pe_2d = llm_pe_table[uv[:, 0], uv[:, 1]]  # (N, d_model)

    # 3. Sumar al contenido (¡sin entrenar PEs!)
    scene_tokens_3d = point_features + pe_2d  # frozen PE lookup
    return scene_tokens_3d