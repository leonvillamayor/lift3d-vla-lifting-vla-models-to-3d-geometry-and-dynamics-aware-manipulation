# pseudocódigo conceptual
def project_to_cube_faces(points_xyz, intrinsics=None, extrinsics=None):
    # 6 caras: +X, -X, +Y, -Y, +Z, -Z
    # cada cara: proyección perspectiva simple sobre el plano local
    return coords_uv  # (B, N_pc, 6, 2)