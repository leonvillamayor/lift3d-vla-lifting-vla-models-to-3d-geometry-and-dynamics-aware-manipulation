import numpy as np

def fuse_then_sample(depths, poses, K, scene_bounds, n_points=8192, min_views=2):
    """
    depths: list[H,W] de mapas de profundidad métrica (m)
    poses:  list[4,4] cámara→mundo
    K:      [3,3] intrínsecos
    """
    pc_views = []
    for d, T in zip(depths, poses):
        h, w = d.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = d
        x = (u - K[0,2]) * z / K[0,0]
        y = (v - K[1,2]) * z / K[1,1]
        pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)  # H,W,4
        pts_world = (T @ pts_cam.reshape(-1, 4).T).T[:, :3]
        pc_views.append(pts_world)

    # fusion por voxel hashing, conserva vistas mínimas
    fused = hash_voxel_aggregate(pc_views, voxel=0.005, min_views=min_views)
    mask = (fused[:,0] >= scene_bounds[0,0]) & (fused[:,0] <= scene_bounds[0,1])
    fused = fused[mask]
    return farthest_point_sample(fused, n_points)