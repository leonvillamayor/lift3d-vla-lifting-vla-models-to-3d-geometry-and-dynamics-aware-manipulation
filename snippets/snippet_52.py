# Detector: dos modos separados → instancia ambigua
def grasp_ambiguity_check(pose_history, point_cloud, fruits):
    candidates = [project_to_3d(p) for p in pose_history[-10:]]
    clusters = dbscan(candidates, eps=0.015)
    if len(clusters) >= 2:
        for c in clusters:
            if any(dist(c.centroid, f.center) < 0.02 for f in fruits):
                return "INSTANCE_AMBIGUOUS", clusters
    return "OK", None