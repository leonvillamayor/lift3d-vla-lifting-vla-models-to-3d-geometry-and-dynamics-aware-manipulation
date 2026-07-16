# Pseudo-decisión de sensor stack al kickoff
def choose_observation_space(task_profile: dict) -> str:
    if task_profile["geometric_precision_demand"] >= 0.7 \
       or task_profile["clutter_level"] in {"dense", "bin_packing"}:
        return "pointcloud"          # IPC + Depth → PCD
    if task_profile["needs_semantic_grounding"]:
        return "rgb"                 # VLA web-scale baseline
    return "rgb+depth_fusion"        # Lift3D-VLA style