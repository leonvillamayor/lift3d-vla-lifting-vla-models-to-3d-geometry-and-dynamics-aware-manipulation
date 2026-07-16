# Pseudo-código: decisión de despliegue basada en el espectro
def choose_vla(task_profile: dict) -> str:
    """
    task_profile = {
        "contact_rich": bool,
        "tool_use": bool,
        "bimanual": bool,
        "tolerance_mm": float,
        "ood_expected": bool
    }
    """
    complexity_score = sum([
        task_profile["contact_rich"],
        task_profile["tool_use"],
        task_profile["bimanual"],
        task_profile["tolerance_mm"] < 5,
        task_profile["ood_expected"]
    ])
    
    if complexity_score <= 1:
        return "2D VLA (latency-optimized)"
    elif complexity_score <= 3:
        return "Lift3D-VLA (3D branch active)"
    else:
        return "Lift3D-VLA + OOD augmentations + GC-MAE retraining"