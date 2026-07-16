# Pseudo-flujo de decisión tras leer refs [24]-[66]
stack_recommendation = {
    "pre_train_dataset":  "[30] Open X-Embodiment + [31] DROID",
    "vision_encoder":     "[37] MAE ViT + PointContrast [35] head",
    "vla_backbone":       "[51] OpenVLA (open weights) o [52] pi0 (flow)",
    "action_head":        "Lift3D (este paper) con dynamics-aware 6-DoF",
    "eval_before_hw":     "[33] MetaWorld + [34] RLBench + [32] RoboMind",
    "future_watch":       "[55] Cosmos Policy, [57] LaST0, [46] CL3R",
}