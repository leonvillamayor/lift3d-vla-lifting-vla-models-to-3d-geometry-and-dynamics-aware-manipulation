# Pseudo-código de triage post-Stage-2
def diagnose_failure(task, rollout):
    if task.is_precision_heavy():   # pick place, insert
        inspect(rollout.pe_3d_error, rollout.voxel_resolution)
    elif task.is_articulated():      # drawer, box close
        inspect(rollout.dynamics_lambda, rollout.warmup_steps)
    elif task.is_fine_grained():     # water plants, deformable
        return "ir a Stage-3 antes de tocar el lifting"
    else:
        inspect(rollout.execution_quality_vs_count)