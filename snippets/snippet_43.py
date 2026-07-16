# Lectura accionable: dado presupuesto, qué fila apuntar
budget = "depth_sensor_no_pretraining"   # ej. sin pipeline pre-3D
mapping = {
    "rgb_only":            68.2,
    "2ml_with_cam_params": 81.2,   # <- primer habilitador barato
    "depth_recon":         83.8,
    "static_only":         85.8,
    "dynamic_only":        86.1,
    "full_gc_mae":         88.6,   # <- techo actual
}
target = mapping[budget]
gap_to_ceiling = 88.6 - target