# Pseudocódigo de la regla de despliegue
def action_head_budget(num_layers, strides):
    # Parámetros del cabezal ∝ num_layers (transformer compartido por cabezal)
    # Coste temporal por step ∝ sum(1/s for s in strides) * num_layers
    # (stride s = atiende cada s tokens → coste 1/s)
    temporal_cost = sum(1.0 / s for s in strides) * num_layers
    return {
        "params": num_layers * D_action**2,   # D_action = dim oculto del cabezal
        "temporal_ops_per_step": temporal_cost,
    }

configs = [
    (1, [1]),                       # baseline barato
    (2, [1]),                       # +1.8 pp, coste ~2×
    (2, [1, 2, 4]),                 # +3.3 pp sobre baseline, coste ~2.5×
    (4, [1, 2, 4]),                 # +3.8 pp, coste ~4.4×
    (4, [1, 2, 4, 8]),              # +5.2 pp, coste ~5.8×  ← default paper
]