# Pseudocódigo de validación OOD pre-deployment (~30 líneas)
def ood_gate(obs_batch, encoder, pe3d_stats):
    pe_3d = encoder(obs_batch)["pe_3d"]          # (B, N_faces, D)
    norm  = pe_3d.norm(dim=-1).mean()
    if abs(norm - pe3d_stats["mean"]) > 3 * pe3d_stats["std"]:
        return {"gate": "fail", "reason": "pe_3d_drift"}
    if pe_3d.isnan().any():
        return {"gate": "fail", "reason": "nan_in_pe"}
    return {"gate": "pass"}