# pseudo-uso: reemplazar el bloque "MAE[37] + PointContrast[35]" por VGGT
# cuando el hw no tenga depth/Estéreo confiable.
vggt_feats = VGGT.from_pretrained("facebook/vggt-base").encode(rgb_stack)  # (B, N, D)