# Pseudocódigo rama 2D
img = cameras[k]                        # (3, H, W) RGB normalizado
feat = vit_encoder(img)                 # (h, w, D_enc) con h=H/14, w=W/14
feat = feat.permute(2,0,1)              # (D_enc, h, w) para gather