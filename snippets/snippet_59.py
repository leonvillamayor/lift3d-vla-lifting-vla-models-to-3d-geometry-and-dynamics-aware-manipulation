# Pseudo-código: encoder Point-MAE como feature extractor para VLA
import torch

class PointMAEBackbone(nn.Module):
    def __init__(self, ckpt="pointmae_pretrained.pth", embed_dim=384):
        super().__init__()
        # Patch embedding tipo PointNet++ (FPS + k-NN)
        self.patch_embed = PointNetPP(k=32, in_dim=3, out_dim=embed_dim)
        # ViT encoder (solo parte visible en pre-train)
        self.encoder = TransformerEncoder(depth=12, dim=embed_dim)
        self.load_pretrained(ckpt)  # congela o fine-tunea

    def forward(self, xyz):  # xyz: (B, N, 3)
        patches, centers = self.patch_embed(xyz)      # tokens visibles
        feats = self.encoder(patches)                 # (B, P, D)
        return feats, centers                         # para fusionar con image tokens