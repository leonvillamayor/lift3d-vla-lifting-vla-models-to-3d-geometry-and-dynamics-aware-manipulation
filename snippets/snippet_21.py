class PointDecoder(nn.Module):
    def __init__(self, d_in, d_hidden=512, M=8192):
        self.proj = nn.Linear(d_in, d_hidden)
        self.up = nn.Sequential(*[ResBlock(d_hidden) for _ in range(3)])
        self.head = nn.Linear(d_hidden, 3)  # dx, dy, dz por punto
        self.M = M
    
    def forward(self, z):                    # (N_pts, d_in)
        h = self.up(self.proj(z))            # (N_pts, d_hidden)
        offsets = self.head(h)               # (N_pts, 3)
        # upsampling: cada punto genera M/N hijos con offset pequeño
        return points_curr[:, :3] + offsets  # predicción residual