class DualBranchDecoder(nn.Module):
    def __init__(self, d_model, n_layers=4, n_heads=8):
        super().__init__()
        #rama desacoplada y simétrica
        self.static  = TransformerStack(d_model, n_layers, n_heads)
        self.dynamic = TransformerStack(d_model, n_layers, n_heads)
        #proyecciones finales por Eq. 8
        self.head_static  = nn.Linear(d_model, 3)   #regresión geométrica
        self.head_dynamic = nn.Linear(d_model, 3)   #flujo / delta-t

    def forward(self, pe_3d):
        s = self.static(pe_3d)
        d = self.dynamic(pe_3d)
        loss = self.aux_loss(s, d, gt)  #Eq. 8
        return loss