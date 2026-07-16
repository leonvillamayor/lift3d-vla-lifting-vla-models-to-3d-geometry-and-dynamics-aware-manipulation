# Pseudocódigo de un forward GC-MAE simplificado
visible_pts, masked_pts = sample_visible_masked(point_cloud, mask_ratio=0.6)
geom_recon = decoder_recon(encoder(visible_pts), masked_tokens)
fut_pred  = decoder_dyn(encoder(visible_pts))  # predice t+1 sobre lo visible
loss = alpha * chamfer(geom_recon, masked_pts) + beta * flow_loss(fut_pred, future_pts)