# Stage 1: supervision limpia, sin ruido geométrico
batch = loader_2d.sample()           # RGB + acción GT
z_vlm = vlm.encode_vision(batch.rgb)  # pe_2d como ya vimos
loss = ddpm_action_loss(head(z_vlm), batch.action_gt)
loss.backward()