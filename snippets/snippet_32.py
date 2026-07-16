for epoch in range(15):
    for batch in synth_loader:           # batch_size=4096, N=1024
        pts = batch["pointcloud"]        # (B, 1024, 3)
        mask = rand_mask(pts, ratio=0.6) # MAE-style
        recon, motion_pred = mae_enc_dec(pts, mask)
        L_static  = chamfer(recon[mask], pts[mask])
        L_dynamic = flow_loss(motion_pred, batch["delta_pts"])
        loss = L_static + lam * L_dynamic
        loss.backward()
        adamw.step(); sched.step()