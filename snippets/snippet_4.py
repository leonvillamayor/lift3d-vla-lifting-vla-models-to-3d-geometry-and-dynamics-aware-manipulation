# pseudocódigo de integración (estilo OpenVLA-style)
patch_feats = encoder.patch_embed(rgb, depth)              # (B, N_p, D)
xyz, n = encoder.lift(depth, K)                            # (B, N_p, 3)
scene_tokens = encoder.cross_modality(patch_feats, xyz, n) # (B, N_s, D_s)
scene_tokens = dynamics_adapter(scene_tokens, prev_frames)
prefix = torch.cat([instr_emb, scene_tokens, hist_emb], dim=1)
action = vla_llm(prefix, action_query)                     # VLA congelado