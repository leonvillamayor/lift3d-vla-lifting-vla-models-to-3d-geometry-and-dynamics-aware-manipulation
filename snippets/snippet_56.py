keypose = chained_diffuser.predict_keypose(rgb_pc, instruction)   # ref [19]
action  = lift3d_vla(rgb_pc, instr, target_ee=keypose)           # Lift3D