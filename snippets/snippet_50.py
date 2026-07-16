def misalignment_risk(bottle_tip_xyz, beaker_rim_center, beaker_radius, tip_radius):
    lateral = np.linalg.norm(bottle_tip_xyz[:2] - beaker_rim_center[:2])
    # projection of tip into rim plane; >0 means off-axis
    return lateral - (beaker_radius - tip_radius)