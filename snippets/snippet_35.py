# Pseudo-código: selector de tasks con diversidad garantizada
TASK_FAMILIES = {
    "precise_positioning": ["reach", "push", "pick-place", ...],
    "articulated":         ["drawer-close", "door-open", "window-open", ...],
    "fine_grained":        ["button-press", "dial-turn", "sweep", ...],
}

def select_tasks(min_per_family=4, total=13):
    """Garantiza cobertura balanceada: ningún eje domina el split."""
    selected = []
    for family, tasks in TASK_FAMILIES.items():
        k = min(min_per_family, len(tasks))
        selected.extend(sample(tasks, k))
    # Rellena hasta `total` muestreando de las familias más débiles
    while len(selected) < total:
        family = min(TASK_FAMILIES, key=lambda f: count(selected, f))
        selected.append(sample(TASK_FAMILIES[family], 1)[0])
    return selected