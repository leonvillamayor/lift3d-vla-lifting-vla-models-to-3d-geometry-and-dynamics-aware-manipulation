# Pseudo-bucle de evaluación estilo RLBench keyframe-matching
# (no es código del paper; sirve para razonar el delta)
for task in rlbench_suite:           # 22 tareas
    for variation in range(10):      # instancias por tarea
        for seed in range(3):        # 3 semillas
            obs = env.reset(task, variation, seed)
            while not done:
                # aquí está la diferencia: scene tokens 3D + action chunk
                chunk = policy.act(obs)        # |chunk| = H futuro
                for a in chunk:
                    obs = env.step(a)
                if keyframe_match(obs, target, tol=1*cm):
                    success += 1
# SR = successes / (22 * 10 * 3)