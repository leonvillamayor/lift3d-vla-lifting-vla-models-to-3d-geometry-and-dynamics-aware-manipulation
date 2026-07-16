import numpy as np
from scipy.optimize import curve_fit

def power_law_saturating(K, a, b, c):
    # a: asíntota superior, b: escala, c: exponente de aprendizaje
    return a - b * np.power(K, -c)

# Tres puntos mínimo para estabilizar el fit
K = np.array([20_000, 50_000, 100_000, 140_000])
acc = np.array([85.6, 86.8, 88.2, 88.6])  # los 2 intermedios son estimados

popt, _ = curve_fit(power_law_saturating, K, acc, p0=[90, 50, 0.2])
a_hat, b_hat, c_hat = popt
print(f"Asíntota estimada: {a_hat:.2f}%")
print(f"Para ganar +0.5 pp extra necesitas ~{estimate_extra_data(K, acc, target=0.5):.0f}K episodios")