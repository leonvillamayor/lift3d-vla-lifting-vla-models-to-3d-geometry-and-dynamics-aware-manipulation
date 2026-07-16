"""
Reproduce the Lift3D-VLA data-scaling story:
"Going from 20K -> 140K episodes (7x data) only buys +3.0 pp,
 that's ~0.5 pp per doubling -> the curve is already bending."

Run:  python lift3d_vla_scaling.py
Requires: numpy, matplotlib
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Data points reported / interpolated from the paper-for-humans summary.
#    Endpoint values are taken from the article. Intermediate points are
#    interpolated on a log scale to mimic a saturating curve; they are
#    clearly tagged so the figure is honest about what is measured vs fitted.
# ---------------------------------------------------------------------------
endpoints = np.array(
    [
        (20_000, 85.6),   # reported
        (140_000, 88.6),  # reported
    ],
    dtype=float,
)

# Interpolated points used only to make the visual curve smooth.
interp_episodes = np.array([20_000, 40_000, 70_000, 100_000, 140_000], dtype=float)
interp_success = np.array([85.6, 86.4, 87.4, 88.0, 88.6], dtype=float)  # smooth-ish

# ---------------------------------------------------------------------------
# 2. Fit a power law in log-log space:  success = a * episodes^b
#    We normalize episodes by 1e4 so coefficients stay readable.
# ---------------------------------------------------------------------------
x = np.log10(interp_episodes)
y = np.log10(interp_success)
b, log10_a = np.polyfit(x, y, 1)
a = 10 ** log10_a

print(f"Fitted power law:  success ≈ {a:.4f} * (episodes)^{b:.4f}")

# ---------------------------------------------------------------------------
# 3. Marginal gain per doubling: evaluate the model at N and 2N.
# ---------------------------------------------------------------------------
def success_at(episodes: float) -> float:
    return a * episodes**b


def pp_per_doubling(episodes: float) -> float:
    return success_at(2 * episodes) - success_at(episodes)


for n in [20_000, 40_000, 80_000, 140_000]:
    print(f"  at {n:>7,} eps -> {success_at(n):.2f}%   "
          f"+{pp_per_doubling(n):.2f} pp per 2x")

# ---------------------------------------------------------------------------
# 4. Plot.
# ---------------------------------------------------------------------------
xs = np.logspace(np.log10(15_000), np.log10(200_000), 200)
ys = success_at(xs)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.scatter(endpoints[:, 0], endpoints[:, 1],
           s=90, color="black", zorder=3, label="Reported endpoints")
ax.scatter(interp_episodes, interp_success,
           s=40, color="tab:gray", zorder=2,
           label="Interpolated (illustrative)")
ax.plot(xs, ys, color="tab:blue", lw=2,
        label=fr"Power-law fit: $S \propto N^{{{b:.2f}}}$")

# Annotate the headline finding.
ax.annotate(
    "+3.0 pp for 7× data\n(~0.5 pp / doubling)",
    xy=(140_000, 88.6), xytext=(45_000, 91.5),
    fontsize=10, ha="left",
    arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.2),
    color="tab:red",
)

ax.set_xscale("log")
ax.set_xlabel("Training episodes (log scale)")
ax.set_ylabel("Success rate (%)")
ax.set_title("Lift3D-VLA: data scaling is bending")
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("lift3d_vla_scaling.png", dpi=150)
plt.show()