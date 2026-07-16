"""
Inaccurate Grasp Position failure mode in VLA manipulation.

Scenario from Lift3D-VLA taxonomy (sensing-side):
Two identical fruits sit 2-3 cm apart. A 2D-image-based policy
predicts a grasp point that falls *between* them (a "ghost point")
because it averages over two visually similar objects.

This script:
  1. Builds a tiny synthetic 3D scene with two identical fruit point clouds.
  2. Simulates a 2D-image baseline that collapses the depth axis and
     therefore outputs the mid-point between the two centroids.
  3. Simulates a 3D-geometry-aware policy (Lift3D-VLA style) that
     treats each fruit as an explicit instance and picks a real one.
  4. Computes the grasp-position error against both ground-truth centroids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


# ---------- Scene ---------------------------------------------------------

@dataclass(frozen=True)
class Fruit:
    """A rigid object defined by a point cloud in the camera frame."""
    name: str
    points: np.ndarray  # shape (N, 3), x right, y down, z forward

    @property
    def centroid(self) -> np.ndarray:
        return self.points.mean(axis=0)


def build_scene(distance_cm: float = 2.5, n_points: int = 200, seed: int = 0) -> list[Fruit]:
    """Two identical spherical fruits, separated along the x-axis."""
    rng = np.random.default_rng(seed)
    radius = 0.02  # 2 cm sphere radius

    def sphere(centre: np.ndarray) -> np.ndarray:
        # Uniform points inside a ball (rejection sampling keeps it simple).
        pts = rng.uniform(-1.0, 1.0, size=(n_points * 2, 3))
        mask = np.linalg.norm(pts, axis=1) <= 1.0
        pts = pts[mask][:n_points]
        return pts * radius + centre

    offset = distance_cm / 100.0 / 2.0
    fruit_a = Fruit("apple_A", sphere(np.array([-offset, 0.0, 0.50])))
    fruit_b = Fruit("apple_B", sphere(np.array([+offset, 0.0, 0.50])))
    return [fruit_a, fruit_b]


# ---------- Policies ------------------------------------------------------

@dataclass
class GraspPrediction:
    position: np.ndarray  # (x, y, z) in metres
    confidence: float
    target_name: str | None  # None means "ambiguous / ghost"


def baseline_2d_policy(fruits: Iterable[Fruit], image_h: int = 64, image_w: int = 64) -> GraspPrediction:
    """Collapse depth, project to a 2D pixel, then unproject with the mean z.

    This mimics image-only VLA policies that ignore per-instance 3D structure.
    Two identical fruits therefore fuse into a single visual blob whose centre
    is the pixel midpoint; unprojecting it yields the 'ghost point'.
    """
    fruits = list(fruits)
    centroids = np.stack([f.centroid for f in fruits])  # (K, 3)
    mean_z = float(centroids[:, 2].mean())

    # Image-space average of the centroids' x, y components.
    image_xy = centroids[:, :2].mean(axis=0)
    ghost = np.array([image_xy[0], image_xy[1], mean_z])

    # Confidence is high because the visual signal is unambiguous to the model.
    return GraspPrediction(position=ghost, confidence=0.91, target_name=None)


def lift3d_vla_style_policy(fruits: Iterable[Fruit]) -> GraspPrediction:
    """3D-geometry- and instance-aware policy.

    Lift3D-VLA lifts 2D features into an explicit 3D volumetric / point
    representation, so identical-looking objects remain *distinct instances*
    in 3D space. The policy can therefore select one fruit, grasp its true
    centroid, and report which one it picked.
    """
    fruits = list(fruits)
    # A real model would score each instance; here we pick the leftmost for
    # determinism — what matters is that the pick is one of the real fruits.
    chosen = min(fruits, key=lambda f: f.centroid[0])
    return GraspPrediction(position=chosen.centroid, confidence=0.83, target_name=chosen.name)


# ---------- Evaluation ----------------------------------------------------

def grasp_error(predicted: np.ndarray, candidates: list[np.ndarray]) -> tuple[float, float]:
    """Distance from the predicted grasp to the nearest / farthest fruit centroid."""
    dists = [float(np.linalg.norm(predicted - c)) for c in candidates]
    return min(dists), max(dists)


def main() -> None:
    scene = build_scene(distance_cm=2.5)
    centroids = [f.centroid for f in scene]

    baseline = baseline_2d_policy(scene)
    lift3d = lift3d_vla_style_policy(scene)

    print(f"Fruit centroids (m): {centroids[0]}\n                  {centroids[1]}")
    print(f"True separation   : {np.linalg.norm(centroids[0] - centroids[1]) * 100:.2f} cm\n")

    for label, pred in [("2D baseline (ghost)", baseline), ("Lift3D-VLA (3D-aware)", lift3d)]:
        near, far = grasp_error(pred.position, centroids)
        verdict = "lands on a real fruit" if pred.target_name else "GHOST POINT — between both fruits"
        print(f"{label:>22s}: pos={pred.position} | near={near*100:.2f} cm | {verdict}")


if __name__ == "__main__":
    main()