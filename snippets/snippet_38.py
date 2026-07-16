"""
Multi-task evaluation harness for Lift3D-VLA style experiments
on MetaWorld (10 tasks) and RLBench (9 tasks).

This is illustrative code showing the evaluation structure typically
used in VLA (Vision-Language-Action) papers when comparing against
other VLA baselines like OpenVLA, RT-2, 3D-LOTUS, etc.
"""

from __future__ import annotations

import dataclasses
import logging
import statistics
import time
from collections.abc import Callable, Sequence
from typing import Protocol, TypeAlias

import numpy as np

# Standard robotics env API (MetaWorld builds on top of this).
try:
    import gymnasium as gym
    import metaworld  # noqa: F401  -- registers MT1/MT10 envs
except ImportError:
    gym = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ----------------------------- Types --------------------------------------

ObsType: TypeAlias = dict[str, np.ndarray]
ActionType: TypeAlias = np.ndarray
TaskName: TypeAlias = str

# The 10 MetaWorld tasks commonly used in multi-task VLA evaluations.
METAWORLD_TASKS: tuple[TaskName, ...] = (
    "reach-v3",
    "push-v3",
    "pick-place-v3",
    "door-open-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "peg-insert-side-v3",
    "window-open-v3",
    "sweep-v3",
    "basketball-v3",
)

# The 9 RLBench tasks commonly reported in VLA papers.
RLBENCH_TASKS: tuple[TaskName, ...] = (
    "pick_and_lift",
    "stack_blocks",
    "put_in_drawer",
    "close_drawer",
    "turn_tap",
    "push_button",
    "open_box",
    "place_cup",
    "sweep_to_dustpan",
)


class VLA(Protocol):
    """Protocol any VLA policy (Lift3D-VLA, OpenVLA, RT-2, ...) satisfies."""

    def predict(
        self,
        observation: ObsType,
        instruction: str,
    ) -> ActionType: ...


# ----------------------------- Harness ------------------------------------

@dataclasses.dataclass(frozen=True, slots=True)
class TaskResult:
    task: TaskName
    success_rate: float
    avg_episode_length: float
    num_episodes: int


@dataclasses.dataclass(frozen=True, slots=True)
class BenchmarkReport:
    benchmark: str
    per_task: list[TaskResult]
    overall_success_rate: float

    def summary_table(self) -> str:
        rows = ["task | success | ep_len", "-" * 40]
        for t in self.per_task:
            rows.append(
                f"{t.task:<24} | {t.success_rate:6.1%} | {t.avg_episode_length:5.1f}"
            )
        rows.append("-" * 40)
        rows.append(f"OVERALL: {self.overall_success_rate:6.1%}")
        return "\n".join(rows)


def _language_instruction_mw(task: TaskName) -> str:
    # MetaWorld has no built-in natural-language instruction; we follow the
    # common practice in VLA papers of templating one from the task name.
    verb, obj = task.replace("-v3", "").rsplit("-", 1)
    return f"{verb.replace('-', ' ')} the {obj.replace('-', ' ')}."


def evaluate_mt10_metaworld(
    policy: VLA,
    *,
    num_episodes_per_task: int = 25,
    max_steps: int = 200,
    seed: int = 0,
    instruction_fn: Callable[[TaskName], str] = _language_instruction_mw,
) -> BenchmarkReport:
    """Run the standard MT10 evaluation on MetaWorld for a VLA policy."""
    if gym is None:
        raise RuntimeError(
            "Install with: pip install gymnasium metaworld"
        )
    rng = np.random.default_rng(seed)
    per_task: list[TaskResult] = []

    for task in METAWORLD_TASKS:
        env = gym.make("Meta-World-v3", env_name=task, seed=int(rng.integers(0, 2**31 - 1)))
        successes: list[bool] = []
        ep_lengths: list[int] = []

        for ep in range(num_episodes_per_task):
            obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            instr = instruction_fn(task)
            for step in range(max_steps):
                action = policy.predict(obs, instr)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, _reward, terminated, truncated, info = env.step(action)
                if bool(info.get("success", terminated)):
                    successes.append(True)
                    ep_lengths.append(step + 1)
                    break
            else:
                successes.append(False)
                ep_lengths.append(max_steps)

        env.close()
        per_task.append(TaskResult(
            task=task,
            success_rate=sum(successes) / len(successes),
            avg_episode_length=statistics.mean(ep_lengths),
            num_episodes=num_episodes_per_task,
        ))
        logger.info("MT10 %-24s  success=%.1f%%",
                    task, 100 * per_task[-1].success_rate)

    overall = statistics.mean(t.success_rate for t in per_task)
    return BenchmarkReport("MetaWorld-MT10", per_task, overall)


def evaluate_rlbench_subset(
    policy: VLA,
    *,
    num_episodes_per_task: int = 25,
    max_episodes: int = 1,
    instruction_fn: Callable[[TaskName], str] | None = None,
) -> BenchmarkReport:
    """
    Stub harness mirroring the structure authors use for RLBench.

    RLBench requires the CoppeliaSim simulator; this function illustrates
    the evaluation contract without forcing the Coppelia dependency.
    """
    if instruction_fn is None:
        instruction_fn = lambda t: t.replace("_", " ") + "."  # noqa: E731

    # In practice: from rlbench.environment import Environment; from rlbench.tasks import ...
    per_task: list[TaskResult] = [
        TaskResult(
            task=t,
            success_rate=0.0,    # populated by the real environment below
            avg_episode_length=0.0,
            num_episodes=num_episodes_per_task,
        )
        for t in RLBENCH_TASKS
    ]
    overall = 0.0  # placeholder
    return BenchmarkReport("RLBench-subset9", per_task, overall)


def compare_vla_policies(
    policies: Sequence[tuple[str, VLA]],
    *,
    num_episodes_per_task: int = 25,
    seed: int = 0,
) -> None:
    """Side-by-side comparison harness for VLA models on MetaWorld-MT10."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== VLA comparison run @ {timestamp} ===\n")

    reports: list[tuple[str, BenchmarkReport]] = []
    for name, policy in policies:
        print(f"\n--- Evaluating: {name} ---")
        rep = evaluate_mt10_metaworld(
            policy,
            num_episodes_per_task=num_episodes_per_task,
            seed=seed,
        )
        print(rep.summary_table())
        reports.append((name, rep))

    # Pretty side-by-side
    header = f"{'Task':<24}" + "".join(f"{n:>10}" for n, _ in reports)
    print("\n" + header)
    print("-" * len(header))
    for i, task in enumerate(METAWORLD_TASKS):
        row = f"{task:<24}"
        for _, rep in reports:
            row += f"{rep.per_task[i].success_rate * 100:>9.1f}%"
        print(row)
    overall_row = f"{'OVERALL':<24}"
    for _, rep in reports:
        overall_row += f"{rep.overall_success_rate * 100:>9.1f}%"
    print(overall_row)


# ----------------------------- Demo policies ------------------------------

class Lift3DVLA_Stub:
    """Minimal stub satisfying the VLA protocol for the example to run."""

    def __init__(self, success_bias: float = 0.7) -> None:
        self._bias = success_bias
        self._rng = np.random.default_rng()

    def predict(self, observation: ObsType, instruction: str) -> ActionType:
        # A real Lift3D-VLA policy uses 3D point-cloud + text + robot state
        # through a transformer to predict 7-DoF actions.
        return self._rng.standard_normal(4) * 0.1


class OpenVLA_Stub:
    def __init__(self, success_bias: float = 0.55) -> None:
        self._bias = success_bias
        self._rng = np.random.default_rng()

    def predict(self, observation: ObsType, instruction: str) -> ActionType:
        return self._rng.standard_normal(4) * 0.1


# ----------------------------- Main ---------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    policies: list[tuple[str, VLA]] = [
        ("Lift3D-VLA (ours)", Lift3DVLA_Stub(success_bias=0.78)),
        ("OpenVLA-7B",        OpenVLA_Stub(success_bias=0.62)),
    ]
    compare_vla_policies(policies, num_episodes_per_task=10, seed=42)