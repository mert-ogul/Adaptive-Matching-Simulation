"""Dispatch policies for assigning tasks to experts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from .config import Algorithm
from .models import Expert, SystemState, Task
from .probabilities import build_asymmetric_matrix, failure_prob, update_belief


class Policy(Protocol):
    """Strategy interface for dispatching tasks to experts."""

    def select_assignment(
        self,
        state: SystemState,
        rng: np.random.Generator,
    ) -> tuple[Expert, Task] | None:
        """Return an (expert, task) pair to assign, or None if no assignment is possible."""


@dataclass(slots=True)
class RandomPolicy:
    """Random policy: choose an idle expert and a task uniformly at random."""

    def select_assignment(
        self,
        state: SystemState,
        rng: np.random.Generator,
    ) -> tuple[Expert, Task] | None:
        idle_experts = [e for e in state.experts if not e.busy]
        if not idle_experts or not state.task_pool:
            return None

        expert_idx = int(rng.integers(0, len(idle_experts)))
        task_idx = int(rng.integers(0, len(state.task_pool)))

        expert = idle_experts[expert_idx]
        task_id = state.task_pool[task_idx]
        task = state.tasks[task_id]
        return expert, task


def create_random_policy() -> Policy:
    """Factory for the Random policy."""

    return RandomPolicy()


@dataclass(slots=True)
class BackpressurePolicy:
    """Backpressure(ε) policy over discretized belief bins."""

    p_matrix: npt.NDArray[np.float64]
    eps: float
    _n_bins: int = field(init=False, repr=False)
    _weight_tol: float = field(default=1e-12, init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.eps <= 1.0):
            raise ValueError("eps must be in (0, 1]")
        self._n_bins = int(np.ceil(1.0 / self.eps))

    def _bin_of_belief(self, belief: npt.NDArray[np.float64]) -> tuple[int, int]:
        z_c1 = float(np.clip(belief[0], 0.0, 1.0 - 1e-15))
        z_c2 = float(np.clip(belief[1], 0.0, 1.0 - 1e-15))
        i = min(max(int(z_c1 / self.eps), 0), self._n_bins - 1)
        j = min(max(int(z_c2 / self.eps), 0), self._n_bins - 1)
        return i, j

    def _build_bin_counts(self, state: SystemState) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for task_id in state.task_pool:
            task = state.tasks[task_id]
            bin_index = self._bin_of_belief(task.belief)
            counts[bin_index] = counts.get(bin_index, 0) + 1
        return counts

    def _backpressure_weight(
        self,
        task: Task,
        expert_index: int,
        counts: dict[tuple[int, int], int],
    ) -> float:
        current_bin = self._bin_of_belief(task.belief)
        n_current = counts.get(current_bin, 0)
        psi = failure_prob(self.p_matrix, expert_index, task.belief)
        belief_after_failure = update_belief(self.p_matrix, expert_index, task.belief)
        next_bin = self._bin_of_belief(belief_after_failure)
        n_next = counts.get(next_bin, 0)
        return n_current - psi * n_next

    def select_assignment(
        self,
        state: SystemState,
        rng: np.random.Generator,
    ) -> tuple[Expert, Task] | None:
        idle_experts = [e for e in state.experts if not e.busy]
        if not idle_experts or not state.task_pool:
            return None

        expert_idx = int(rng.integers(0, len(idle_experts)))
        expert = idle_experts[expert_idx]

        counts = self._build_bin_counts(state)
        best_weight = -np.inf
        best_task_ids: list[int] = []

        for task_id in state.task_pool:
            task = state.tasks[task_id]
            weight = self._backpressure_weight(task, expert.index, counts)
            if weight > best_weight + self._weight_tol:
                best_weight = weight
                best_task_ids = [task_id]
            elif abs(weight - best_weight) <= self._weight_tol:
                best_task_ids.append(task_id)

        if not best_task_ids:
            return None
        selected_idx = int(rng.integers(0, len(best_task_ids)))
        selected_task = state.tasks[best_task_ids[selected_idx]]
        return expert, selected_task


def create_backpressure_policy(a: float, delta: float, eps: float) -> Policy:
    """Factory for the Backpressure(ε) policy."""

    p_matrix = build_asymmetric_matrix(a, delta)
    return BackpressurePolicy(p_matrix=p_matrix, eps=eps)


def create_policy(
    algorithm: Algorithm,
    a: float,
    delta: float,
    eps: float | None = None,
) -> Policy:
    """Factory for selecting a policy by algorithm name."""

    if algorithm == "random":
        return create_random_policy()
    if algorithm == "backpressure":
        if eps is None:
            raise ValueError("eps must be provided for backpressure policy")
        return create_backpressure_policy(a=a, delta=delta, eps=eps)
    raise ValueError(f"Unsupported algorithm: {algorithm}")
