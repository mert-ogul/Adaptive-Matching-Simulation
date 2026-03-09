"""Discrete-event simulator for the Asymmetric(a, δ) system."""

from __future__ import annotations

import heapq
from typing import Any

import numpy as np

from .config import Algorithm, SimulationConfig
from .models import Event, EventType, Expert, SystemState, Task, TaskType
from .policies import Policy, create_policy
from .probabilities import build_asymmetric_matrix, update_belief
from .stats import SimulationResult, aggregate_replications, compute_single_run_metrics


class Simulator:
    """Event-driven simulator for the expert system."""

    def __init__(self, config: SimulationConfig, policy: Policy) -> None:
        self.config = config
        self.policy = policy
        self.rng = np.random.default_rng(config.seed)
        self.p_matrix = build_asymmetric_matrix(config.a, config.delta)

        experts = [
            Expert(expert_id="s1", index=0),
            Expert(expert_id="s2", index=1),
        ]

        self.state = SystemState(
            time=0.0,
            lam=config.lam,
            experts=experts,
            tasks={},
            task_pool=[],
            completed_task_ids=[],
            event_queue=[],
            rng=self.rng,
        )
        self._next_task_id = 0
        self._next_event_seq = 0

        first_arrival = self._sample_interarrival()
        self._schedule_arrival(first_arrival)

    # Sampling helpers -----------------------------------------------------
    def _sample_interarrival(self) -> float:
        return float(self.rng.exponential(1.0 / self.config.lam))

    def _sample_service_time(self) -> float:
        return float(self.rng.exponential(1.0))

    # Event scheduling -----------------------------------------------------
    def _push_event(self, event: Event) -> None:
        heapq.heappush(self.state.event_queue, (event.time, event.seq, event))

    def _schedule_arrival(self, time: float) -> None:
        if time > self.config.horizon:
            return
        event = Event(
            time=time,
            seq=self._next_event_seq,
            event_type=EventType.ARRIVAL,
        )
        self._next_event_seq += 1
        self._push_event(event)

    def _schedule_service_completion(self, expert_index: int, task_id: int, time: float) -> None:
        if time > self.config.horizon:
            return
        event = Event(
            time=time,
            seq=self._next_event_seq,
            event_type=EventType.SERVICE_COMPLETION,
            expert_index=expert_index,
            task_id=task_id,
        )
        self._next_event_seq += 1
        self._push_event(event)

    # Event handlers -------------------------------------------------------
    def _handle_arrival(self, event: Event) -> None:
        self.state.time = event.time
        task_id = self._next_task_id
        self._next_task_id += 1

        true_type: TaskType = "c1" if self.rng.random() < 0.5 else "c2"
        belief = np.array([0.5, 0.5], dtype=float)
        task = Task(
            task_id=task_id,
            true_type=true_type,
            arrival_time=self.state.time,
            belief=belief,
        )
        self.state.tasks[task_id] = task
        self.state.task_pool.append(task_id)

        # Schedule next arrival
        next_time = self.state.time + self._sample_interarrival()
        self._schedule_arrival(next_time)

        self._dispatch()

    def _handle_service_completion(self, event: Event) -> None:
        assert event.expert_index is not None
        assert event.task_id is not None

        self.state.time = event.time
        expert = self.state.experts[event.expert_index]
        task = self.state.tasks[event.task_id]

        # Update expert busy statistics
        if expert.last_start_busy is not None:
            expert.busy_time += self.state.time - expert.last_start_busy
        expert.busy = False
        expert.current_task_id = None
        expert.last_start_busy = None

        # Determine success or failure based on true type
        type_index = 0 if task.true_type == "c1" else 1
        success_prob = self.p_matrix[expert.index, type_index]
        success = bool(self.rng.random() < success_prob)

        task.attempts += 1
        task.attempt_history.append((expert.expert_id, success, self.state.time))

        if success:
            task.departure_time = self.state.time
            self.state.completed_task_ids.append(task.task_id)
        else:
            # Belief update and return to pool
            task.belief = update_belief(self.p_matrix, expert.index, task.belief)
            self.state.task_pool.append(task.task_id)

        self._dispatch()

    # Dispatch loop --------------------------------------------------------
    def _dispatch(self) -> None:
        while True:
            idle_experts = [e for e in self.state.experts if not e.busy]
            if not idle_experts or not self.state.task_pool:
                return

            assignment = self.policy.select_assignment(self.state, self.rng)
            if assignment is None:
                return

            expert, task = assignment
            if expert.busy or task.task_id not in self.state.task_pool:
                # Policy returned an invalid assignment; stop dispatching to avoid looping.
                return

            # Remove the task from the pool
            self.state.task_pool.remove(task.task_id)

            # Mark expert as busy and schedule completion
            expert.busy = True
            expert.current_task_id = task.task_id
            expert.last_start_busy = self.state.time

            completion_time = self.state.time + self._sample_service_time()
            self._schedule_service_completion(expert.index, task.task_id, completion_time)

    # Main loop ------------------------------------------------------------
    def run(self) -> SimulationResult:
        while self.state.event_queue:
            time, _seq, event = heapq.heappop(self.state.event_queue)
            if time > self.config.horizon:
                break

            if event.event_type == EventType.ARRIVAL:
                self._handle_arrival(event)
            elif event.event_type == EventType.SERVICE_COMPLETION:
                self._handle_service_completion(event)
            else:
                raise RuntimeError(f"Unknown event type: {event.event_type}")

        # Advance time to horizon for utilization calculations
        self.state.time = self.config.horizon
        # If any expert is still busy, account for remaining busy time up to horizon
        for expert in self.state.experts:
            if expert.busy and expert.last_start_busy is not None:
                expert.busy_time += self.config.horizon - expert.last_start_busy
                expert.last_start_busy = self.config.horizon

        return SimulationResult(
            config=self.config,
            tasks=list(self.state.tasks.values()),
            experts=self.state.experts,
            horizon=self.config.horizon,
        )


# Public API ---------------------------------------------------------------
def run_simulation(
    lam: float,
    a: float,
    delta: float,
    horizon: float,
    warmup: float,
    seed: int,
    algorithm: Algorithm = "random",
    eps: float | None = None,
) -> dict[str, Any]:
    """Run a single simulation and return aggregate metrics."""
    from .config import SimulationConfig

    config = SimulationConfig(
        lam=lam,
        a=a,
        delta=delta,
        horizon=horizon,
        warmup=warmup,
        seed=seed,
        algorithm=algorithm,
        eps=eps,
    )
    policy = create_policy(config.algorithm, config.a, config.delta, config.eps)
    simulator = Simulator(config, policy)
    result = simulator.run()
    metrics = compute_single_run_metrics(result, warmup=config.warmup)
    return metrics


def run_replications(
    lam: float,
    a: float,
    delta: float,
    horizon: float,
    warmup: float,
    n_reps: int,
    base_seed: int,
    algorithm: Algorithm = "random",
    eps: float | None = None,
) -> dict[str, Any]:
    """Run multiple independent replications and return CI summaries."""
    rep_metrics: list[dict[str, Any]] = []
    for i in range(n_reps):
        seed = base_seed + i
        config = SimulationConfig(
            lam=lam,
            a=a,
            delta=delta,
            horizon=horizon,
            warmup=warmup,
            seed=seed,
            algorithm=algorithm,
            eps=eps,
        )
        policy = create_policy(config.algorithm, config.a, config.delta, config.eps)
        simulator = Simulator(config, policy)
        result = simulator.run()
        metrics = compute_single_run_metrics(result, warmup=config.warmup)
        rep_metrics.append(metrics)

    aggregated = aggregate_replications(rep_metrics)
    aggregated["replication_metrics"] = rep_metrics
    return aggregated
