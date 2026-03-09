import numpy as np
import pytest

from expert_sim.models import Expert, SystemState, Task
from expert_sim.policies import RandomPolicy, create_backpressure_policy, create_policy


def _make_state(num_tasks: int, seed: int = 0) -> SystemState:
    rng = np.random.default_rng(seed)
    experts = [Expert(expert_id="s1", index=0), Expert(expert_id="s2", index=1)]
    tasks = {}
    task_pool: list[int] = []
    for i in range(num_tasks):
        tasks[i] = Task(
            task_id=i,
            true_type="c1",
            arrival_time=0.0,
            belief=np.array([0.5, 0.5]),
        )
        task_pool.append(i)
    return SystemState(
        time=0.0,
        lam=0.8,
        experts=experts,
        tasks=tasks,
        task_pool=task_pool,
        completed_task_ids=[],
        event_queue=[],
        rng=rng,
    )


def test_random_policy_respects_idle_and_pool() -> None:
    state = _make_state(num_tasks=3)
    policy = RandomPolicy()
    assignment = policy.select_assignment(state, state.rng)
    assert assignment is not None
    expert, task = assignment
    assert not expert.busy
    assert task.task_id in state.task_pool


def test_random_policy_uniformity() -> None:
    state = _make_state(num_tasks=2, seed=1)
    policy = RandomPolicy()
    counts = {(e.expert_id, t_id): 0 for e in state.experts for t_id in state.task_pool}
    for _ in range(200):
        assignment = policy.select_assignment(state, state.rng)
        assert assignment is not None
        expert, task = assignment
        counts[(expert.expert_id, task.task_id)] += 1
    # All combinations should be visited non-trivially
    for key, count in counts.items():
        assert count > 0, f"Combination {key} was never selected"


def test_backpressure_policy_respects_idle_and_pool() -> None:
    rng = np.random.default_rng(123)
    experts = [
        Expert(expert_id="s1", index=0, busy=False),
        Expert(expert_id="s2", index=1, busy=True),  # force choice of s1
    ]
    tasks = {
        0: Task(task_id=0, true_type="c1", arrival_time=0.0, belief=np.array([0.9, 0.1])),
        1: Task(task_id=1, true_type="c2", arrival_time=0.0, belief=np.array([0.1, 0.9])),
    }
    state = SystemState(
        time=0.0,
        lam=0.8,
        experts=experts,
        tasks=tasks,
        task_pool=[0, 1],
        completed_task_ids=[],
        event_queue=[],
        rng=rng,
    )
    policy = create_backpressure_policy(a=0.5, delta=0.2, eps=0.5)
    assignment = policy.select_assignment(state, state.rng)
    assert assignment is not None
    expert, task = assignment
    assert expert.expert_id == "s1"
    # Under these beliefs and eps=0.5, task 0 has the larger weight for s1.
    assert task.task_id == 0


def test_create_policy_requires_eps_for_backpressure() -> None:
    with pytest.raises(ValueError):
        create_policy("backpressure", a=0.5, delta=0.2, eps=None)
