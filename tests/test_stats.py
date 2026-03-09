import numpy as np

from expert_sim.config import SimulationConfig
from expert_sim.models import Expert, Task
from expert_sim.stats import (
    SimulationResult,
    compute_single_run_metrics,
    meets_significant_digits,
    relative_half_width,
    significant_digits_from_ci,
    t_confidence_interval,
)


def test_t_confidence_interval_basic() -> None:
    means = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ci = t_confidence_interval(means, alpha=0.05)
    assert ci["high"] > ci["mean"] > ci["low"]
    assert ci["half_width"] > 0


def test_warmup_filtering() -> None:
    config = SimulationConfig(
        lam=0.8,
        a=0.5,
        delta=0.2,
        horizon=100.0,
        warmup=10.0,
        seed=1,
    )
    tasks = [
        Task(
            task_id=0,
            true_type="c1",
            arrival_time=0.0,
            belief=np.array([0.5, 0.5]),
            departure_time=5.0,
        ),
        Task(
            task_id=1,
            true_type="c1",
            arrival_time=0.0,
            belief=np.array([0.5, 0.5]),
            departure_time=20.0,
        ),
    ]
    experts = [Expert(expert_id="s1", index=0), Expert(expert_id="s2", index=1)]
    result = SimulationResult(config=config, tasks=tasks, experts=experts, horizon=config.horizon)
    metrics = compute_single_run_metrics(result, warmup=config.warmup)
    # Only the task completing at time 20 should be counted
    assert metrics["n_completed_after_warmup"] == 1


def test_significant_digits_examples_from_assignment_style() -> None:
    assert significant_digits_from_ci(mean=1.2, half_width=0.3) >= 2
    assert significant_digits_from_ci(mean=45.0, half_width=6.0) >= 2
    assert significant_digits_from_ci(mean=7.89, half_width=0.01) >= 3


def test_meets_significant_digits() -> None:
    assert meets_significant_digits(mean=1.2, half_width=0.3, min_significant_digits=2)
    assert not meets_significant_digits(mean=1.2, half_width=1.2, min_significant_digits=2)


def test_relative_half_width_handles_zero_mean() -> None:
    assert np.isinf(relative_half_width(mean=0.0, half_width=0.1))
