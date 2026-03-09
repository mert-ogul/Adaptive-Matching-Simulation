from expert_sim.config import SimulationConfig
from expert_sim.models import Task
from expert_sim.policies import create_policy, create_random_policy
from expert_sim.probabilities import build_asymmetric_matrix
from expert_sim.simulator import Simulator
from expert_sim.stats import SimulationResult, compute_single_run_metrics


def _run_short_sim(seed: int = 0) -> SimulationResult:
    config = SimulationConfig(
        lam=0.8,
        a=0.5,
        delta=0.2,
        horizon=10.0,
        warmup=0.0,
        seed=seed,
    )
    policy = create_random_policy()
    sim = Simulator(config, policy)
    return sim.run()


def _run_short_backpressure_sim(seed: int = 0) -> SimulationResult:
    config = SimulationConfig(
        lam=0.8,
        a=0.5,
        delta=0.2,
        horizon=10.0,
        warmup=0.0,
        seed=seed,
        algorithm="backpressure",
        eps=1.0 / 40.0,
    )
    policy = create_policy(config.algorithm, config.a, config.delta, config.eps)
    sim = Simulator(config, policy)
    return sim.run()


def test_failure_returns_task_to_pool_or_completes() -> None:
    result = _run_short_sim(seed=1)
    tasks: list[Task] = result.tasks
    # At least one task should have been attempted (some may still be in pool at horizon)
    assert any(t.attempts >= 1 for t in tasks)


def test_success_sets_departure_time() -> None:
    result = _run_short_sim(seed=2)
    has_completed = any(t.departure_time is not None for t in result.tasks)
    assert has_completed


def test_utilization_bounded_between_zero_and_one() -> None:
    result = _run_short_sim(seed=3)
    metrics = compute_single_run_metrics(result, warmup=0.0)
    util = metrics["expert_utilization"]
    assert isinstance(util, dict)
    for value in util.values():
        assert 0.0 <= value <= 1.0


def test_probabilities_consistent() -> None:
    p = build_asymmetric_matrix(a=0.5, delta=0.2)
    # Success probabilities must be in [0, 1]
    assert ((p >= 0.0) & (p <= 1.0)).all()


def test_backpressure_simulation_runs() -> None:
    result = _run_short_backpressure_sim(seed=4)
    metrics = compute_single_run_metrics(result, warmup=0.0)
    assert metrics["total_arrivals"] > 0
