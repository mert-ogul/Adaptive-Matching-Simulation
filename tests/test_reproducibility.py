from expert_sim.simulator import run_replications, run_simulation


def test_reproducibility_single_run() -> None:
    params = dict(lam=0.8, a=0.5, delta=0.2, horizon=100.0, warmup=10.0, seed=123)
    m1 = run_simulation(**params)
    m2 = run_simulation(**params)
    assert m1 == m2


def test_independent_replications_differ() -> None:
    params = dict(lam=0.8, a=0.5, delta=0.2, horizon=100.0, warmup=10.0)
    res = run_replications(n_reps=3, base_seed=100, **params)
    reps = res["replication_metrics"]
    # At least two replications should differ in some metric
    assert any(reps[i] != reps[j] for i in range(len(reps)) for j in range(i + 1, len(reps)))


def test_reproducibility_single_run_backpressure() -> None:
    params = dict(
        lam=0.8,
        a=0.5,
        delta=0.2,
        horizon=100.0,
        warmup=10.0,
        seed=123,
        algorithm="backpressure",
        eps=1.0 / 40.0,
    )
    m1 = run_simulation(**params)
    m2 = run_simulation(**params)
    assert m1 == m2
