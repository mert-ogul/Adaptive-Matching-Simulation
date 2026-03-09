from typing import Any

from expert_sim.experiments import run_adaptive_rep_experiment, run_grid_experiment


def _fake_metrics(mean_sojourn: float) -> dict[str, Any]:
    return {
        "mean_sojourn_time": mean_sojourn,
        "n_completed_after_warmup": 10.0,
        "total_arrivals": 12.0,
        "tasks_in_system_at_horizon": 2.0,
        "expert_utilization": {"s1": 0.5, "s2": 0.5},
    }


def test_adaptive_replications_stop_early_when_precision_met(monkeypatch: Any) -> None:
    def fake_run_simulation(**_kwargs: Any) -> dict[str, Any]:
        return _fake_metrics(10.0)

    monkeypatch.setattr("expert_sim.experiments.run_simulation", fake_run_simulation)

    result = run_adaptive_rep_experiment(
        lam=0.8,
        a=0.5,
        delta=0.2,
        horizon=100.0,
        warmup=10.0,
        base_seed=1,
        min_reps=4,
        max_reps=20,
        rep_batch_size=3,
        min_significant_digits=2,
    )
    assert result["n_reps_used"] == 4
    assert result["precision_met"] is True


def test_adaptive_replications_hit_max_when_precision_not_met(monkeypatch: Any) -> None:
    def fake_run_simulation(seed: int, **_kwargs: Any) -> dict[str, Any]:
        return _fake_metrics(1.0 if seed % 2 == 0 else -1.0)

    monkeypatch.setattr("expert_sim.experiments.run_simulation", fake_run_simulation)

    result = run_adaptive_rep_experiment(
        lam=0.8,
        a=0.5,
        delta=0.2,
        horizon=100.0,
        warmup=10.0,
        base_seed=1,
        min_reps=4,
        max_reps=10,
        rep_batch_size=2,
        min_significant_digits=2,
    )
    assert result["n_reps_used"] == 10
    assert result["precision_met"] is False


def test_grid_experiment_includes_precision_columns(monkeypatch: Any) -> None:
    def fake_adaptive(**_kwargs: Any) -> dict[str, Any]:
        return {
            "mean_sojourn_time": {
                "mean": 5.0,
                "std": 1.0,
                "half_width": 0.5,
                "low": 4.5,
                "high": 5.5,
            },
            "n_reps_used": 40,
            "precision_met": True,
            "achieved_significant_digits": 2,
            "min_significant_digits_target": 2,
            "relative_half_width": 0.1,
            "replication_metrics": [],
        }

    monkeypatch.setattr("expert_sim.experiments.run_adaptive_rep_experiment", fake_adaptive)

    _summary_df, numeric_df = run_grid_experiment(
        a=0.5,
        lam_values=[0.8],
        delta_values=[0.2],
        horizon=100.0,
        warmup=10.0,
        n_reps=30,
        base_seed=1,
        algorithm="random",
        adaptive_reps=True,
    )
    row = numeric_df.iloc[0]
    assert int(row["n_reps_used"]) == 40
    assert bool(row["precision_met"]) is True
    assert int(row["achieved_significant_digits"]) == 2
