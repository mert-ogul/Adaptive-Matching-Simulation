"""Convenience script to run Task 1/Task 2 grids with adaptive Task D precision checks."""

from pathlib import Path

from expert_sim.experiments import DEFAULT_DELTAS, DEFAULT_LAMS, run_grid_experiment


def main() -> None:
    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)
    a = 0.5
    horizon = 10_000.0
    warmup = 1_000.0
    n_reps = 30
    base_seed = 42
    adaptive_reps = True
    min_reps = 30
    max_reps = 300
    rep_batch_size = 10
    min_significant_digits = 2

    summary_random, numeric_random = run_grid_experiment(
        a=a,
        lam_values=DEFAULT_LAMS,
        delta_values=DEFAULT_DELTAS,
        horizon=horizon,
        warmup=warmup,
        n_reps=n_reps,
        base_seed=base_seed,
        algorithm="random",
        adaptive_reps=adaptive_reps,
        min_reps=min_reps,
        max_reps=max_reps,
        rep_batch_size=rep_batch_size,
        min_significant_digits=min_significant_digits,
    )
    summary_backpressure, numeric_backpressure = run_grid_experiment(
        a=a,
        lam_values=DEFAULT_LAMS,
        delta_values=DEFAULT_DELTAS,
        horizon=horizon,
        warmup=warmup,
        n_reps=n_reps,
        base_seed=base_seed,
        algorithm="backpressure",
        eps=1.0 / 40.0,
        adaptive_reps=adaptive_reps,
        min_reps=min_reps,
        max_reps=max_reps,
        rep_batch_size=rep_batch_size,
        min_significant_digits=min_significant_digits,
    )
    from expert_sim.experiments import write_grid_results

    write_grid_results(summary_random, numeric_random, outdir / "random")
    write_grid_results(summary_backpressure, numeric_backpressure, outdir / "backpressure_eps_1_40")


if __name__ == "__main__":
    main()
