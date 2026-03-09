"""Experiment helpers for running Asymmetric(a, δ) simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple

import pandas as pd

from .config import Algorithm
from .simulator import run_replications, run_simulation
from .stats import (
    aggregate_replications,
    meets_significant_digits,
    relative_half_width,
    significant_digits_from_ci,
)

DEFAULT_LAMS: Sequence[float] = (0.75, 0.77, 0.79, 0.81, 0.83, 0.85)
DEFAULT_DELTAS: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5)


def run_single_experiment(
    lam: float,
    a: float,
    delta: float,
    horizon: float,
    warmup: float,
    seed: int,
    algorithm: Algorithm = "random",
    eps: float | None = None,
) -> dict[str, Any]:
    """Run a single simulation and return metrics."""
    return run_simulation(
        lam=lam,
        a=a,
        delta=delta,
        horizon=horizon,
        warmup=warmup,
        seed=seed,
        algorithm=algorithm,
        eps=eps,
    )


def run_rep_experiment(
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
    """Run multiple replications for a single parameter setting."""
    return run_replications(
        lam=lam,
        a=a,
        delta=delta,
        horizon=horizon,
        warmup=warmup,
        n_reps=n_reps,
        base_seed=base_seed,
        algorithm=algorithm,
        eps=eps,
    )


def run_adaptive_rep_experiment(
    lam: float,
    a: float,
    delta: float,
    horizon: float,
    warmup: float,
    base_seed: int,
    algorithm: Algorithm = "random",
    eps: float | None = None,
    min_reps: int = 30,
    max_reps: int = 300,
    rep_batch_size: int = 10,
    min_significant_digits: int = 2,
) -> dict[str, Any]:
    """Run replications until the precision target is reached or max_reps is hit."""
    if min_reps < 2:
        raise ValueError("min_reps must be at least 2")
    if max_reps < min_reps:
        raise ValueError("max_reps must be >= min_reps")
    if rep_batch_size < 1:
        raise ValueError("rep_batch_size must be >= 1")
    if min_significant_digits < 1:
        raise ValueError("min_significant_digits must be >= 1")

    rep_metrics: list[dict[str, Any]] = []
    n_reps_used = 0

    while n_reps_used < max_reps:
        if n_reps_used < min_reps:
            target_reps = min(min_reps, max_reps)
        else:
            target_reps = min(n_reps_used + rep_batch_size, max_reps)

        for i in range(n_reps_used, target_reps):
            seed = base_seed + i
            metrics = run_simulation(
                lam=lam,
                a=a,
                delta=delta,
                horizon=horizon,
                warmup=warmup,
                seed=seed,
                algorithm=algorithm,
                eps=eps,
            )
            rep_metrics.append(metrics)
        n_reps_used = len(rep_metrics)

        if n_reps_used < min_reps:
            continue

        aggregated = aggregate_replications(rep_metrics)
        ci = aggregated["mean_sojourn_time"]
        if meets_significant_digits(
            mean=float(ci["mean"]),
            half_width=float(ci["half_width"]),
            min_significant_digits=min_significant_digits,
        ):
            break

    aggregated = aggregate_replications(rep_metrics)
    ci = aggregated["mean_sojourn_time"]
    achieved_significant_digits = significant_digits_from_ci(
        mean=float(ci["mean"]),
        half_width=float(ci["half_width"]),
    )
    aggregated["replication_metrics"] = rep_metrics
    aggregated["n_reps_used"] = len(rep_metrics)
    aggregated["min_significant_digits_target"] = min_significant_digits
    aggregated["achieved_significant_digits"] = achieved_significant_digits
    aggregated["precision_met"] = achieved_significant_digits >= min_significant_digits
    aggregated["relative_half_width"] = relative_half_width(
        mean=float(ci["mean"]),
        half_width=float(ci["half_width"]),
    )
    return aggregated


def run_grid_experiment(
    a: float,
    lam_values: Iterable[float],
    delta_values: Iterable[float],
    horizon: float,
    warmup: float,
    n_reps: int,
    base_seed: int,
    algorithm: Algorithm = "random",
    eps: float | None = None,
    adaptive_reps: bool = False,
    min_reps: int = 30,
    max_reps: int = 300,
    rep_batch_size: int = 10,
    min_significant_digits: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run a grid of experiments over λ and δ.

    Returns
    -------
    summary_df:
        DataFrame indexed by λ with columns δ. Each cell is a string
        of the form ``\"mean ± half_width\"`` for the mean sojourn time.
    numeric_df:
        Long-form DataFrame with columns ``lam``, ``delta``,
        ``mean_sojourn_time``, ``ci_half_width``, ``ci_low``, ``ci_high``,
        and precision diagnostics (replication count, target/achieved significant digits).
    """
    records = []
    lam_list = list(lam_values)
    delta_list = list(delta_values)

    for lam in lam_list:
        for delta in delta_list:
            if adaptive_reps:
                result = run_adaptive_rep_experiment(
                    lam=lam,
                    a=a,
                    delta=delta,
                    horizon=horizon,
                    warmup=warmup,
                    base_seed=base_seed,
                    algorithm=algorithm,
                    eps=eps,
                    min_reps=min_reps,
                    max_reps=max_reps,
                    rep_batch_size=rep_batch_size,
                    min_significant_digits=min_significant_digits,
                )
            else:
                result = run_rep_experiment(
                    lam=lam,
                    a=a,
                    delta=delta,
                    horizon=horizon,
                    warmup=warmup,
                    n_reps=n_reps,
                    base_seed=base_seed,
                    algorithm=algorithm,
                    eps=eps,
                )
            ci = result["mean_sojourn_time"]
            achieved_significant_digits = significant_digits_from_ci(
                mean=float(ci["mean"]),
                half_width=float(ci["half_width"]),
            )
            records.append(
                {
                    "algorithm": algorithm,
                    "eps": eps,
                    "lam": lam,
                    "delta": delta,
                    "mean_sojourn_time": ci["mean"],
                    "ci_half_width": ci["half_width"],
                    "ci_low": ci["low"],
                    "ci_high": ci["high"],
                    "n_reps_used": int(result.get("n_reps_used", n_reps)),
                    "precision_met": bool(
                        result.get(
                            "precision_met",
                            achieved_significant_digits >= min_significant_digits,
                        )
                    ),
                    "achieved_significant_digits": int(
                        result.get("achieved_significant_digits", achieved_significant_digits)
                    ),
                    "min_significant_digits_target": int(
                        result.get("min_significant_digits_target", min_significant_digits)
                    ),
                    "relative_half_width": float(
                        result.get(
                            "relative_half_width",
                            relative_half_width(
                                mean=float(ci["mean"]),
                                half_width=float(ci["half_width"]),
                            ),
                        )
                    ),
                }
            )

    numeric_df = pd.DataFrame.from_records(records)
    # Build summary table with formatted strings
    summary_df = numeric_df.pivot(index="lam", columns="delta", values="mean_sojourn_time")
    half_width_df = numeric_df.pivot(index="lam", columns="delta", values="ci_half_width")

    formatted = summary_df.copy().astype(str)
    for lam in summary_df.index:
        for delta in summary_df.columns:
            mean = summary_df.loc[lam, delta]
            hw = half_width_df.loc[lam, delta]
            formatted.loc[lam, delta] = f"{mean:.2f} ± {hw:.2f}"

    return formatted, numeric_df


def write_grid_results(summary_df: pd.DataFrame, numeric_df: pd.DataFrame, outdir: Path) -> None:
    """Write grid experiment results to CSV files."""
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "summary.csv"
    numeric_path = outdir / "numeric.csv"
    summary_df.to_csv(summary_path)
    numeric_df.to_csv(numeric_path, index=False)


def format_console_table(summary_df: pd.DataFrame) -> str:
    """Return a pretty-printed string version of the summary table."""
    # Ensure stable ordering
    df = summary_df.sort_index().sort_index(axis=1)
    table_str: str = df.to_string()
    return table_str
