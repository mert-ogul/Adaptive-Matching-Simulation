"""Statistics utilities for simulation outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from scipy import stats as sps

from .config import SimulationConfig
from .models import Expert, Task


@dataclass(slots=True)
class SimulationResult:
    """Raw outputs from a single simulation run."""

    config: SimulationConfig
    tasks: list[Task]
    experts: list[Expert]
    horizon: float


def filter_completed_after_warmup(result: SimulationResult, warmup: float) -> list[Task]:
    """Return tasks completed strictly after the warm-up time."""
    return [t for t in result.tasks if t.departure_time is not None and t.departure_time > warmup]


def compute_single_run_metrics(
    result: SimulationResult,
    warmup: float,
) -> dict[str, Any]:
    """Compute aggregate metrics for a single simulation run."""
    completed = filter_completed_after_warmup(result, warmup)
    if completed:
        durations = [cast(float, t.departure_time) - t.arrival_time for t in completed]
        sojourn_times: npt.NDArray[np.float64] = np.asarray(durations, dtype=float)
        mean_sojourn = float(cast(float, sojourn_times.mean()))
    else:
        mean_sojourn = float("nan")

    total_arrivals = len(result.tasks)
    tasks_in_system = sum(1 for t in result.tasks if t.departure_time is None)

    expert_utilization: dict[str, float] = {}
    for expert in result.experts:
        expert_utilization[expert.expert_id] = expert.busy_time / result.horizon

    return {
        "mean_sojourn_time": mean_sojourn,
        "n_completed_after_warmup": len(completed),
        "total_arrivals": total_arrivals,
        "tasks_in_system_at_horizon": tasks_in_system,
        "expert_utilization": expert_utilization,
    }


def t_confidence_interval(
    means: npt.NDArray[np.float64],
    alpha: float = 0.05,
) -> dict[str, float]:
    """Return t-based confidence interval for an array of replication means."""
    if means.ndim != 1:
        raise ValueError("means must be one-dimensional")
    n = means.size
    if n < 2:
        raise ValueError("At least two replications are required for a t-interval")
    sample_mean = float(cast(float, means.mean()))
    sample_std = float(cast(float, means.std(ddof=1)))
    df = n - 1
    t_crit = float(sps.t.ppf(1.0 - alpha / 2.0, df))
    half_width = t_crit * sample_std / np.sqrt(n)
    return {
        "mean": sample_mean,
        "std": sample_std,
        "half_width": half_width,
        "low": sample_mean - half_width,
        "high": sample_mean + half_width,
    }


def significant_digits_from_ci(mean: float, half_width: float) -> int:
    """Estimate reliable significant digits in ``mean`` given CI half-width.

    Uses the order-of-magnitude rule:

    ``digits = floor(log10(|mean|)) - floor(log10(half_width)) + 1``
    """
    if not (np.isfinite(mean) and np.isfinite(half_width)):
        return 0
    if mean == 0.0:
        return 0
    if half_width < 0.0:
        return 0
    if half_width == 0.0:
        # Degenerate case (all replication means equal): effectively very high precision.
        return 16

    order_mean = math.floor(math.log10(abs(mean)))
    order_half_width = math.floor(math.log10(half_width))
    return max(0, order_mean - order_half_width + 1)


def meets_significant_digits(
    mean: float,
    half_width: float,
    min_significant_digits: int = 2,
) -> bool:
    """Return whether CI precision meets the target significant digits."""
    if min_significant_digits < 1:
        raise ValueError("min_significant_digits must be >= 1")
    return significant_digits_from_ci(mean, half_width) >= min_significant_digits


def relative_half_width(mean: float, half_width: float) -> float:
    """Return |half_width / mean|, or inf when mean is zero."""
    if mean == 0.0:
        return float("inf")
    return abs(half_width / mean)


def aggregate_replications(rep_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate replication-level metrics into means and confidence intervals."""
    if not rep_metrics:
        raise ValueError("rep_metrics must not be empty")

    scalar_keys = [
        "mean_sojourn_time",
        "n_completed_after_warmup",
        "total_arrivals",
        "tasks_in_system_at_horizon",
    ]

    aggregated: dict[str, Any] = {}
    for key in scalar_keys:
        values: npt.NDArray[np.float64] = np.asarray(
            [float(m[key]) for m in rep_metrics],
            dtype=float,
        )
        aggregated[key] = t_confidence_interval(values)

    return aggregated
