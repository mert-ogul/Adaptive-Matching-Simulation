"""Configuration objects for Asymmetric(a, δ) simulations."""

from dataclasses import dataclass
from typing import Literal

Algorithm = Literal["random", "backpressure"]


@dataclass(slots=True)
class SimulationConfig:
    """Configuration for a single simulation run.

    Parameters
    ----------
    lam:
        Arrival rate λ of the Poisson process.
    a:
        Success probability p[s1, c2] in the Asymmetric(a, δ) model.
    delta:
        Asymmetry parameter δ in the Asymmetric(a, δ) model.
    horizon:
        Simulation time horizon. Events scheduled after this time are ignored.
    warmup:
        Warm-up time. Tasks completed at or before this time are excluded from
        steady-state statistics.
    seed:
        Base random seed for the NumPy random generator.
    """

    lam: float
    a: float
    delta: float
    horizon: float
    warmup: float
    seed: int
    n_experts: int = 2
    algorithm: Algorithm = "random"
    eps: float | None = None

    def __post_init__(self) -> None:
        if self.lam <= 0.0:
            raise ValueError("lam must be positive")
        if not (0.0 < self.a < 1.0):
            raise ValueError("a must be in (0, 1)")
        if not (0.0 < self.delta < 1.0):
            raise ValueError("delta must be in (0, 1)")
        if not (0.0 <= self.warmup < self.horizon):
            raise ValueError("Require 0 <= warmup < horizon")
        if self.n_experts != 2:
            raise ValueError("Current implementation assumes exactly two experts.")
        if self.algorithm not in {"random", "backpressure"}:
            raise ValueError("algorithm must be one of {'random', 'backpressure'}")
        if self.algorithm == "backpressure":
            if self.eps is None:
                raise ValueError("eps is required when algorithm='backpressure'")
            if not (0.0 < self.eps <= 1.0):
                raise ValueError("eps must be in (0, 1] when algorithm='backpressure'")
