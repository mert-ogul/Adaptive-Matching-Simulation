"""
Simulation of the Asymmetric(a, δ) expert system.

The main public entry points are:

- ``run_simulation``: run a single replication of the system.
- ``run_replications``: run multiple independent replications and compute confidence intervals.
"""

from .simulator import run_replications, run_simulation

__all__ = ["run_simulation", "run_replications"]
