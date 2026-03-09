"""Success probabilities and belief update logic for the Asymmetric(a, δ) system."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt


def build_asymmetric_matrix(a: float, delta: float) -> npt.NDArray[np.float64]:
    """Return the 2x2 success probability matrix for Asymmetric(a, δ).

    The matrix is indexed as p[expert_index, task_type_index] with the ordering:

    - expert_index 0: s1
    - expert_index 1: s2
    - task_type_index 0: c1
    - task_type_index 1: c2
    """
    p: npt.NDArray[np.float64] = np.empty((2, 2), dtype=float)
    p[0, 0] = 1.0 - delta / 2.0  # s1, c1
    p[0, 1] = a  # s1, c2
    p[1, 0] = 1.0 - delta / 2.0  # s2, c1
    p[1, 1] = delta  # s2, c2
    return p


def failure_prob(
    p: npt.NDArray[np.float64],
    expert_index: int,
    belief: npt.NDArray[np.float64],
) -> float:
    """Compute ψ_s(z), the failure probability for expert s given belief z."""
    ps = p[expert_index]
    success_prob = float(belief @ ps)
    return 1.0 - success_prob


def update_belief(
    p: npt.NDArray[np.float64],
    expert_index: int,
    belief: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return updated belief φ_s(z) after a failure by expert s.

    Parameters
    ----------
    p:
        2x2 success probability matrix as returned by :func:`build_asymmetric_matrix`.
    expert_index:
        Index of the expert (0 for s1, 1 for s2).
    belief:
        Prior belief vector z = (z_c1, z_c2) with z_c1 + z_c2 = 1.
    """
    z = np.asarray(belief, dtype=float)
    if z.shape != (2,):
        raise ValueError("belief must be a vector of length 2")

    ps = p[expert_index]
    one_minus_ps = 1.0 - ps
    numerator = z * one_minus_ps
    psi = float(cast(float, numerator.sum()))
    if psi <= 0.0:
        # Degenerate case: no information gain; return original belief.
        return z.copy()
    updated: npt.NDArray[np.float64] = numerator / psi
    # Ensure numerical stability.
    total = float(cast(float, updated.sum()))
    if not np.isclose(total, 1.0):
        updated /= total
    return updated
