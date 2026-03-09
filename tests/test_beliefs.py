import numpy as np

from expert_sim.probabilities import build_asymmetric_matrix, update_belief


def test_asymmetric_matrix_values() -> None:
    a = 0.5
    delta = 0.2
    p = build_asymmetric_matrix(a, delta)
    assert p.shape == (2, 2)
    assert np.isclose(p[0, 0], 1 - delta / 2)
    assert np.isclose(p[1, 0], 1 - delta / 2)
    assert np.isclose(p[0, 1], a)
    assert np.isclose(p[1, 1], delta)


def test_belief_update_matches_formula() -> None:
    a = 0.5
    delta = 0.1
    p = build_asymmetric_matrix(a, delta)
    z = np.array([0.5, 0.5])
    # Use expert s2 (index 1) as in the text
    updated = update_belief(p, expert_index=1, belief=z)

    # Manual computation from definition
    one_minus_p = 1 - p[1]
    psi = (z * one_minus_p).sum()
    expected = z * one_minus_p / psi

    assert np.allclose(updated, expected)
    assert np.isclose(updated.sum(), 1.0)
