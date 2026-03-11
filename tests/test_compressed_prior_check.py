import numpy as np

from ultra_slow_de.compressed_prior_check import compute_fde_at_recombination


def test_compute_fde_at_recombination_model_a_basic():
    # shape: (n_chains, n_steps, n_params)
    chains = np.array(
        [
            [
                [67.5, 0.31, -1.0, 0.0, 0.0, 2.0],
                [68.0, 0.30, -1.0, 0.02, -0.01, 2.5],
                [67.2, 0.32, -0.98, -0.02, 0.01, 1.8],
            ]
        ],
        dtype=float,
    )
    names = ["h0", "omega_m", "w0", "B", "C", "omega"]
    out = compute_fde_at_recombination(chains, names, model="a", max_samples=10)
    assert out["model"] == "a"
    assert out["n_samples_used"] == 3
    assert np.isfinite(out["fde_p95"])
    assert out["fde_p95"] < 1e-3


def test_compute_fde_at_recombination_model_b_basic():
    chains = np.array(
        [
            [
                [68.0, 0.30, 0.6],
                [67.8, 0.31, 0.5],
            ]
        ],
        dtype=float,
    )
    names = ["h0", "omega_m", "mu"]
    out = compute_fde_at_recombination(chains, names, model="b", max_samples=10)
    assert out["model"] == "b"
    assert out["n_samples_used"] == 2
    assert np.isfinite(out["fde_p95"])
    assert out["fde_p95"] < 1e-3
