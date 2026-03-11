import numpy as np
import pytest

from ultra_slow_de.baseline_lcdm import H_lcdm
from ultra_slow_de.model_b import (
    H_model_b,
    ModelBParams,
    PotentialType,
    solve_model_b,
)
from ultra_slow_de.params import CosmoParams


def test_model_b_near_lcdm_in_thawing_limit():
    """With mu << 1 and frozen IC, Model B should approximate ΛCDM."""
    z = np.linspace(0.0, 2.0, 50)
    cosmo = CosmoParams(h0=70.0, omega_m=0.3, omega_r=0.0, omega_k=0.0)
    model = ModelBParams(potential=PotentialType.QUADRATIC, mu=0.001, dphi_i=0.0)
    h_lcdm = H_lcdm(z, cosmo)
    h_b = H_model_b(z, cosmo.h0, omega_m=cosmo.omega_m, omega_r=cosmo.omega_r, model=model)
    assert np.allclose(h_b, h_lcdm, rtol=1e-3), f"max rel diff: {np.max(np.abs(h_b/h_lcdm - 1))}"


def test_model_b_w_near_minus_one_thawing():
    """w_phi should be very close to -1 in the thawing limit."""
    z = np.linspace(0.0, 2.0, 50)
    model = ModelBParams(potential=PotentialType.QUADRATIC, mu=0.001, dphi_i=0.0)
    result = solve_model_b(z, omega_m=0.3, omega_r=0.0, model=model)
    assert np.all(np.abs(result["w_phi"] + 1.0) < 1e-4)


def test_model_b_e_z_positive_and_finite():
    z = np.linspace(0.0, 3.0, 80)
    model = ModelBParams(potential=PotentialType.QUADRATIC, mu=0.5, dphi_i=0.0)
    result = solve_model_b(z, omega_m=0.3, omega_r=0.0, model=model)
    assert np.all(np.isfinite(result["E_z"]))
    assert np.all(result["E_z"] > 0)


def test_model_b_cosine_potential_runs():
    """Cosine potential should integrate without error."""
    z = np.linspace(0.0, 2.0, 30)
    model = ModelBParams(potential=PotentialType.COSINE, lam=0.5, f_tilde=2.0, dphi_i=0.0)
    result = solve_model_b(z, omega_m=0.3, omega_r=0.0, model=model)
    assert np.all(np.isfinite(result["E_z"]))


def test_model_b_closure():
    """Omega_m + Omega_r + Omega_phi should sum to 1 at z=0."""
    z = np.array([0.0])
    model = ModelBParams(potential=PotentialType.QUADRATIC, mu=0.1, dphi_i=0.0)
    result = solve_model_b(z, omega_m=0.3, omega_r=0.0, model=model)
    omega_total = 0.3 + result["omega_phi"][0]
    assert np.isclose(omega_total, 1.0, atol=1e-3), f"omega_total={omega_total}"
