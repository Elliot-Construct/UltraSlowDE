"""Tests for the BlackJAX NUTS sampler for Model A.

These tests use a small synthetic dataset so they run quickly.  They verify:
  * JAX physics helpers (xde, H(z), D_M(z)) match the reference numpy code.
  * Bijection round-trips (theta <-> psi) are correct and invertible.
  * The log-posterior is finite at a known good parameter point.
  * run_nuts_model_a returns a valid MultiChainMCMCResult.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ultra_slow_de.data_sources import SourceRecord
from ultra_slow_de.datasets import GaussianDataset, covariance_from_sigma
from ultra_slow_de.sampler import MultiChainMCMCResult
from ultra_slow_de.sampler_nuts import (
    _psi_to_theta,
    _theta_to_psi,
    _log_jacobian_psi_to_theta,
    _xde_a_jax,
    _hz_grid_a_jax,
    _comoving_dist_jax,
    _make_logpost_jax,
    _prepare,
    run_nuts_model_a,
    PARAM_NAMES_A,
    _BOUNDS_NP,
)
from ultra_slow_de.model_a import xde_model_a, H_model_a
from ultra_slow_de.params import CosmoParams, ModelAParams

# ---------------------------------------------------------------------------
# Tiny synthetic SN + BAO dataset (no real data needed)
# ---------------------------------------------------------------------------

def _make_synthetic_datasets(seed: int = 42):
    """Build a minimal synthetic dataset that exercises the full likelihood path."""
    rng = np.random.default_rng(seed)
    src = SourceRecord(
        name="synthetic", kind="sn", source_url="test",
        version="0", license="test", provenance="test"
    )
    # SN Ia: 10 points
    z_sn = np.linspace(0.05, 0.8, 10)
    cosmo = CosmoParams(h0=67.4, omega_m=0.315)
    ma = ModelAParams(w0=-1.0, B=0.05, C=0.02, omega=1.5)
    h_grid = H_model_a(np.linspace(0.0, 1.5, 200), cosmo, ma)
    from ultra_slow_de.observables import luminosity_distance_flat
    dl = luminosity_distance_flat(np.linspace(0.0, 1.5, 200), h_grid)
    dl_sn = np.interp(z_sn, np.linspace(0.0, 1.5, 200), dl)
    mu_fid = 5.0 * np.log10(dl_sn * 1e6 / 10.0)
    mu_obs = mu_fid + rng.normal(0.0, 0.15, size=len(z_sn))
    sn_ds = GaussianDataset(
        name="synth_sn", kind="sn", z=z_sn, y_obs=mu_obs,
        cov=covariance_from_sigma(0.15 * np.ones(len(z_sn))),
        source=src,
    )

    # BAO: 3 points (dm_rd only)
    z_bao = np.array([0.38, 0.51, 0.70])
    rd = 147.09
    from ultra_slow_de.observables import dm_over_rd
    dm_rd_fid = dm_over_rd(z_bao, np.linspace(0.0, 1.5, 200), h_grid, rd)
    dm_rd_obs = dm_rd_fid + rng.normal(0.0, 0.05, size=len(z_bao))
    bao_src = SourceRecord(
        name="synthetic_bao", kind="bao", source_url="test",
        version="0", license="test", provenance="test"
    )
    bao_ds = GaussianDataset(
        name="synth_bao", kind="bao", z=z_bao, y_obs=dm_rd_obs,
        cov=covariance_from_sigma(0.05 * np.ones(len(z_bao))),
        source=bao_src,
    )

    return [(sn_ds, "mb"), (bao_ds, "dm_rd")]


DATASETS = _make_synthetic_datasets()
THETA_FIDUCIAL = np.array([67.4, 0.315, -1.0, 0.05, 0.02, 1.5])
RD = 147.09


# ---------------------------------------------------------------------------
# Bijection tests
# ---------------------------------------------------------------------------

def test_theta_psi_roundtrip():
    """theta -> psi -> theta is the identity."""
    theta0 = jnp.asarray(THETA_FIDUCIAL)
    psi = _theta_to_psi(theta0)
    theta_back = _psi_to_theta(psi)
    np.testing.assert_allclose(np.asarray(theta_back), np.asarray(theta0), rtol=1e-10)


def test_psi_unconstrained():
    """psi is in (-inf, inf) for all interior theta values."""
    theta0 = jnp.asarray(THETA_FIDUCIAL)
    psi = _theta_to_psi(theta0)
    assert np.all(np.isfinite(np.asarray(psi)))


def test_log_jacobian_negative():
    """The log-Jacobian is always negative (log of a product < 1)."""
    theta0 = jnp.asarray(THETA_FIDUCIAL)
    psi = _theta_to_psi(theta0)
    lj = float(_log_jacobian_psi_to_theta(psi))
    assert np.isfinite(lj)
    assert lj < 0.0


# ---------------------------------------------------------------------------
# JAX physics vs. reference numpy code
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("omega", [0.5, 1.5, 3.0])
def test_xde_jax_matches_numpy(omega):
    """_xde_a_jax reproduces xde_model_a for a range of omega values."""
    ma = ModelAParams(w0=-1.0, B=0.05, C=0.02, omega=omega)
    a_grid = np.linspace(0.1, 1.0, 50)
    ln_a = np.log(a_grid)

    ref = xde_model_a(a_grid, ma)
    got = np.asarray(_xde_a_jax(jnp.asarray(ln_a), ma.w0, ma.B, ma.C, ma.omega))
    np.testing.assert_allclose(got, ref, rtol=1e-9)


def test_hz_grid_jax_matches_numpy():
    """_hz_grid_a_jax reproduces H_model_a (with matching Omega_r) for the fiducial point."""
    from ultra_slow_de.sampler_nuts import _OMEGA_R_H2
    h0 = 67.4
    omega_m = 0.315
    omega_r = _OMEGA_R_H2 / (h0 / 100.0) ** 2
    cosmo = CosmoParams(h0=h0, omega_m=omega_m, omega_r=omega_r)
    ma = ModelAParams(w0=-1.0, B=0.05, C=0.02, omega=1.5)
    z_grid = np.linspace(1e-4, 3.0, 300)

    ref = H_model_a(z_grid, cosmo, ma)
    z_jax = jnp.asarray(z_grid)
    got = np.asarray(_hz_grid_a_jax(
        h0, omega_m, ma.w0, ma.B, ma.C, ma.omega, z_jax
    ))
    # JAX and numpy should agree to near machine precision.
    np.testing.assert_allclose(got, ref, rtol=1e-9)


def test_comoving_dist_jax():
    """D_M computed in JAX matches scipy cumtrapz reference."""
    from ultra_slow_de.observables import comoving_distance_flat
    cosmo = CosmoParams(h0=67.4, omega_m=0.315)
    ma = ModelAParams(w0=-1.0, B=0.05, C=0.02, omega=1.5)
    z_grid = np.linspace(1e-4, 3.0, 400)
    h_np = H_model_a(z_grid, cosmo, ma)

    ref = comoving_distance_flat(z_grid, h_np)
    h_jax = jnp.asarray(h_np)
    z_jax = jnp.asarray(z_grid)
    got = np.asarray(_comoving_dist_jax(z_jax, h_jax))
    # Match to within 0.01% (trapezoidal rule, identical grids)
    np.testing.assert_allclose(got, ref, rtol=1e-4)


# ---------------------------------------------------------------------------
# Log-posterior sanity checks
# ---------------------------------------------------------------------------

def test_logpost_finite_at_fiducial():
    """Log-posterior is finite at the fiducial parameter point."""
    prep = _prepare(DATASETS, RD, include_planck=False)
    logpost = _make_logpost_jax(prep)
    psi = _theta_to_psi(jnp.asarray(THETA_FIDUCIAL))
    val = float(logpost(psi))
    assert np.isfinite(val), f"log-posterior = {val}"


def test_logpost_gradient_finite():
    """JAX autodiff of log-posterior is finite at the fiducial point."""
    prep = _prepare(DATASETS, RD, include_planck=False)
    logpost = _make_logpost_jax(prep)
    psi = _theta_to_psi(jnp.asarray(THETA_FIDUCIAL))
    grad = np.asarray(jax.grad(logpost)(psi))
    assert np.all(np.isfinite(grad)), f"gradient has non-finite element: {grad}"


# ---------------------------------------------------------------------------
# run_nuts_model_a integration test (fast: 2 chains × 50 samples)
# ---------------------------------------------------------------------------

def test_run_nuts_model_a_smoke():
    """run_nuts_model_a returns a valid MultiChainMCMCResult for a tiny run."""
    result = run_nuts_model_a(
        datasets=DATASETS,
        theta0=THETA_FIDUCIAL,
        rd=RD,
        include_planck=False,
        n_chains=2,
        n_warmup=100,
        n_samples=50,
        seed=1,
    )
    assert isinstance(result, MultiChainMCMCResult)
    assert result.chains.shape == (2, 50, 6)
    assert result.loglike.shape == (2, 50)
    assert result.param_names == PARAM_NAMES_A
    assert np.all(np.isfinite(result.loglike))
    # All samples should be within prior bounds
    lo = _BOUNDS_NP[:, 0]
    hi = _BOUNDS_NP[:, 1]
    for c in range(2):
        assert np.all(result.chains[c] >= lo - 1e-6)
        assert np.all(result.chains[c] <= hi + 1e-6)
    assert result.accept_rates.shape == (2,)
    assert np.all(result.accept_rates >= 0.0) and np.all(result.accept_rates <= 1.0)
    assert result.backend == "blackjax-nuts"
    assert result.rhat_per_param is None or len(result.rhat_per_param) == 6
