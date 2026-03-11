import numpy as np
import pytest

from ultra_slow_de.data_sources import SourceRecord
from ultra_slow_de.datasets import GaussianDataset, covariance_from_sigma
from ultra_slow_de.sampler import run_mcmc, run_mcmc_multichain


def _mock_h_dataset():
    """Tiny synthetic H(z) dataset peaking around fiducial ΛCDM."""
    z = np.array([0.1, 0.3, 0.5, 0.7])
    # fiducial: h0=70, omega_m=0.3
    from ultra_slow_de.baseline_lcdm import H_lcdm
    from ultra_slow_de.params import CosmoParams
    cosmo = CosmoParams(h0=70.0, omega_m=0.3)
    h_fid = H_lcdm(z, cosmo)
    sigma = 2.0 * np.ones_like(z)
    src = SourceRecord(name="mock", kind="h", source_url="test",
                       version="0", license="test", provenance="test")
    return GaussianDataset(name="mock_h", kind="h", z=z, y_obs=h_fid,
                           cov=covariance_from_sigma(sigma), source=src)


def test_mcmc_lcdm_pilot_runs():
    ds = _mock_h_dataset()
    theta0 = np.array([70.0, 0.3])
    bounds = np.array([[60.0, 80.0], [0.1, 0.5]])
    sigma = np.array([1.0, 0.02])
    result = run_mcmc([(ds, "h")], model="lcdm", theta0=theta0,
                      bounds=bounds, proposal_sigma=sigma, n_steps=50, seed=0)
    assert result.chain.shape == (50, 2)
    assert 0.0 <= result.accept_rate <= 1.0
    assert np.all(np.isfinite(result.loglike))
    assert result.backend == "numpy"
    assert result.device == "cpu"
    assert result.used_fallback is False


def test_mcmc_model_a_pilot_runs():
    ds = _mock_h_dataset()
    theta0 = np.array([70.0, 0.3, -1.0, 0.0, 0.0, 1.0])
    bounds = np.array([[60, 80], [0.1, 0.5], [-1.5, -0.5],
                       [-0.1, 0.1], [-0.1, 0.1], [0.1, 5.0]])
    sigma = np.array([1.0, 0.02, 0.05, 0.01, 0.01, 0.1])
    result = run_mcmc([(ds, "h")], model="a", theta0=theta0,
                      bounds=bounds, proposal_sigma=sigma, n_steps=30, seed=1)
    assert result.chain.shape == (30, 6)
    assert result.param_names == ["h0", "omega_m", "w0", "B", "C", "omega"]


def test_mcmc_multichain_lcdm_returns_diagnostics():
    ds = _mock_h_dataset()
    theta0 = np.array([70.0, 0.3])
    bounds = np.array([[60.0, 80.0], [0.1, 0.5]])
    sigma = np.array([1.0, 0.02])
    result = run_mcmc_multichain(
        [(ds, "h")],
        model="lcdm",
        theta0=theta0,
        bounds=bounds,
        proposal_sigma=sigma,
        n_steps=40,
        n_chains=3,
        seed=7,
    )

    assert result.chains.shape == (3, 40, 2)
    assert result.loglike.shape == (3, 40)
    assert result.accept_rates.shape == (3,)
    assert np.all((result.accept_rates >= 0.0) & (result.accept_rates <= 1.0))
    assert result.rhat_per_param is not None
    assert result.ess_per_param is not None
    assert result.rhat_per_param.shape == (2,)
    assert result.ess_per_param.shape == (2,)
    assert np.all(np.isfinite(result.rhat_per_param))
    assert np.all(result.ess_per_param > 0)
    assert result.backend == "numpy"
    assert result.device == "cpu"
    assert result.used_fallback is False
