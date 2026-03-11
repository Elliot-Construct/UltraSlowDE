import numpy as np
import pytest

from ultra_slow_de.data_sources import SourceRecord
from ultra_slow_de.datasets import GaussianDataset, covariance_from_sigma
from ultra_slow_de.sampler import run_mcmc_backend, run_mcmc_multichain_backend


def _mock_h_dataset():
    z = np.array([0.1, 0.3, 0.5, 0.7])
    from ultra_slow_de.baseline_lcdm import H_lcdm
    from ultra_slow_de.params import CosmoParams

    cosmo = CosmoParams(h0=70.0, omega_m=0.3)
    h_fid = H_lcdm(z, cosmo)
    sigma = 2.0 * np.ones_like(z)
    src = SourceRecord(name="mock", kind="h", source_url="test", version="0", license="test", provenance="test")
    return GaussianDataset(
        name="mock_h",
        kind="h",
        z=z,
        y_obs=h_fid,
        cov=covariance_from_sigma(sigma),
        source=src,
    )


def _lcdm_setup():
    theta0 = np.array([70.0, 0.3])
    bounds = np.array([[60.0, 80.0], [0.1, 0.5]])
    sigma = np.array([1.0, 0.02])
    return theta0, bounds, sigma


def test_backend_dispatch_numpy_is_deterministic():
    ds = _mock_h_dataset()
    theta0, bounds, sigma = _lcdm_setup()

    res1 = run_mcmc_multichain_backend(
        [(ds, "h")],
        model="lcdm",
        theta0=theta0,
        bounds=bounds,
        proposal_sigma=sigma,
        n_steps=25,
        n_chains=2,
        seed=123,
        backend="numpy",
    )
    res2 = run_mcmc_multichain_backend(
        [(ds, "h")],
        model="lcdm",
        theta0=theta0,
        bounds=bounds,
        proposal_sigma=sigma,
        n_steps=25,
        n_chains=2,
        seed=123,
        backend="numpy",
    )

    assert np.array_equal(res1.chains, res2.chains)
    assert np.array_equal(res1.loglike, res2.loglike)
    assert np.array_equal(res1.accept_rates, res2.accept_rates)
    assert res1.backend == "numpy"
    assert res1.device == "cpu"
    assert res1.used_fallback is False


def test_backend_dispatch_jax_smoke_or_fallback():
    ds = _mock_h_dataset()
    theta0, bounds, sigma = _lcdm_setup()

    res = run_mcmc_multichain_backend(
        [(ds, "h")],
        model="lcdm",
        theta0=theta0,
        bounds=bounds,
        proposal_sigma=sigma,
        n_steps=20,
        n_chains=2,
        seed=11,
        backend="jax",
    )

    assert res.chains.shape == (2, 20, 2)
    assert res.loglike.shape == (2, 20)
    assert res.accept_rates.shape == (2,)
    assert res.backend in ("numpy", "jax")
    if res.backend == "numpy":
        assert res.used_fallback is True
        assert res.device == "cpu"
    else:
        assert res.used_fallback is False
        assert isinstance(res.device, str)
        assert len(res.device) > 0


def test_backend_dispatch_single_chain_smoke():
    ds = _mock_h_dataset()
    theta0, bounds, sigma = _lcdm_setup()

    res = run_mcmc_backend(
        [(ds, "h")],
        model="lcdm",
        theta0=theta0,
        bounds=bounds,
        proposal_sigma=sigma,
        n_steps=20,
        seed=9,
        backend="jax",
    )

    assert res.chain.shape == (20, 2)
    assert res.loglike.shape == (20,)
    assert res.backend in ("numpy", "jax")


def test_backend_dispatch_rejects_unknown_backend():
    ds = _mock_h_dataset()
    theta0, bounds, sigma = _lcdm_setup()

    with pytest.raises(ValueError):
        run_mcmc_multichain_backend(
            [(ds, "h")],
            model="lcdm",
            theta0=theta0,
            bounds=bounds,
            proposal_sigma=sigma,
            n_steps=10,
            n_chains=2,
            backend="not-a-backend",
        )
