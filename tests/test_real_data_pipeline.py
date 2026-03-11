"""Tests for real-data pipeline components.

These tests validate the built-in datasets, SN marginalised likelihood,
BAO observables, and the combined inference pipeline.
"""

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ultra_slow_de.builtin_data import (
    load_all_bao,
    load_desi_dh,
    load_desi_dm,
    load_desi_dv,
    load_eboss_dh,
    load_eboss_dm,
    load_planck_compressed,
)
from ultra_slow_de.datasets import validate_dataset
from ultra_slow_de.inference import (
    joint_logposterior,
    loglike_for_dataset,
    loglike_planck_compressed,
    predict_observable,
)
from ultra_slow_de.likelihood import sn_loglike_marg
from ultra_slow_de.params import CosmoParams, ModelAParams


# -- Fiducial cosmology with radiation --
_h = 67.4 / 100.0
_COSMO = CosmoParams(h0=67.4, omega_m=0.315, omega_r=4.176e-5 / _h**2)


# =====================================================================
#  Built-in BAO datasets
# =====================================================================

@pytest.mark.parametrize(
    "loader,kind",
    [
        (load_desi_dv, "dv_rd"),
        (load_desi_dm, "dm_rd"),
        (load_desi_dh, "dh_rd"),
        (load_eboss_dm, "dm_rd"),
        (load_eboss_dh, "dh_rd"),
    ],
)
def test_builtin_bao_datasets_are_valid(loader, kind):
    ds = loader()
    assert ds.kind == kind
    validate_dataset(ds)  # raises if invalid


def test_load_all_bao_returns_five_datasets():
    bao = load_all_bao()
    assert len(bao) == 5
    for ds, obs in bao:
        assert obs in ("dv_rd", "dm_rd", "dh_rd")


def test_planck_compressed_dataset_is_valid():
    ds = load_planck_compressed()
    assert ds.kind == "cmb_compressed"
    assert len(ds.z) == 2
    validate_dataset(ds)


# =====================================================================
#  BAO observable predictions
# =====================================================================

def test_bao_observables_finite_and_positive():
    z = np.array([0.3, 0.7, 1.0, 1.5, 2.3])
    for obs in ("dm_rd", "dh_rd", "dv_rd"):
        pred = predict_observable(z, obs, _COSMO)
        assert np.all(np.isfinite(pred)), f"{obs} has non-finite values"
        assert np.all(pred > 0), f"{obs} has non-positive values"


def test_dm_rd_monotonically_increasing():
    z = np.linspace(0.1, 3.0, 50)
    dm = predict_observable(z, "dm_rd", _COSMO)
    assert np.all(np.diff(dm) > 0)


# =====================================================================
#  SN Marginalised Likelihood
# =====================================================================

def test_sn_loglike_marg_is_finite_for_synthetic_data():
    """Synthetic SN dataset to test the M_B marginalisation."""
    from ultra_slow_de.data_sources import SourceRecord
    from ultra_slow_de.datasets import GaussianDataset

    rng = np.random.default_rng(42)
    z = np.linspace(0.01, 1.5, 50)
    mu_true = predict_observable(z, "mu", _COSMO)
    M_B = -19.25
    sigma = 0.15
    m_obs = mu_true + M_B + rng.normal(0, sigma, size=len(z))
    cov = np.diag(np.full(len(z), sigma**2))

    src = SourceRecord("test", "sn", "", "", "", "")
    ds = GaussianDataset("test_sn", "mb", z, m_obs, cov, src)

    ll = sn_loglike_marg(ds, mu_true)
    assert np.isfinite(ll)


def test_sn_loglike_marg_prefers_correct_model():
    """The marginalised SN likelihood should prefer the true μ(z) shape."""
    from ultra_slow_de.data_sources import SourceRecord
    from ultra_slow_de.datasets import GaussianDataset

    rng = np.random.default_rng(123)
    z = np.linspace(0.01, 1.5, 100)
    mu_true = predict_observable(z, "mu", _COSMO)
    M_B = -19.25
    sigma = 0.10
    m_obs = mu_true + M_B + rng.normal(0, sigma, size=len(z))
    cov = np.diag(np.full(len(z), sigma**2))

    src = SourceRecord("test", "sn", "", "", "", "")
    ds = GaussianDataset("test_sn", "mb", z, m_obs, cov, src)

    ll_true = sn_loglike_marg(ds, mu_true)
    # Wrong model: distort the shape (not a constant shift, which M_B absorbs)
    mu_wrong = mu_true * 1.02
    ll_wrong = sn_loglike_marg(ds, mu_wrong)
    assert ll_true > ll_wrong


# =====================================================================
#  BAO log-likelihoods
# =====================================================================

def test_bao_loglike_is_finite():
    for ds, obs in load_all_bao():
        ll = loglike_for_dataset(ds, obs, _COSMO)
        assert np.isfinite(ll), f"{ds.name} loglike not finite"


# =====================================================================
#  Planck compressed priors
# =====================================================================

def test_planck_loglike_finite_with_radiation():
    ll = loglike_planck_compressed(_COSMO)
    assert np.isfinite(ll)


def test_planck_shift_parameter_near_observed():
    """R should be within ~5σ of observed value at fiducial."""
    from ultra_slow_de.inference import _h_on_grid
    from ultra_slow_de.observables import comoving_distance_flat
    from ultra_slow_de.constants import C_KM_S

    z_g, h_g = _h_on_grid(_COSMO, None, None, z_max=1200.0, n_z=10000)
    dm_g = comoving_distance_flat(z_g, h_g)
    dm_star = float(np.interp(1089.92, z_g, dm_g))
    R = np.sqrt(_COSMO.omega_m) * dm_star * _COSMO.h0 / C_KM_S
    assert abs(R - 1.7502) < 5 * 0.0046


# =====================================================================
#  Combined joint log-posterior
# =====================================================================

def test_joint_logposterior_model_a_a0_equals_lcdm():
    """Model A with B=C=0 should recover ΛCDM exactly."""
    bao = load_all_bao()

    ll_lcdm = joint_logposterior(bao, _COSMO, include_planck=True)
    ma_zero = ModelAParams(w0=-1.0, B=0.0, C=0.0, omega=2.0)
    ll_a0 = joint_logposterior(bao, _COSMO, model_a=ma_zero, include_planck=True)
    assert abs(ll_lcdm - ll_a0) < 1e-6
