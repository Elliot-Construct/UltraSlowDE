import numpy as np
import pytest

from ultra_slow_de.baseline_lcdm import H_lcdm
from ultra_slow_de.model_a import H_model_a, xde_model_a, amplitude, phase
from ultra_slow_de.params import CosmoParams, ModelAParams


def test_model_a_recovers_lcdm_when_bc_zero_and_w0_minus_one():
    z = np.linspace(0.0, 2.0, 120)
    cosmo = CosmoParams(h0=70.0, omega_m=0.3, omega_r=0.0, omega_k=0.0)
    model = ModelAParams(w0=-1.0, B=0.0, C=0.0, omega=1.0)
    assert np.allclose(H_model_a(z, cosmo, model), H_lcdm(z, cosmo), rtol=1e-12, atol=1e-12)


def test_xde_small_omega_is_finite():
    a = np.geomspace(0.2, 1.0, 200)
    model = ModelAParams(w0=-1.0, B=0.05, C=0.0, omega=1e-10)
    xde = xde_model_a(a, model)
    assert np.all(np.isfinite(xde))
    assert np.all(xde > 0.0)


def test_xde_omega_zero_limit():
    """At omega=0, w=w0+C (constant), so X_de = a^{-3(1+w0+C)}."""
    a = np.geomspace(0.1, 1.0, 50)
    model = ModelAParams(w0=-0.8, B=0.1, C=0.05, omega=0.0)
    xde = xde_model_a(a, model)
    expected = a ** (-3.0 * (1.0 + model.w0 + model.C))
    assert np.allclose(xde, expected, rtol=1e-12)


def test_amplitude_and_phase():
    """amplitude = sqrt(B^2+C^2), phase = arctan2(B,C)."""
    model = ModelAParams(w0=-1.0, B=0.05, C=0.05, omega=1.0)
    amp = amplitude(model)
    phi = phase(model)
    assert np.isclose(amp, 0.05 * np.sqrt(2), rtol=1e-12)
    assert np.isclose(phi, np.pi / 4, rtol=1e-10)


def test_xde_continuity_near_omega_zero():
    """xde should be continuous across the omega=0 branch."""
    a = np.array([0.5, 1.0])
    model_zero = ModelAParams(w0=-1.0, B=0.0, C=0.1, omega=0.0)
    model_tiny = ModelAParams(w0=-1.0, B=0.0, C=0.1, omega=1e-8)
    xde_zero = xde_model_a(a, model_zero)
    xde_tiny = xde_model_a(a, model_tiny)
    assert np.allclose(xde_zero, xde_tiny, rtol=1e-6)