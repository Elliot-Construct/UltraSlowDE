import numpy as np

from ultra_slow_de.baseline_lcdm import H_lcdm
from ultra_slow_de.growth_backend import predict_fsig8
from ultra_slow_de.observables import fsig8_pred
from ultra_slow_de.params import CosmoParams


def test_growth_backend_production_mode_returns_finite_values():
    z_grid = np.linspace(1e-4, 3.5, 300)
    z_eff = np.array([0.698, 1.48])
    cosmo = CosmoParams(h0=67.4, omega_m=0.315)
    h_grid = H_lcdm(z_grid, cosmo)

    pred = predict_fsig8(
        z_eff=z_eff,
        z_grid=z_grid,
        h_grid=h_grid,
        omega_m0=cosmo.omega_m,
        mode="production",
        backend="auto",
    )
    assert pred.backend_used in ("internal_perturbation_ode", "class", "camb")
    assert pred.exploratory is False
    assert pred.values.shape == z_eff.shape
    assert np.all(np.isfinite(pred.values))


def test_growth_backend_exploratory_gamma_matches_reference():
    z_grid = np.linspace(1e-4, 3.5, 250)
    z_eff = np.array([0.5, 1.0])
    cosmo = CosmoParams(h0=67.4, omega_m=0.315)
    h_grid = H_lcdm(z_grid, cosmo)

    pred = predict_fsig8(
        z_eff=z_eff,
        z_grid=z_grid,
        h_grid=h_grid,
        omega_m0=cosmo.omega_m,
        mode="exploratory_gamma",
        backend="auto",
    )
    ref = fsig8_pred(z_eff, z_grid, h_grid, cosmo.omega_m)
    assert pred.backend_used == "gamma"
    assert pred.exploratory is True
    np.testing.assert_allclose(pred.values, ref, rtol=1e-10, atol=1e-12)
