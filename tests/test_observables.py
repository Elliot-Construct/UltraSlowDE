import numpy as np

from ultra_slow_de.baseline_lcdm import H_lcdm
from ultra_slow_de.observables import deceleration_parameter, luminosity_distance_flat
from ultra_slow_de.params import CosmoParams


def test_luminosity_distance_zero_at_z_zero():
    z = np.linspace(0.0, 2.0, 100)
    cosmo = CosmoParams()
    h = H_lcdm(z, cosmo)
    dl = luminosity_distance_flat(z, h)
    assert np.isclose(dl[0], 0.0)


def test_luminosity_distance_monotonic_for_positive_z():
    z = np.linspace(0.0, 2.0, 100)
    cosmo = CosmoParams()
    h = H_lcdm(z, cosmo)
    dl = luminosity_distance_flat(z, h)
    assert np.all(np.diff(dl) >= 0.0)


def test_deceleration_is_finite():
    z = np.linspace(0.0, 2.0, 100)
    cosmo = CosmoParams()
    h = H_lcdm(z, cosmo)
    q = deceleration_parameter(z, h)
    assert np.all(np.isfinite(q))