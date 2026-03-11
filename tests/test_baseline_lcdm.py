import numpy as np

from ultra_slow_de.baseline_lcdm import E_lcdm, H_lcdm
from ultra_slow_de.params import CosmoParams


def test_e_lcdm_at_z0_is_one_for_flat_closure():
    cosmo = CosmoParams(h0=70.0, omega_m=0.3, omega_r=0.0, omega_k=0.0)
    assert np.isclose(E_lcdm(0.0, cosmo), 1.0)


def test_h_lcdm_positive_on_grid():
    z = np.linspace(0.0, 2.0, 50)
    cosmo = CosmoParams(h0=70.0, omega_m=0.3, omega_r=0.0, omega_k=0.0)
    h = H_lcdm(z, cosmo)
    assert np.all(h > 0.0)