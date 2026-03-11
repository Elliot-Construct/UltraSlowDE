import numpy as np

from .params import CosmoParams


def E_lcdm(z: np.ndarray | float, cosmo: CosmoParams) -> np.ndarray:
    z_arr = np.asarray(z, dtype=float)
    omega_de = cosmo.resolved_omega_de()
    e2 = (
        cosmo.omega_r * (1.0 + z_arr) ** 4
        + cosmo.omega_m * (1.0 + z_arr) ** 3
        + cosmo.omega_k * (1.0 + z_arr) ** 2
        + omega_de
    )
    return np.sqrt(e2)


def H_lcdm(z: np.ndarray | float, cosmo: CosmoParams) -> np.ndarray:
    return cosmo.h0 * E_lcdm(z, cosmo)