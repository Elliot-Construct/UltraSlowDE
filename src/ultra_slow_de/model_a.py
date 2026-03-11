import numpy as np

from .params import CosmoParams, ModelAParams


def w_model_a(a: np.ndarray | float, model: ModelAParams) -> np.ndarray:
    a_arr = np.asarray(a, dtype=float)
    ln_a = np.log(a_arr)
    return model.w0 + model.B * np.sin(model.omega * ln_a) + model.C * np.cos(model.omega * ln_a)


def xde_model_a(a: np.ndarray | float, model: ModelAParams) -> np.ndarray:
    """Dark-energy density ratio X_de(a) = rho_de(a) / rho_de0.

    For w(a) = w0 + B*sin(omega*ln a) + C*cos(omega*ln a):
      ln X_de = -3(1+w0)*ln_a
                - 3*B * (1-cos(omega*ln_a))/omega
                - 3*C * sin(omega*ln_a)/omega

    Uses numerically stable sinc forms valid as omega -> 0.
    At omega=0: w = w0+C, so X_de = a^{-3(1+w0+C)}.
    """
    a_arr = np.asarray(a, dtype=float)
    ln_a = np.log(a_arr)

    if np.isclose(model.omega, 0.0):
        return np.exp(-3.0 * (1.0 + model.w0 + model.C) * ln_a)

    # Numerically stable via sinc (numpy: np.sinc(y) = sin(pi*y)/(pi*y))
    # C term: sin(omega*ln_a)/omega = ln_a * sinc(omega*ln_a/pi) where sinc=sin(x)/x
    # B term: (1-cos(omega*ln_a))/omega = ln_a * sin(half_x) * sinc(half_x/pi)
    half_x = 0.5 * model.omega * ln_a
    # B term → 0 as omega→0 (sin(half_x) ~ half_x → 0)
    b_integral = ln_a * np.sin(half_x) * np.sinc(half_x / np.pi)
    # C term → ln_a as omega→0 (sinc(0) = 1)
    c_integral = ln_a * np.sinc(model.omega * ln_a / np.pi)

    ln_xde = (
        -3.0 * (1.0 + model.w0) * ln_a
        - 3.0 * model.B * b_integral
        - 3.0 * model.C * c_integral
    )
    return np.exp(ln_xde)


def amplitude(params: ModelAParams) -> float:
    """Derived polar amplitude A = sqrt(B^2 + C^2)."""
    return float(np.sqrt(params.B**2 + params.C**2))


def phase(params: ModelAParams) -> float:
    """Derived polar phase phi = arctan2(B, C)."""
    return float(np.arctan2(params.B, params.C))


def E_model_a(z: np.ndarray | float, cosmo: CosmoParams, model: ModelAParams) -> np.ndarray:
    z_arr = np.asarray(z, dtype=float)
    a = 1.0 / (1.0 + z_arr)
    omega_de = cosmo.resolved_omega_de()
    xde = xde_model_a(a, model)
    e2 = (
        cosmo.omega_r * (1.0 + z_arr) ** 4
        + cosmo.omega_m * (1.0 + z_arr) ** 3
        + cosmo.omega_k * (1.0 + z_arr) ** 2
        + omega_de * xde
    )
    return np.sqrt(e2)


def H_model_a(z: np.ndarray | float, cosmo: CosmoParams, model: ModelAParams) -> np.ndarray:
    return cosmo.h0 * E_model_a(z, cosmo, model)