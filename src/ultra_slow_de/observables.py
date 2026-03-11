import numpy as np
from scipy.integrate import cumulative_trapezoid, quad

from .constants import C_KM_S


def luminosity_distance_flat(z: np.ndarray, h_of_z: np.ndarray) -> np.ndarray:
    z_arr = np.asarray(z, dtype=float)
    h_arr = np.asarray(h_of_z, dtype=float)
    inv_h = 1.0 / h_arr
    integral = cumulative_trapezoid(inv_h, z_arr, initial=0.0)
    return (1.0 + z_arr) * C_KM_S * integral


def deceleration_parameter(z: np.ndarray, h_of_z: np.ndarray) -> np.ndarray:
    z_arr = np.asarray(z, dtype=float)
    h_arr = np.asarray(h_of_z, dtype=float)
    dh_dz = np.gradient(h_arr, z_arr)
    return (1.0 + z_arr) * dh_dz / h_arr - 1.0


# ---------------------------------------------------------------------------
# BAO observables
# ---------------------------------------------------------------------------
# All BAO distances are in units of r_d (the sound-horizon scale at drag
# epoch).  The caller must supply r_d in Mpc.

def comoving_distance_flat(z: np.ndarray, h_of_z: np.ndarray) -> np.ndarray:
    """D_M(z) = c ∫₀ᶻ dz'/H(z')  for flat geometry, in Mpc."""
    z_arr = np.asarray(z, dtype=float)
    h_arr = np.asarray(h_of_z, dtype=float)
    inv_h = 1.0 / h_arr
    return C_KM_S * cumulative_trapezoid(inv_h, z_arr, initial=0.0)


def _h_at_z_interp(z_target: float, z_grid: np.ndarray,
                    h_grid: np.ndarray) -> float:
    """Linearly interpolate H on a grid to get H(z_target)."""
    return float(np.interp(z_target, z_grid, h_grid))


def dm_over_rd(z_eff: np.ndarray, z_grid: np.ndarray,
               h_grid: np.ndarray, rd: float) -> np.ndarray:
    """D_M(z_eff) / r_d evaluated by integrating H(z) on z_grid."""
    dm_grid = comoving_distance_flat(z_grid, h_grid)
    dm_at_zeff = np.interp(np.asarray(z_eff), z_grid, dm_grid)
    return dm_at_zeff / rd


def dh_over_rd(z_eff: np.ndarray, z_grid: np.ndarray,
               h_grid: np.ndarray, rd: float) -> np.ndarray:
    """D_H(z_eff) / r_d  where D_H = c / H(z)."""
    h_at_z = np.interp(np.asarray(z_eff), z_grid, h_grid)
    return C_KM_S / (h_at_z * rd)


def dv_over_rd(z_eff: np.ndarray, z_grid: np.ndarray,
               h_grid: np.ndarray, rd: float) -> np.ndarray:
    """D_V(z_eff) / r_d  (spherically averaged BAO distance)."""
    dm = dm_over_rd(z_eff, z_grid, h_grid, rd) * rd     # back to Mpc
    dh = C_KM_S / np.interp(np.asarray(z_eff), z_grid, h_grid)  # Mpc
    z_arr = np.asarray(z_eff, dtype=float)
    dv = (z_arr * dm**2 * dh) ** (1.0 / 3.0)
    return dv / rd


def distance_modulus(z: np.ndarray, h_of_z: np.ndarray) -> np.ndarray:
    """μ(z) = 5 log₁₀(D_L / 10 pc), with D_L in Mpc → pc conversion."""
    dl_mpc = luminosity_distance_flat(z, h_of_z)
    dl_pc = dl_mpc * 1e6  # Mpc → pc
    with np.errstate(divide="ignore"):
        mu = 5.0 * np.log10(dl_pc / 10.0)
    return mu


# ---------------------------------------------------------------------------
# Growth rate and fσ₈
# ---------------------------------------------------------------------------

def growth_factor_ratio(
    z_eff: np.ndarray,
    z_grid: np.ndarray,
    h_grid: np.ndarray,
    z_max_int: float = 10.0,
) -> np.ndarray:
    """D(z)/D(0): linear growth-factor ratio via the Heath–Carroll integral.

    For a flat FRW universe with arbitrary dark energy:
        D(z) ∝ H(z) ∫_z^∞ (1+z') / H(z')³  dz'

    H(z) is interpolated from the supplied grid; for z > z_grid[-1] a
    matter-dominated extrapolation H ∝ (1+z)^(3/2) is used (dark energy is
    completely negligible above z ~ 3).

    Parameters
    ----------
    z_eff     : effective redshifts at which D/D(0) is required
    z_grid    : fine z-array from the Hubble-function solver
    h_grid    : H(z) values on z_grid  [km/s/Mpc]
    z_max_int : upper integration limit (default 10; tail beyond this is ~0)
    """
    from scipy.integrate import quad
    from scipy.interpolate import interp1d

    z_g = np.asarray(z_grid, dtype=float)
    h_g = np.asarray(h_grid, dtype=float)
    z_eff_arr = np.atleast_1d(np.asarray(z_eff, dtype=float))

    h_interp = interp1d(z_g, h_g, kind="linear", fill_value="extrapolate",
                        bounds_error=False)
    z_max_g = float(z_g[-1])
    h_max_g = float(h_g[-1])

    def _h_ext(zp: float) -> float:
        if zp <= z_max_g:
            return float(h_interp(zp))
        return h_max_g * ((1.0 + zp) / (1.0 + z_max_g)) ** 1.5

    def _integrand(zp: float) -> float:
        hp = _h_ext(zp)
        return (1.0 + zp) / hp ** 3

    z_upper = max(z_max_int, z_max_g)
    norm, _ = quad(_integrand, 0.0, z_upper, limit=300)
    h0 = float(h_g[0])
    D0 = h0 * norm

    ratios = np.empty(len(z_eff_arr), dtype=float)
    for i, zi in enumerate(z_eff_arr):
        hi = _h_ext(float(zi))
        integral_i, _ = quad(_integrand, float(zi), z_upper, limit=300)
        ratios[i] = hi * integral_i / D0

    return ratios


def fsig8_pred(
    z_eff: np.ndarray,
    z_grid: np.ndarray,
    h_grid: np.ndarray,
    omega_m: float,
    sigma8_0: float = 0.811,
    gamma: float = 0.55,
) -> np.ndarray:
    """Predict fσ₈(z) using the growth-rate γ-approximation.

    f(z) = [Ω_m(z)]^γ  (Linder & Cahn 2007; γ=0.55 for flat w-DE)
    σ₈(z) = σ₈,0 × D(z)/D(0)

    This is a background-level sanity check: σ₈,0 is fixed to the Planck 2018
    TT,TE,EE+lowE value (0.811) and not sampled.  Perturbation-level clustering
    effects (dark-energy sound speed, quintessence clustering) are not modelled.

    Parameters
    ----------
    z_eff    : effective redshifts of the fσ₈ observations
    z_grid   : fine z-array from the Hubble solver
    h_grid   : H(z) on z_grid  [km/s/Mpc]
    omega_m  : Ω_m,0 (current matter density parameter)
    sigma8_0 : σ₈(z=0); default is Planck 2018 best-fit 0.811
    gamma    : growth-rate index; default 0.55
    """
    z_eff_arr = np.atleast_1d(np.asarray(z_eff, dtype=float))
    h0 = float(h_grid[0])
    h_at_z = np.interp(z_eff_arr, z_grid, h_grid)

    # Ω_m(z) = Ω_m,0 H₀² (1+z)³ / H(z)²
    omega_m_z = omega_m * h0 ** 2 * (1.0 + z_eff_arr) ** 3 / h_at_z ** 2

    f_z = omega_m_z ** gamma
    D_ratio = growth_factor_ratio(z_eff_arr, z_grid, h_grid)
    return f_z * sigma8_0 * D_ratio