import numpy as np

from .baseline_lcdm import H_lcdm
from .datasets import GaussianDataset
from .growth_backend import predict_fsig8
from .likelihood import dataset_loglike, gaussian_loglike, sn_loglike_marg, SNLikelihoodCached
from .model_a import H_model_a
from .model_b import H_model_b, ModelBParams
from .observables import (
    comoving_distance_flat,
    dh_over_rd,
    dm_over_rd,
    distance_modulus,
    dv_over_rd,
    luminosity_distance_flat,
)
from .params import CosmoParams, ModelAParams

# Planck 2018 h²Ω_r: photons + 3 neutrino species (N_eff=3.046)
# Fixed early-universe constant used for D_M(z*) in the Planck prior.
_OMEGA_R_H2: float = 4.18e-5

# Default dense z-grid for integrating BAO distances
# 400 points is more than sufficient for sub-0.1% accuracy in all distance observables
_N_ZGRID = 400
_Z_MAX_GRID = 3.5


def _h_on_grid(
    cosmo: CosmoParams,
    model_a: ModelAParams | None,
    model_b: ModelBParams | None,
    z_max: float = _Z_MAX_GRID,
    n_z: int = _N_ZGRID,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (z_grid, H_grid) on a fine mesh for integration."""
    z_grid = np.linspace(0.0, z_max, n_z)
    if model_b is not None:
        h_grid = H_model_b(z_grid, cosmo.h0, cosmo.omega_m, cosmo.omega_r, model_b)
    elif model_a is not None:
        h_grid = H_model_a(z_grid, cosmo, model_a)
    else:
        h_grid = H_lcdm(z_grid, cosmo)
    return z_grid, h_grid


def predict_observable(
    z: np.ndarray,
    observable: str,
    cosmo: CosmoParams,
    model_a: ModelAParams | None = None,
    model_b: ModelBParams | None = None,
    rd: float = 147.09,
    _precomputed_grid: tuple[np.ndarray, np.ndarray] | None = None,
    growth_likelihood_mode: str = "exploratory_gamma",
    growth_backend: str = "auto",
) -> np.ndarray:
    """Predict an observable for ΛCDM, Model A, or Model B.

    Supported observables:
      'h'      — H(z) in km/s/Mpc
      'dl'     — luminosity distance D_L(z) in Mpc
      'mu'     — distance modulus μ(z)
      'dm_rd'  — D_M(z) / r_d  (comoving transverse distance / sound horizon)
      'dh_rd'  — D_H(z) / r_d  (Hubble distance / sound horizon)
      'dv_rd'  — D_V(z) / r_d  (volume-averaged distance / sound horizon)
      'fsig8'  — fσ₈(z) growth-rate sanity check (σ₈,0 = 0.811 fixed)

    For BAO observables, z values are treated as effective redshifts and
    H(z) is integrated on a dense internal grid.

    _precomputed_grid : optional (z_grid, h_grid) to skip ODE re-evaluation.
        Used by joint_logposterior to share one grid across all datasets.
    """
    if model_a is not None and model_b is not None:
        raise ValueError("Provide model_a or model_b, not both.")

    obs = observable.lower()

    # Direct H(z)
    if obs == "h":
        if model_b is not None:
            return H_model_b(z, cosmo.h0, cosmo.omega_m, cosmo.omega_r, model_b)
        if model_a is not None:
            return H_model_a(z, cosmo, model_a)
        return H_lcdm(z, cosmo)

    # Luminosity distance — needs integration from z=0
    if obs == "dl":
        z_grid, h_grid = _precomputed_grid if _precomputed_grid is not None else _h_on_grid(cosmo, model_a, model_b)
        dl_grid = luminosity_distance_flat(z_grid, h_grid)
        return np.interp(np.asarray(z), z_grid, dl_grid)

    # Distance modulus — interpolate d_L, then convert (avoids -inf at z=0)
    if obs == "mu":
        z_grid, h_grid = _precomputed_grid if _precomputed_grid is not None else _h_on_grid(cosmo, model_a, model_b)
        dl_grid = luminosity_distance_flat(z_grid, h_grid)
        dl_at_z = np.interp(np.asarray(z), z_grid, dl_grid)
        dl_pc = dl_at_z * 1e6
        with np.errstate(divide="ignore"):
            return 5.0 * np.log10(dl_pc / 10.0)

    # BAO distances — need integration on a fine grid
    if obs in ("dm_rd", "dh_rd", "dv_rd"):
        z_grid, h_grid = _precomputed_grid if _precomputed_grid is not None else _h_on_grid(cosmo, model_a, model_b)
        if obs == "dm_rd":
            return dm_over_rd(z, z_grid, h_grid, rd)
        if obs == "dh_rd":
            return dh_over_rd(z, z_grid, h_grid, rd)
        return dv_over_rd(z, z_grid, h_grid, rd)

    # Growth rate × σ₈
    if obs == "fsig8":
        z_grid, h_grid = _precomputed_grid if _precomputed_grid is not None else _h_on_grid(cosmo, model_a, model_b)
        pred = predict_fsig8(
            z_eff=z,
            z_grid=z_grid,
            h_grid=h_grid,
            omega_m0=cosmo.omega_m,
            sigma8_0=0.811,
            mode=growth_likelihood_mode,
            backend=growth_backend,
        )
        return pred.values

    raise ValueError(
        f"Unsupported observable '{observable}'. "
        "Use 'h', 'dl', 'mu', 'dm_rd', 'dh_rd', 'dv_rd', or 'fsig8'."
    )


def loglike_for_dataset(
    ds: GaussianDataset,
    observable: str,
    cosmo: CosmoParams,
    model_a: ModelAParams | None = None,
    model_b: ModelBParams | None = None,
    rd: float = 147.09,
    sn_cache: SNLikelihoodCached | None = None,
    _precomputed_grid: tuple[np.ndarray, np.ndarray] | None = None,
    growth_likelihood_mode: str = "exploratory_gamma",
    growth_backend: str = "auto",
) -> float:
    """Compute log-likelihood for a single dataset.

    For SN Ia datasets (observable='mb'), uses analytical M_B
    marginalisation.  Pass ``sn_cache`` for O(N^2) repeated evaluation.
    For all other observables, uses the standard Gaussian log-likelihood.
    Pass ``_precomputed_grid`` to reuse a shared H(z) grid across datasets.
    """
    obs = observable.lower()
    if obs == "mb":
        # SN Ia: predict μ(z), marginalise over M_B
        mu_model = predict_observable(ds.z, "mu", cosmo, model_a, model_b, rd,
                                      _precomputed_grid=_precomputed_grid)
        if sn_cache is not None:
            return sn_cache.loglike(mu_model)
        return sn_loglike_marg(ds, mu_model)

    y_model = predict_observable(ds.z, observable=observable, cosmo=cosmo,
                                 model_a=model_a, model_b=model_b, rd=rd,
                                 _precomputed_grid=_precomputed_grid,
                                 growth_likelihood_mode=growth_likelihood_mode,
                                 growth_backend=growth_backend)
    return dataset_loglike(ds, y_model)


def loglike_planck_compressed(
    cosmo: CosmoParams,
    model_a: ModelAParams | None = None,
    model_b: ModelBParams | None = None,
    rs_star: float = 144.43,
    R_obs: float = 1.7502,
    lA_obs: float = 301.471,
    sigma_R: float = 0.0046,
    sigma_lA: float = 0.090,
) -> float:
    """Gaussian log-likelihood for Planck 2018 compressed distance priors.

    R  = √(Ω_m) · (H₀/c) · D_M(z*)
    lA = π · D_M(z*) / r_s(z*)

    where z* ≈ 1089.92 and r_s(z*) ≈ 144.43 Mpc (sound horizon at
    recombination, distinct from the drag-epoch r_d used for BAO).
    """
    z_star = 1089.92
    # For CMB distances at z~1090, dark energy is completely negligible; use
    # background ΛCDM H(z).  Include standard radiation density so that
    # D_M(z*), R, and ℓ_A are evaluated with correct early-universe physics.
    # Ω_r = h²Ω_r / h²  where h = H₀/100 (varies per parameter sample).
    omega_r = _OMEGA_R_H2 / (cosmo.h0 / 100.0) ** 2
    from .params import CosmoParams as _CP
    cosmo_rad = _CP(h0=cosmo.h0, omega_m=cosmo.omega_m, omega_r=omega_r)
    z_grid, h_grid = _h_on_grid(cosmo_rad, None, None, z_max=1200.0, n_z=10000)
    from .observables import comoving_distance_flat
    dm_grid = comoving_distance_flat(z_grid, h_grid)
    dm_star = float(np.interp(z_star, z_grid, dm_grid))  # Mpc

    r_pred = np.sqrt(cosmo.omega_m) * dm_star * cosmo.h0 / C_KM_S
    la_pred = np.pi * dm_star / rs_star

    residual = np.array([r_pred - R_obs, la_pred - lA_obs])
    cov = np.diag([sigma_R**2, sigma_lA**2])
    return float(gaussian_loglike(residual, cov))


def joint_logposterior(
    datasets: list[tuple[GaussianDataset, str]],
    cosmo: CosmoParams,
    model_a: ModelAParams | None = None,
    model_b: ModelBParams | None = None,
    rd: float = 147.09,
    include_planck: bool = False,
    sn_cache: SNLikelihoodCached | None = None,
    growth_likelihood_mode: str = "exploratory_gamma",
    growth_backend: str = "auto",
) -> float:
    """Combined log-posterior from SN + BAO + (optionally) Planck priors.

    Parameters
    ----------
    datasets : list of (GaussianDataset, observable_name) pairs.
        observable_name is passed to ``loglike_for_dataset``.
        Use 'mb' for Pantheon+ data, 'dm_rd'/'dh_rd'/'dv_rd' for BAO.
    cosmo, model_a, model_b : model parameters.
    rd : sound horizon at drag epoch in Mpc.
    include_planck : if True, add Planck compressed distance priors.
    sn_cache : optional pre-factorised SN covariance for fast evaluation.
    """
    # Pre-compute H(z) grid once for this parameter point.
    # This avoids redundant ODE solves (especially for Model B) across datasets.
    shared_grid: tuple[np.ndarray, np.ndarray] | None = None
    _first_obs_needs_grid = any(
        obs.lower() not in ("h",)
        for _, obs in datasets
    )
    if _first_obs_needs_grid:
        shared_grid = _h_on_grid(cosmo, model_a, model_b)

    ll = sum(
        loglike_for_dataset(ds, obs, cosmo, model_a, model_b, rd,
                            sn_cache=sn_cache, _precomputed_grid=shared_grid,
                            growth_likelihood_mode=growth_likelihood_mode,
                            growth_backend=growth_backend)
        for ds, obs in datasets
    )
    if include_planck:
        ll += loglike_planck_compressed(cosmo, model_a, model_b)
    return float(ll)


# Re-export for convenience
from .constants import C_KM_S