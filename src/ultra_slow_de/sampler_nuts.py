"""NUTS sampler for Model A cosmology via BlackJAX.

Uses No-U-Turn Sampling (NUTS) with dual-averaging step-size and window-based
mass-matrix adaptation to efficiently explore the Model A posterior.  The key
advantage over random-walk MH is that NUTS uses gradient information computed
by JAX autodiff, which resolves the slow mixing in the degenerate ``omega``
direction near ``B ≈ C ≈ 0``.

Public entry point
------------------
:func:`run_nuts_model_a` — mirrors the signature of
:func:`~.sampler.run_mcmc_multichain` and returns a
:class:`~.sampler.MultiChainMCMCResult`.

The log-posterior is built in *unconstrained* space using logit/logit-inverse
bijections for each bounded parameter so that NUTS can move freely without
hard reflections.

Physics assumptions (match production run)
------------------------------------------
* Flat geometry: Ω_k = 0
* Fixed radiation: Ω_r = _OMEGA_R_H2 / h² (Planck 2018 photons + 3 neutrinos; negligible for z<3.5 SN/BAO, matters
  for D_M(z*) in the CMB prior at z~10³)
* Ω_de = 1 − Ω_m − Ω_r

"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import gc

import jax
import jax.numpy as jnp
import numpy as np

# Planck 2018 h²Ω_r (photons + 3 neutrinos, N_eff=3.046)
_OMEGA_R_H2: float = 4.18e-5

try:
    from tqdm import tqdm as _tqdm_bar  # type: ignore[import]
except ImportError:  # pragma: no cover
    class _tqdm_bar:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n: int = 1): pass
        def set_postfix(self, **kw): pass

try:
    import blackjax  # type: ignore[import]
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "BlackJAX is required for the NUTS sampler.  "
        "Install with: pip install blackjax"
    ) from _exc

from .sampler import MultiChainMCMCResult, _aggregate_ess, _rhat_split

# Enable 64-bit precision in JAX globally (required for cosmological accuracy)
jax.config.update("jax_enable_x64", True)

C_KM_S: float = 299792.458  # km/s — speed of light

# ---------------------------------------------------------------------------
# Bounds for Model A parameters:  [h0, omega_m, w0, B, C, omega]
# ---------------------------------------------------------------------------
_BOUNDS_NP = np.array(
    [
        [60.0, 80.0],   # h0
        [0.1,  0.5],    # omega_m
        [-1.5, -0.5],   # w0
        [-0.3,  0.3],   # B
        [-0.3,  0.3],   # C
        [0.1,   5.0],   # omega
    ],
    dtype=np.float64,
)
PARAM_NAMES_A = ["h0", "omega_m", "w0", "B", "C", "omega"]
N_PARAMS_A = 6

_LO = jnp.asarray(_BOUNDS_NP[:, 0])
_HI = jnp.asarray(_BOUNDS_NP[:, 1])
_WIDTH = _HI - _LO


# ---------------------------------------------------------------------------
# Logit / sigmoid bijections (bounded ↔ unconstrained)
# ---------------------------------------------------------------------------

def _theta_to_psi(theta: jax.Array) -> jax.Array:
    """Bounded θ ∈ [lo, hi]  →  unconstrained ψ = logit((θ−lo)/width)."""
    p = jnp.clip((theta - _LO) / _WIDTH, 1e-7, 1.0 - 1e-7)
    return jnp.log(p / (1.0 - p))


def _psi_to_theta(psi: jax.Array) -> jax.Array:
    """Unconstrained ψ  →  bounded θ = lo + width · sigmoid(ψ)."""
    return _LO + _WIDTH * jax.nn.sigmoid(psi)


def _log_jacobian_psi_to_theta(psi: jax.Array) -> jax.Array:
    """log |∂θ/∂ψ| = Σ_i [log σ(ψ_i) + log(1−σ(ψ_i)) + log width_i]."""
    sig = jax.nn.sigmoid(psi)
    return jnp.sum(jnp.log(sig) + jnp.log1p(-sig) + jnp.log(_WIDTH))


# ---------------------------------------------------------------------------
# JAX-differentiable Model A physics
# ---------------------------------------------------------------------------

def _xde_a_jax(ln_a: jax.Array, w0: float, B: float, C: float, omega: float) -> jax.Array:
    """X_de(a) for Model A — numerically stable, JAX-differentiable.

    For w(a) = w0 + B sin(ω ln a) + C cos(ω ln a):
      ln X_de = −3(1+w0) ln_a − 3B · b_int − 3C · c_int
    where
      b_int = ln_a · sin(ω ln_a / 2) · sinc(ω ln_a / (2π))  [= (1−cos)/ω]
      c_int = ln_a · sinc(ω ln_a / π)                         [= sin(ω ln_a)/ω]

    numpy sinc convention: sinc(y) = sin(πy)/(πy), so sinc(x/π) = sin(x)/x.
    This form is continuously extendable to ω = 0 and is automatically smooth.
    """
    half_x = 0.5 * omega * ln_a
    b_int = ln_a * jnp.sin(half_x) * jnp.sinc(half_x / jnp.pi)
    c_int = ln_a * jnp.sinc(omega * ln_a / jnp.pi)
    ln_xde = (
        -3.0 * (1.0 + w0) * ln_a
        - 3.0 * B * b_int
        - 3.0 * C * c_int
    )
    return jnp.exp(ln_xde)


def _hz_grid_a_jax(
    h0: float,
    omega_m: float,
    w0: float,
    B: float,
    C: float,
    omega: float,
    z_grid: jax.Array,
) -> jax.Array:
    """H(z) on a fixed z grid for Model A with standard fixed radiation density.

    Omega_r = _OMEGA_R_H2 / (H0/100)^2  (Planck 2018: photons + 3 neutrinos).
    At z < 3.5, radiation is negligible (<0.1%); the term matters only for
    the CMB distance integral at z ~ 10^3.
    """
    a_grid = 1.0 / (1.0 + z_grid)
    ln_a = jnp.log(jnp.clip(a_grid, 1e-10, None))
    # Fixed radiation density (JAX-traceable: depends on H0 sample)
    omega_r = _OMEGA_R_H2 / (h0 / 100.0) ** 2
    omega_de = 1.0 - omega_m - omega_r
    xde = _xde_a_jax(ln_a, w0, B, C, omega)
    e2 = (
        omega_r * (1.0 + z_grid) ** 4
        + omega_m * (1.0 + z_grid) ** 3
        + omega_de * xde
    )
    return h0 * jnp.sqrt(jnp.clip(e2, 1e-20, None))


def _comoving_dist_jax(z_grid: jax.Array, h_grid: jax.Array) -> jax.Array:
    """D_M(z) via trapezoidal integration of c/H(z).  Shape (n_z,)."""
    dz = jnp.diff(z_grid)
    inv_h = C_KM_S / h_grid
    mid = 0.5 * (inv_h[:-1] + inv_h[1:])
    dm = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(dz * mid)])
    return dm  # Mpc


# ---------------------------------------------------------------------------
# Pre-computed dataset state (computed once, frozen into JAX log-posterior)
# ---------------------------------------------------------------------------

@dataclass
class _PrepData:
    # Pantheon+ SN Ia --------------------------------------------------------
    z_sn: jax.Array        # (N_sn,)
    y_sn: jax.Array        # observed m_b (N_sn,)
    c_inv_sn: jax.Array    # C^{-1} (N_sn × N_sn) — pre-computed from Cholesky
    c_inv_ones_sn: jax.Array  # C^{-1} · 1  (N_sn,)
    e_sn: float            # 1^T C^{-1} 1 — scalar
    sn_const: float        # constant term: −½(logdet + (N−1)log2π + logE)

    # BAO datasets -----------------------------------------------------------
    # Each element: (z_arr, y_obs, C_inv, obs_code, ll_const)
    # obs_code: 0 = dm_rd, 1 = dh_rd, 2 = dv_rd
    bao: list[tuple[jax.Array, jax.Array, jax.Array, int, float]]

    # Integration grids ------------------------------------------------------
    rd: float              # sound horizon at drag epoch (Mpc)
    z_low_grid: jax.Array  # z ∈ [ε, 3.5] for SN + BAO
    z_cmb_grid: jax.Array  # z ∈ [ε, 1200] for Planck prior
    include_planck: bool

    # Planck compressed prior parameters ------------------------------------
    r_obs: float = 1.7502
    la_obs: float = 301.471
    sigma_r: float = 0.0046
    sigma_la: float = 0.090
    rs_star: float = 144.43  # Mpc — sound horizon at recombination
    planck_const: float = 0.0

    # fσ8 growth-rate data (optional; None when no fsig8 dataset included) ---
    fsig8_z: jax.Array | None = None      # effective redshifts
    fsig8_y: jax.Array | None = None      # observed fσ8 values
    fsig8_c_inv: jax.Array | None = None  # inverse covariance
    fsig8_sigma8_0: float = 0.811         # fiducial σ8,0 (used when float_sigma8=False)
    float_sigma8: bool = False            # if True, σ8,0 is a free nuisance parameter
    fsig8_const: float = 0.0


_OBS_CODE = {"dm_rd": 0, "dh_rd": 1, "dv_rd": 2}


def _prepare(
    datasets: list[tuple[Any, str]],
    rd: float,
    include_planck: bool = True,
    n_z_low: int = 400,
    n_z_cmb: int = 10000,
    float_sigma8: bool = False,
) -> _PrepData:
    """Pre-compute all dataset-specific matrices.  Called once per model fit."""
    # Split SN vs BAO
    sn_ds = None
    bao_list: list[tuple[jax.Array, jax.Array, jax.Array, int, float]] = []

    fsig8_z_np = None
    fsig8_y_np = None
    fsig8_c_inv_np = None
    fsig8_const = 0.0

    for ds, obs in datasets:
        obs_lo = obs.lower()
        if obs_lo == "mb":
            sn_ds = ds
        elif obs_lo == "fsig8":
            from scipy.linalg import cho_factor, cho_solve
            cho_f = cho_factor(ds.cov, lower=True, check_finite=False)
            fsig8_z_np = np.asarray(ds.z, dtype=np.float64)
            fsig8_y_np = np.asarray(ds.y_obs, dtype=np.float64)
            fsig8_c_inv_np = cho_solve(cho_f, np.eye(len(ds.y_obs)), check_finite=False)
            L_f = np.tril(cho_f[0])
            logdet_f = 2.0 * float(np.sum(np.log(np.diag(L_f))))
            fsig8_const = -0.5 * (logdet_f + len(ds.y_obs) * np.log(2.0 * np.pi))
        else:
            code = _OBS_CODE.get(obs_lo, 0)
            from scipy.linalg import cho_factor, cho_solve
            cho = cho_factor(ds.cov, lower=True, check_finite=False)
            c_inv = cho_solve(cho, np.eye(len(ds.y_obs)), check_finite=False)
            L_b = np.tril(cho[0])
            logdet_b = 2.0 * float(np.sum(np.log(np.diag(L_b))))
            ll_const_b = -0.5 * (logdet_b + len(ds.y_obs) * np.log(2.0 * np.pi))
            bao_list.append((
                jnp.asarray(ds.z, dtype=jnp.float64),
                jnp.asarray(ds.y_obs, dtype=jnp.float64),
                jnp.asarray(c_inv, dtype=jnp.float64),
                code,
                ll_const_b,
            ))

    if sn_ds is None:
        raise ValueError("No SN Ia dataset (observable='mb') found in datasets.")

    # SN Ia pre-factorisation
    from scipy.linalg import cho_factor, cho_solve
    n_sn = len(sn_ds.y_obs)
    cho_sn = cho_factor(sn_ds.cov, lower=True, check_finite=False)
    c_inv_sn_np = cho_solve(cho_sn, np.eye(n_sn), check_finite=False)
    ones = np.ones(n_sn)
    c_inv_ones = c_inv_sn_np @ ones
    e_val = float(ones @ c_inv_ones)

    # Stable log-determinant via Cholesky diagonal
    import scipy.linalg as _sla
    L_sn = np.tril(cho_sn[0])
    logdet_sn = 2.0 * float(np.sum(np.log(np.diag(L_sn))))
    sn_const = -0.5 * (logdet_sn + (n_sn - 1) * np.log(2.0 * np.pi) + np.log(e_val))

    # Integration grids
    z_low = np.linspace(1e-4, 3.5, n_z_low)
    z_cmb = np.linspace(1e-4, 1200.0, n_z_cmb)
    planck_logdet = np.log(0.0046**2 * 0.090**2)
    planck_const = -0.5 * (planck_logdet + 2.0 * np.log(2.0 * np.pi))

    return _PrepData(
        z_sn=jnp.asarray(sn_ds.z, dtype=jnp.float64),
        y_sn=jnp.asarray(sn_ds.y_obs, dtype=jnp.float64),
        c_inv_sn=jnp.asarray(c_inv_sn_np, dtype=jnp.float64),
        c_inv_ones_sn=jnp.asarray(c_inv_ones, dtype=jnp.float64),
        e_sn=e_val,
        sn_const=sn_const,
        bao=bao_list,
        rd=rd,
        z_low_grid=jnp.asarray(z_low, dtype=jnp.float64),
        z_cmb_grid=jnp.asarray(z_cmb, dtype=jnp.float64),
        include_planck=include_planck,
        planck_const=planck_const,
        fsig8_z=jnp.asarray(fsig8_z_np, dtype=jnp.float64) if fsig8_z_np is not None else None,
        fsig8_y=jnp.asarray(fsig8_y_np, dtype=jnp.float64) if fsig8_y_np is not None else None,
        fsig8_c_inv=jnp.asarray(fsig8_c_inv_np, dtype=jnp.float64) if fsig8_c_inv_np is not None else None,
        float_sigma8=(float_sigma8 and fsig8_z_np is not None),
        fsig8_const=fsig8_const,
    )


# ---------------------------------------------------------------------------
# JAX growth-factor helper (for fσ8 likelihood)
# ---------------------------------------------------------------------------

def _fsig8_jax(
    z_eff: jax.Array,
    h_low: jax.Array,
    z_low: jax.Array,
    h0: "jax.Array | float",
    omega_m: "jax.Array | float",
    sigma8_0: "jax.Array | float" = 0.811,
    gamma: float = 0.55,
) -> jax.Array:
    """Predict fσ8(z) using JAX.

    Growth factor D(z)/D(0) via the Heath-Carroll integral (trapezoidal):
        D(z) ∝ H(z) ∫_z^∞ (1+z') / H(z')³ dz'
    Extended above z_low[-1] (≈3.5) using matter-domination H ∝ (1+z)^1.5.
    Growth-rate approximation: f(z) = [Ω_m(z)]^γ  (γ=0.55).
    """
    # Extend the grid to z_ext with matter-domination tail
    z_top = z_low[-1]
    h_top = h_low[-1]
    n_ext = 100
    z_ext = jnp.linspace(z_top, 10.0, n_ext)
    h_ext = h_top * ((1.0 + z_ext) / (1.0 + z_top)) ** 1.5

    # Full grid (drop duplicate at z_top)
    z_full = jnp.concatenate([z_low, z_ext[1:]])
    h_full = jnp.concatenate([h_low, h_ext[1:]])

    # Integrand (1+z)/H^3
    integrand = (1.0 + z_full) / h_full ** 3
    dz = jnp.diff(z_full)
    mid = 0.5 * (integrand[:-1] + integrand[1:])

    # Cumulative integral from z_full[-1] back to each z (right-to-left)
    cum_rtol = jnp.concatenate([jnp.cumsum((mid * dz)[::-1])[::-1], jnp.array([0.0])])

    # D(0) ∝ H(z=0) * ∫_0^∞  — use the first grid point ≈ 0
    h0_grid = h_full[0]
    D0 = h0_grid * cum_rtol[0]

    # D(z_eff)/D(0)
    int_from_z = jnp.interp(z_eff, z_full, cum_rtol)
    h_at_z = jnp.interp(z_eff, z_full, h_full)
    D_ratio = h_at_z * int_from_z / D0

    # Ω_m(z) = omega_m * (1+z)^3 / E^2(z)
    e2_at_z = (h_at_z / h0) ** 2
    omega_m_z = omega_m * (1.0 + z_eff) ** 3 / jnp.clip(e2_at_z, 1e-20, None)
    f_z = jnp.power(jnp.clip(omega_m_z, 1e-10, None), gamma)

    return sigma8_0 * D_ratio * f_z


# ---------------------------------------------------------------------------
# JAX log-posterior builder
# ---------------------------------------------------------------------------

def _make_logpost_jax(prep: _PrepData):
    """Return a JAX-JIT-compiled log-posterior in *unconstrained* ψ space.

    The returned function maps ψ ∈ ℝ⁶  →  log p(θ(ψ) | data) + log|Jac|.
    Including the log-Jacobian ensures the transformed posterior is proper and
    that NUTS adapts step-sizes correctly.
    """
    # Freeze all dataset data as Python-level closures so JAX can trace them
    # as compile-time constants (no dynamic dispatch needed in hot path).
    z_sn = prep.z_sn
    y_sn = prep.y_sn
    c_inv_sn = prep.c_inv_sn
    c_inv_ones_sn = prep.c_inv_ones_sn
    e_sn = prep.e_sn
    sn_const = prep.sn_const
    bao = prep.bao
    z_low = prep.z_low_grid
    z_cmb = prep.z_cmb_grid
    rd = prep.rd
    include_planck = prep.include_planck
    r_obs = prep.r_obs
    la_obs = prep.la_obs
    sigma_r = prep.sigma_r
    sigma_la = prep.sigma_la
    rs_star = prep.rs_star
    planck_const = prep.planck_const
    fsig8_z = prep.fsig8_z
    fsig8_y = prep.fsig8_y
    fsig8_c_inv = prep.fsig8_c_inv
    fsig8_sigma8_0 = prep.fsig8_sigma8_0
    fsig8_const = prep.fsig8_const
    has_fsig8 = fsig8_z is not None
    _float_sigma8 = prep.float_sigma8
    # Planck 2018 TT,TE,EE+lowE σ8,0 prior (used when float_sigma8=True)
    _sigma8_prior_mean = 0.811
    _sigma8_prior_sigma = 0.006

    def _ll_theta(theta: jax.Array) -> jax.Array:
        """Log-likelihood at bounded θ = [h0, Ωm, w0, B, C, ω] (+ σ8,0 if floated)."""
        h0 = theta[0]
        omega_m = theta[1]
        w0 = theta[2]
        B = theta[3]
        C = theta[4]
        omega = theta[5]

        # ── H(z) and D_M(z) on the low-z grid ────────────────────────────
        h_low = _hz_grid_a_jax(h0, omega_m, w0, B, C, omega, z_low)
        dm_low = _comoving_dist_jax(z_low, h_low)

        # ── Pantheon+ SN Ia log-likelihood (M_B marginalised) ─────────────
        # μ(z) = 5 log10(D_L / 10 pc);  D_L = D_M · (1+z)
        dm_sn = jnp.interp(z_sn, z_low, dm_low)
        dl_sn = dm_sn * (1.0 + z_sn)
        mu_model = 5.0 * jnp.log10(jnp.clip(dl_sn * 1e6 / 10.0, 1e-30, None))
        delta = y_sn - mu_model
        c_inv_d = c_inv_sn @ delta
        a_val = delta @ c_inv_d
        b_val = c_inv_ones_sn @ delta
        chi2_min = a_val - b_val * b_val / e_sn
        ll = sn_const - 0.5 * chi2_min

        # ── BAO log-likelihoods ───────────────────────────────────────────
        # Python-level loop is fine: unrolled at JIT trace time (fixed number
        # of datasets) so each dataset contributes its own fused kernel.
        for z_b, y_b, c_inv_b, obs_code, ll_const_b in bao:
            dm_b = jnp.interp(z_b, z_low, dm_low)
            h_b = jnp.interp(z_b, z_low, h_low)
            dm_rd_pred = dm_b / rd
            dh_rd_pred = (C_KM_S / h_b) / rd
            # D_V = (z · D_M² · c/H)^{1/3}
            dv_rd_pred = (z_b * dm_b ** 2 * (C_KM_S / h_b)) ** (1.0 / 3.0) / rd
            # Select observable using Python obs_code (known at trace time)
            if obs_code == 1:
                y_pred = dh_rd_pred
            elif obs_code == 2:
                y_pred = dv_rd_pred
            else:
                y_pred = dm_rd_pred
            r_b = y_b - y_pred
            ll = ll + ll_const_b + (-0.5 * (r_b @ (c_inv_b @ r_b)))

        # ── Planck compressed priors ──────────────────────────────────────
        if include_planck:
            z_star = 1089.92
            h_cmb = _hz_grid_a_jax(h0, omega_m, w0, B, C, omega, z_cmb)
            dm_cmb = _comoving_dist_jax(z_cmb, h_cmb)
            dm_star = jnp.interp(jnp.array([z_star]), z_cmb, dm_cmb)[0]
            r_pred = jnp.sqrt(omega_m) * dm_star * h0 / C_KM_S
            la_pred = jnp.pi * dm_star / rs_star
            ll = ll + planck_const + (
                -0.5 * (r_pred - r_obs) ** 2 / sigma_r ** 2
                + -0.5 * (la_pred - la_obs) ** 2 / sigma_la ** 2
            )

        # ── fσ8 growth-rate data (optional) ───────────────────────────────
        if has_fsig8:
            assert fsig8_z is not None
            assert fsig8_y is not None
            assert fsig8_c_inv is not None
            # σ8,0: either the last element of theta (nuisance) or frozen constant
            sigma8_now = theta[6] if _float_sigma8 else fsig8_sigma8_0
            fsig8_pred = _fsig8_jax(
                fsig8_z, h_low, z_low, h0, omega_m, sigma8_now
            )
            r_f = fsig8_y - fsig8_pred
            ll = ll + fsig8_const + (-0.5 * (r_f @ (fsig8_c_inv @ r_f)))
            # Gaussian prior on σ8,0 when floating
            if _float_sigma8:
                ll = ll + (
                    -0.5 * ((sigma8_now - _sigma8_prior_mean) / _sigma8_prior_sigma) ** 2
                )

        return ll

    @jax.jit
    def logpost_psi(psi: jax.Array) -> jax.Array:
        """Log-posterior in unconstrained ψ space (includes log-Jacobian)."""
        theta = _psi_to_theta(psi)
        return _ll_theta(theta) + _log_jacobian_psi_to_theta(psi)

    return logpost_psi


# ---------------------------------------------------------------------------
# JAX H(z) implementations for ΛCDM and Model B
# ---------------------------------------------------------------------------

def _hz_grid_lcdm_jax(
    h0: jax.Array,
    omega_m: jax.Array,
    z_grid: jax.Array,
) -> jax.Array:
    """H(z) for flat ΛCDM with standard fixed radiation density (Planck 2018)."""
    omega_r = _OMEGA_R_H2 / (h0 / 100.0) ** 2
    omega_de = 1.0 - omega_m - omega_r
    e2 = (
        omega_r * (1.0 + z_grid) ** 4
        + omega_m * (1.0 + z_grid) ** 3
        + omega_de
    )
    return h0 * jnp.sqrt(jnp.clip(e2, 1e-20, None))


def _hz_grid_b_jax(
    h0: jax.Array,
    omega_m: jax.Array,
    mu: jax.Array,
    z_grid: jax.Array,
    n_ode_steps: int = 300,
) -> jax.Array:
    """H(z) for Model B (quadratic-potential quintessence) via JAX RK4 scan.

    Uses a fixed-step RK4 integrator (jax.lax.scan) to integrate the
    scalar-field ODE from N_init ≈ -ln(1+10) to N=0.  Ω_r = 0 as in
    the production run.
    """
    omega_de = 1.0 - omega_m

    # Initial conditions (ΛCDM-like frozen field)
    phi_i = jnp.sqrt(6.0 * omega_de) / mu
    dphi_i = jnp.array(0.0, dtype=jnp.float64)

    n_init = -jnp.log(jnp.array(11.0, dtype=jnp.float64))  # z_init = 10
    n_final = jnp.array(0.0, dtype=jnp.float64)
    h_step = (n_final - n_init) / n_ode_steps

    def rhs(n_val, y):
        phi, dphi = y[0], y[1]
        e_n = jnp.exp(-n_val)
        rho_m = omega_m * e_n ** 3
        vt = mu ** 2 * phi ** 2 / 6.0
        dvt = mu ** 2 * phi / 3.0
        safe_denom = jnp.clip(1.0 - dphi ** 2 / 6.0, 1e-10, None)
        e2 = (rho_m + vt) / safe_denom
        eps = 3.0 * rho_m / (2.0 * e2) + dphi ** 2 / 2.0
        ddphi = -(3.0 - eps) * dphi - 3.0 * dvt / jnp.clip(e2, 1e-20, None)
        return jnp.array([dphi, ddphi])

    def rk4_step(carry, _):
        n_val, y = carry
        k1 = rhs(n_val, y)
        k2 = rhs(n_val + 0.5 * h_step, y + 0.5 * h_step * k1)
        k3 = rhs(n_val + 0.5 * h_step, y + 0.5 * h_step * k2)
        k4 = rhs(n_val + h_step, y + h_step * k3)
        y_new = y + (h_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        n_new = n_val + h_step
        return (n_new, y_new), (n_new, y_new)

    y0 = jnp.array([phi_i, dphi_i])
    (_, _), (n_traj, y_traj) = jax.lax.scan(
        rk4_step, (n_init, y0), None, length=n_ode_steps
    )
    # n_traj: (n_ode_steps,), y_traj: (n_ode_steps, 2)

    phi_traj = y_traj[:, 0]
    dphi_traj = y_traj[:, 1]

    # E^2(N) from ODE trajectory
    e_n_traj = jnp.exp(-n_traj)
    rho_m_traj = omega_m * e_n_traj ** 3
    vt_traj = mu ** 2 * phi_traj ** 2 / 6.0
    safe_denom_traj = jnp.clip(1.0 - dphi_traj ** 2 / 6.0, 1e-10, None)
    e2_traj = (rho_m_traj + vt_traj) / safe_denom_traj

    # Prepend the initial point
    n_all = jnp.concatenate([jnp.array([n_init]), n_traj])
    e2_init = (omega_m * jnp.exp(-3.0 * n_init) + mu ** 2 * phi_i ** 2 / 6.0) / jnp.clip(
        1.0 - dphi_i ** 2 / 6.0, 1e-10, None
    )
    e2_all = jnp.concatenate([jnp.array([e2_init]), e2_traj])

    # Interpolate E^2 onto the requested z_grid using N = -ln(1+z)
    n_query = -jnp.log(jnp.clip(1.0 + z_grid, 1e-10, None))
    e2_at_z = jnp.interp(n_query, n_all, e2_all)

    return h0 * jnp.sqrt(jnp.clip(e2_at_z, 1e-20, None))


# ---------------------------------------------------------------------------
# Generic JAX log-posterior builder (all models share the same SN/BAO/Planck
# likelihood structure; only H(z) differs)
# ---------------------------------------------------------------------------

def _make_logpost_jax_generic(
    prep: _PrepData,
    hz_fn,
    bounds_np: np.ndarray,
    use_lcdm_for_planck: bool = False,
):
    """Build an unconstrained JAX log-posterior for any model.

    Parameters
    ----------
    prep : _PrepData — precomputed dataset matrices from :func:`_prepare`.
    hz_fn : callable(theta_bounded, z_grid) -> H(z) array
        JAX-traceable function returning H(z) values in km/s/Mpc.
        ``theta_bounded`` is the full parameter vector in bounded space.
    bounds_np : (n_params, 2) float64 array of prior bounds.

    Returns
    -------
    logpost_psi : callable(psi) -> scalar
        Log-posterior in unconstrained ψ-space (includes log-Jacobian).
    """
    lo = jnp.asarray(bounds_np[:, 0])
    hi = jnp.asarray(bounds_np[:, 1])
    width = hi - lo

    def _theta_to_psi_g(theta: jax.Array) -> jax.Array:
        p = jnp.clip((theta - lo) / width, 1e-7, 1.0 - 1e-7)
        return jnp.log(p / (1.0 - p))

    def _psi_to_theta_g(psi: jax.Array) -> jax.Array:
        return lo + width * jax.nn.sigmoid(psi)

    def _log_jac_g(psi: jax.Array) -> jax.Array:
        sig = jax.nn.sigmoid(psi)
        return jnp.sum(jnp.log(sig) + jnp.log1p(-sig) + jnp.log(width))

    # Freeze prep fields in closures
    z_sn = prep.z_sn; y_sn = prep.y_sn
    c_inv_sn = prep.c_inv_sn; c_inv_ones_sn = prep.c_inv_ones_sn
    e_sn = prep.e_sn; sn_const = prep.sn_const
    bao = prep.bao; z_low = prep.z_low_grid; z_cmb = prep.z_cmb_grid
    rd = prep.rd; include_planck = prep.include_planck
    r_obs = prep.r_obs; la_obs = prep.la_obs
    sigma_r = prep.sigma_r; sigma_la = prep.sigma_la; rs_star = prep.rs_star
    planck_const = prep.planck_const
    fsig8_z_g = prep.fsig8_z
    fsig8_y_g = prep.fsig8_y
    fsig8_c_inv_g = prep.fsig8_c_inv
    fsig8_sigma8_0_g = prep.fsig8_sigma8_0
    fsig8_const_g = prep.fsig8_const
    has_fsig8_g = fsig8_z_g is not None
    _float_sigma8_g = prep.float_sigma8
    # Planck 2018 TT,TE,EE+lowE σ8,0 prior
    _sigma8_prior_mean_g = 0.811
    _sigma8_prior_sigma_g = 0.006

    def _ll_theta(theta: jax.Array) -> jax.Array:
        h_low = hz_fn(theta, z_low)
        dm_low = _comoving_dist_jax(z_low, h_low)

        # SN Ia
        dm_sn = jnp.interp(z_sn, z_low, dm_low)
        dl_sn = dm_sn * (1.0 + z_sn)
        mu_model = 5.0 * jnp.log10(jnp.clip(dl_sn * 1e6 / 10.0, 1e-30, None))
        delta = y_sn - mu_model
        c_inv_d = c_inv_sn @ delta
        a_val = delta @ c_inv_d
        b_val = c_inv_ones_sn @ delta
        chi2_min = a_val - b_val * b_val / e_sn
        ll = sn_const - 0.5 * chi2_min

        # BAO
        for z_b, y_b, c_inv_b, obs_code, ll_const_b in bao:
            dm_b = jnp.interp(z_b, z_low, dm_low)
            h_b = jnp.interp(z_b, z_low, h_low)
            dm_rd_pred = dm_b / rd
            dh_rd_pred = (C_KM_S / h_b) / rd
            dv_rd_pred = (z_b * dm_b ** 2 * (C_KM_S / h_b)) ** (1.0 / 3.0) / rd
            if obs_code == 1:
                y_pred = dh_rd_pred
            elif obs_code == 2:
                y_pred = dv_rd_pred
            else:
                y_pred = dm_rd_pred
            r_b = y_b - y_pred
            ll = ll + ll_const_b + (-0.5 * (r_b @ (c_inv_b @ r_b)))

        # Planck
        if include_planck:
            z_star = 1089.92
            h_cmb = (
                _hz_grid_lcdm_jax(theta[0], theta[1], z_cmb)
                if use_lcdm_for_planck
                else hz_fn(theta, z_cmb)
            )
            dm_cmb = _comoving_dist_jax(z_cmb, h_cmb)
            dm_star = jnp.interp(jnp.array([z_star]), z_cmb, dm_cmb)[0]
            h0 = theta[0]
            r_pred = jnp.sqrt(theta[1]) * dm_star * h0 / C_KM_S
            la_pred = jnp.pi * dm_star / rs_star
            ll = ll + planck_const + (
                -0.5 * (r_pred - r_obs) ** 2 / sigma_r ** 2
                + -0.5 * (la_pred - la_obs) ** 2 / sigma_la ** 2
            )

        # fσ8 growth-rate data (optional)
        if has_fsig8_g:
            assert fsig8_z_g is not None
            assert fsig8_y_g is not None
            assert fsig8_c_inv_g is not None
            # σ8,0: last element of theta (nuisance) or frozen constant
            sigma8_now_g = theta[-1] if _float_sigma8_g else fsig8_sigma8_0_g
            fsig8_pred = _fsig8_jax(
                fsig8_z_g, h_low, z_low, theta[0], theta[1], sigma8_now_g
            )
            r_f = fsig8_y_g - fsig8_pred
            ll = ll + fsig8_const_g + (-0.5 * (r_f @ (fsig8_c_inv_g @ r_f)))
            # Gaussian prior on σ8,0 when floating
            if _float_sigma8_g:
                ll = ll + (
                    -0.5 * ((sigma8_now_g - _sigma8_prior_mean_g) / _sigma8_prior_sigma_g) ** 2
                )

        return ll

    def logpost_psi(psi: jax.Array) -> jax.Array:
        theta = _psi_to_theta_g(psi)
        return _ll_theta(theta) + _log_jac_g(psi)

    return logpost_psi, _psi_to_theta_g, _theta_to_psi_g


# ---------------------------------------------------------------------------
# Generic NUTS runner (ΛCDM, Model A, Model B — any model with JAX H(z))
# ---------------------------------------------------------------------------

def run_nuts_generic(
    model_label: str,
    n_params: int,
    hz_fn,
    bounds_np: np.ndarray,
    param_names: list[str],
    datasets: list[tuple[Any, str]],
    theta0: np.ndarray,
    rd: float = 147.09,
    include_planck: bool = True,
    n_chains: int = 4,
    n_warmup: int = 1000,
    n_samples: int = 4000,
    seed: int = 12345,
    init_scatter: float = 0.5,
    float_sigma8: bool = True,
    use_lcdm_for_planck: bool = False,
) -> MultiChainMCMCResult:
    """Generic NUTS runner for any model with a JAX-traceable H(z) function.

    When fsig8 data is present and ``float_sigma8=True`` (default), σ8,0 is
    added as a nuisance parameter with a Planck Gaussian prior
    N(0.811, 0.006²), stopping the 'fixed σ8 artefact' objection.
    """
    jax.config.update("jax_enable_x64", True)

    # Work with mutable copies so we can extend them without side-effects
    bounds_np = np.array(bounds_np, dtype=np.float64)
    param_names = list(param_names)
    theta0 = np.asarray(theta0, dtype=np.float64)

    print(f"  [NUTS] Preparing datasets for {model_label} "
          f"({len(datasets)} datasets)...", end="", flush=True)
    t0 = time.perf_counter()
    prep = _prepare(datasets, rd, include_planck, float_sigma8=float_sigma8)

    # If σ8,0 is treated as a free parameter, extend bounds / names / theta0
    if prep.float_sigma8:
        bounds_np = np.vstack([bounds_np, [[0.70, 0.95]]])
        param_names = param_names + ["sigma8_0"]
        theta0 = np.append(theta0, 0.811)   # initialise at Planck mean
        n_params = n_params + 1
        print(f"\n  [NUTS] sigma8_0 floated as nuisance parameter "
              f"(Planck prior N(0.811, 0.006^2))")

    logpost, psi_to_theta, theta_to_psi = _make_logpost_jax_generic(
        prep, hz_fn, bounds_np, use_lcdm_for_planck=use_lcdm_for_planck
    )

    # Warm up JIT compilation
    psi_test = theta_to_psi(jnp.asarray(theta0, dtype=jnp.float64))
    _ = logpost(psi_test)
    _ = jax.grad(logpost)(psi_test)
    print(f" done ({time.perf_counter() - t0:.1f}s, incl. JIT compile)")

    psi0 = np.asarray(theta_to_psi(jnp.asarray(theta0, dtype=jnp.float64)))

    lo = bounds_np[:, 0]; hi = bounds_np[:, 1]; width = hi - lo

    def _log_jac(psi_np):
        psi = jnp.asarray(psi_np)
        sig = jax.nn.sigmoid(psi)
        return float(jnp.sum(jnp.log(sig) + jnp.log1p(-sig) + jnp.log(jnp.asarray(width))))

    all_chains = np.empty((n_chains, n_samples, n_params), dtype=np.float64)
    all_ll = np.empty((n_chains, n_samples), dtype=np.float64)
    accept_rates = np.zeros(n_chains, dtype=np.float64)

    for c in range(n_chains):
        rng_np = np.random.default_rng(seed + c * 1000)
        jitter = rng_np.normal(0.0, init_scatter, size=n_params)
        psi_init = jnp.asarray(psi0 + jitter, dtype=jnp.float64)
        key = jax.random.PRNGKey(seed + c * 1000)

        warmup = blackjax.window_adaptation(blackjax.nuts, logpost)
        key, warmup_key = jax.random.split(key)
        print(f"  [NUTS] Chain {c + 1}/{n_chains}: warmup ({n_warmup} steps)...",
              end="", flush=True)
        t_wu = time.perf_counter()
        (state, params), _ = warmup.run(warmup_key, psi_init, n_warmup)
        print(f" done ({time.perf_counter() - t_wu:.1f}s). "
              f"step_size={float(params['step_size']):.4f}")

        nuts_adapted = blackjax.nuts(logpost, **params)

        def _one_step(state, rng_key):
            state, info = nuts_adapted.step(rng_key, state)
            return state, (state.position, info.acceptance_rate)

        key, sample_key = jax.random.split(key)
        sample_keys = jax.random.split(sample_key, n_samples)
        chunk_size = min(100, n_samples)
        n_chunks = n_samples // chunk_size
        remainder = n_samples % chunk_size
        all_pos_chunks: list[np.ndarray] = []
        all_acc_chunks: list[np.ndarray] = []
        cur_state = state

        t_s = time.perf_counter()
        with _tqdm_bar(
            total=n_samples,
            desc=f"  [NUTS] {model_label} chain {c + 1}/{n_chains}",
            unit="step",
            ascii=True,
            leave=True,
        ) as pbar:
            for i in range(n_chunks):
                ck = sample_keys[i * chunk_size:(i + 1) * chunk_size]
                cur_state, (chunk_pos, chunk_acc) = jax.lax.scan(
                    _one_step, cur_state, ck
                )
                all_pos_chunks.append(np.asarray(chunk_pos))
                all_acc_chunks.append(np.asarray(chunk_acc))
                pbar.update(chunk_size)
                pbar.set_postfix(acc=f"{float(np.mean(np.asarray(chunk_acc))):.3f}")
            if remainder > 0:
                rk = sample_keys[n_chunks * chunk_size:]
                cur_state, (r_pos, r_acc) = jax.lax.scan(_one_step, cur_state, rk)
                all_pos_chunks.append(np.asarray(r_pos))
                all_acc_chunks.append(np.asarray(r_acc))
                pbar.update(remainder)

        positions_psi = np.concatenate(all_pos_chunks, axis=0)
        acc_arr = np.concatenate(all_acc_chunks, axis=0)
        mean_acc = float(np.mean(acc_arr))
        print(f"  [NUTS] {model_label} chain {c + 1}/{n_chains}: done "
              f"({time.perf_counter() - t_s:.1f}s). accept_rate={mean_acc:.3f}",
              flush=True)

        # ψ → θ
        positions_jax = jnp.asarray(positions_psi, dtype=jnp.float64)
        theta_samples = np.asarray(jax.vmap(psi_to_theta)(positions_jax))

        # log-likelihood (strip Jacobian from logpost)
        jac_arr = np.array([_log_jac(positions_psi[i]) for i in range(n_samples)])
        logpost_arr = np.asarray(jax.vmap(logpost)(positions_jax))
        ll_samples = logpost_arr - jac_arr

        all_chains[c] = theta_samples
        all_ll[c] = ll_samples
        accept_rates[c] = mean_acc

        # Free JAX/GPU memory before allocating the next chain's compiled kernels.
        del cur_state, positions_psi, positions_jax, theta_samples
        del logpost_arr, jac_arr, acc_arr, warmup, nuts_adapted
        del all_pos_chunks, all_acc_chunks
        gc.collect()
        try:
            jax.clear_caches()  # JAX ≥0.4.1 — frees XLA compilation cache
        except AttributeError:
            pass

    rhat = _rhat_split(all_chains) if n_chains >= 2 and n_samples >= 4 else None
    ess = _aggregate_ess(all_chains) if n_samples >= 4 else None

    device_str = "cpu"
    try:
        _devs = jax.devices()
        if any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in _devs):
            device_str = "gpu"
    except Exception:
        pass

    return MultiChainMCMCResult(
        chains=all_chains,
        loglike=all_ll,
        accept_rates=accept_rates,
        param_names=param_names,
        rhat_per_param=rhat,
        ess_per_param=ess,
        proposal_sigma_final=None,
        backend="blackjax-nuts",
        device=device_str,
        used_fallback=False,
    )


# ---------------------------------------------------------------------------
# Convenience wrappers for ΛCDM and Model B
# ---------------------------------------------------------------------------

def run_nuts_lcdm(
    datasets: list[tuple[Any, str]],
    theta0: np.ndarray,
    rd: float = 147.09,
    include_planck: bool = True,
    n_chains: int = 4,
    n_warmup: int = 800,
    n_samples: int = 2000,
    seed: int = 12345,
    init_scatter: float = 0.5,
    float_sigma8: bool = True,
) -> MultiChainMCMCResult:
    """NUTS for flat ΛCDM, theta = [h0, omega_m] (+ sigma8_0 nuisance if fsig8 data present)."""
    bounds = np.array([[60.0, 80.0], [0.1, 0.5]], dtype=np.float64)

    def hz_fn(theta, z_grid):
        return _hz_grid_lcdm_jax(theta[0], theta[1], z_grid)

    return run_nuts_generic(
        model_label="LCDM",
        n_params=2,
        hz_fn=hz_fn,
        bounds_np=bounds,
        param_names=["h0", "omega_m"],
        datasets=datasets,
        theta0=theta0,
        rd=rd,
        include_planck=include_planck,
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        seed=seed,
        init_scatter=init_scatter,
        float_sigma8=float_sigma8,
        use_lcdm_for_planck=False,
    )


def run_nuts_model_b(
    datasets: list[tuple[Any, str]],
    theta0: np.ndarray,
    rd: float = 147.09,
    include_planck: bool = True,
    n_chains: int = 4,
    n_warmup: int = 800,
    n_samples: int = 2000,
    seed: int = 12345,
    init_scatter: float = 0.3,
    float_sigma8: bool = True,
) -> MultiChainMCMCResult:
    """NUTS for Model B (quadratic quintessence), theta = [h0, omega_m, mu] (+ sigma8_0 if fsig8)."""
    bounds = np.array([[60.0, 80.0], [0.1, 0.5], [0.01, 5.0]], dtype=np.float64)

    def hz_fn(theta, z_grid):
        return _hz_grid_b_jax(theta[0], theta[1], theta[2], z_grid)

    return run_nuts_generic(
        model_label="ModelB",
        n_params=3,
        hz_fn=hz_fn,
        bounds_np=bounds,
        param_names=["h0", "omega_m", "mu"],
        datasets=datasets,
        theta0=theta0,
        rd=rd,
        include_planck=include_planck,
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        seed=seed,
        init_scatter=init_scatter,
        float_sigma8=float_sigma8,
        use_lcdm_for_planck=True,
    )


# ---------------------------------------------------------------------------
# NUTS runner
# ---------------------------------------------------------------------------

def run_nuts_model_a(
    datasets: list[tuple[Any, str]],
    theta0: np.ndarray,
    rd: float = 147.09,
    include_planck: bool = True,
    n_chains: int = 4,
    n_warmup: int = 1000,
    n_samples: int = 4000,
    seed: int = 12345,
    init_scatter: float = 0.5,
    float_sigma8: bool = True,
) -> MultiChainMCMCResult:
    """Run NUTS for Model A using BlackJAX with window-based adaptation.

    Parameters
    ----------
    datasets : list of (GaussianDataset, observable_name) pairs.
        Must include exactly one SN Ia dataset (observable='mb') and any
        number of BAO datasets.
    theta0 : shape (6,) initial bounded parameter vector
        ``[h0, omega_m, w0, B, C, omega]``.
    rd : sound horizon at drag epoch (Mpc).
    include_planck : include Planck 2018 compressed distance priors.
    n_chains : number of independent NUTS chains.
    n_warmup : warm-up steps per chain (used for step-size + mass-matrix
        adaptation; **not** stored in the returned chains).
    n_samples : post-warmup samples per chain.
    seed : base RNG seed; chain ``c`` uses ``seed + c * 1000``.
    init_scatter : Gaussian jitter (in ψ-space) added to the initial position
        for each chain to disperse starting points.

    Returns
    -------
    MultiChainMCMCResult
        Compatible with the rest of the pipeline.  ``chains`` has shape
        ``(n_chains, n_samples, 6)`` (or 7 if sigma8_0 is floated) in the
        original bounded parameter space.
    """
    def hz_fn(theta, z_grid):
        return _hz_grid_a_jax(
            theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], z_grid
        )

    return run_nuts_generic(
        model_label="ModelA",
        n_params=N_PARAMS_A,
        hz_fn=hz_fn,
        bounds_np=_BOUNDS_NP.copy(),
        param_names=list(PARAM_NAMES_A),
        datasets=datasets,
        theta0=theta0,
        rd=rd,
        include_planck=include_planck,
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        seed=seed,
        init_scatter=init_scatter,
        float_sigma8=float_sigma8,
        use_lcdm_for_planck=False,
    )
