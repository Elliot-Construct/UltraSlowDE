from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from .observables import fsig8_pred


@dataclass(frozen=True)
class GrowthPrediction:
    values: np.ndarray
    backend_used: str
    exploratory: bool
    metadata: dict[str, Any]


def _fiducial_omega_b0() -> float:
    # Minimal fiducial baryon fraction used when only Ωm and H0 are supplied.
    return 0.049


def _predict_fsig8_class(
    z_eff: np.ndarray,
    h0: float,
    omega_m0: float,
    sigma8_0: float,
) -> np.ndarray:
    from classy import Class  # type: ignore

    z_eff_arr = np.asarray(z_eff, dtype=float)
    z_max = float(max(np.max(z_eff_arr), 0.0)) + 0.5
    h = float(h0) / 100.0
    omega_b0 = _fiducial_omega_b0()
    omega_cdm0 = max(float(omega_m0) - omega_b0, 1e-5)

    cosmo = Class()
    cosmo.set(
        {
            "h": h,
            "Omega_b": omega_b0,
            "Omega_cdm": omega_cdm0,
            "A_s": 2.1e-9,
            "n_s": 0.965,
            "output": "mPk",
            "P_k_max_1/Mpc": 5.0,
            "z_max_pk": max(z_max, 2.0),
        }
    )
    cosmo.compute()
    try:
        d0 = float(cosmo.scale_independent_growth_factor(0.0))
        fs8_vals = []
        for zi in z_eff_arr:
            d_z = float(cosmo.scale_independent_growth_factor(float(zi)))
            f_z = float(cosmo.scale_independent_growth_factor_f(float(zi)))
            sigma8_z = float(sigma8_0) * d_z / max(d0, 1e-30)
            fs8_vals.append(f_z * sigma8_z)
        return np.asarray(fs8_vals, dtype=float)
    finally:
        cosmo.struct_cleanup()
        cosmo.empty()


def _predict_fsig8_camb(
    z_eff: np.ndarray,
    h0: float,
    omega_m0: float,
    sigma8_0: float,
) -> np.ndarray:
    import camb  # type: ignore
    from camb import model as camb_model  # type: ignore

    z_eff_arr = np.asarray(z_eff, dtype=float)
    z_grid = np.unique(np.concatenate([z_eff_arr, np.array([0.0])]))
    # CAMB typically expects redshifts high->low
    z_grid = np.sort(z_grid)[::-1]

    h = float(h0) / 100.0
    omega_b0 = _fiducial_omega_b0()
    omega_cdm0 = max(float(omega_m0) - omega_b0, 1e-5)

    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=float(h0),
        ombh2=omega_b0 * h * h,
        omch2=omega_cdm0 * h * h,
        mnu=0.06,
        omk=0.0,
        tau=0.054,
    )
    pars.InitPower.set_params(As=2.1e-9, ns=0.965)
    pars.set_dark_energy(w=-1.0)
    pars.set_matter_power(redshifts=list(z_grid), kmax=2.0)
    pars.NonLinear = camb_model.NonLinear_none

    results = camb.get_results(pars)
    fs8_grid = np.asarray(results.get_fsigma8(), dtype=float)
    # get_fsigma8 follows the same ordering as requested redshifts
    # Rescale to requested sigma8_0 (CAMB normalization is set by As above).
    sigma8_grid = np.asarray(results.get_sigma8(), dtype=float)
    sigma8_z0 = float(sigma8_grid[-1]) if len(sigma8_grid) > 0 else 1.0
    scale = float(sigma8_0) / max(sigma8_z0, 1e-30)
    fs8_grid_scaled = fs8_grid * scale
    return np.interp(z_eff_arr, z_grid[::-1], fs8_grid_scaled[::-1])


def backend_availability(backend: str = "auto") -> dict[str, Any]:
    """Return runtime availability and selection metadata for growth backends."""
    backend_l = backend.lower()
    if backend_l not in ("auto", "class", "camb"):
        raise ValueError("backend must be one of {'auto','class','camb'}")

    attempted: list[str] = []
    class_available = False
    camb_available = False

    if backend_l in ("auto", "class"):
        attempted.append("class")
        try:
            import classy  # type: ignore  # noqa: F401

            class_available = True
        except Exception:
            class_available = False

    if backend_l in ("auto", "camb"):
        attempted.append("camb")
        try:
            import camb  # type: ignore  # noqa: F401

            camb_available = True
        except Exception:
            camb_available = False

    if backend_l == "class":
        resolved = "class" if class_available else "class_unavailable"
    elif backend_l == "camb":
        resolved = "camb" if camb_available else "camb_unavailable"
    else:
        if class_available:
            resolved = "class"
        elif camb_available:
            resolved = "camb"
        else:
            resolved = "internal_perturbation_ode"

    return {
        "backend_requested": backend_l,
        "backend_attempted": attempted,
        "class_available": class_available,
        "camb_available": camb_available,
        "backend_resolved": resolved,
    }


def _linear_growth_ode_fsig8(
    z_eff: np.ndarray,
    z_grid: np.ndarray,
    h_grid: np.ndarray,
    omega_m0: float,
    sigma8_0: float,
) -> np.ndarray:
    """Compute fσ8(z) from linear-growth ODE on a supplied background H(z) grid.

    Solves
      D'' + [2 + dlnH/dlna] D' - 3/2 Ω_m(a) D = 0
    in ln(a), where primes denote d/dln(a).
    """
    z_g = np.asarray(z_grid, dtype=float)
    h_g = np.asarray(h_grid, dtype=float)
    z_eff_arr = np.asarray(z_eff, dtype=float)

    a_g = 1.0 / (1.0 + z_g)
    e_g = h_g / h_g[0]
    lna_g = np.log(a_g)
    order = np.argsort(lna_g)
    lna_sorted = lna_g[order]
    e_sorted = e_g[order]

    dlnh_dlna = np.gradient(np.log(np.clip(e_sorted, 1e-30, None)), lna_sorted)

    def _rhs(lna: float, y: np.ndarray) -> np.ndarray:
        d = y[0]
        dd_lna = y[1]
        e_now = float(np.interp(lna, lna_sorted, e_sorted))
        dlnh = float(np.interp(lna, lna_sorted, dlnh_dlna))
        a_now = float(np.exp(lna))
        omega_m_a = omega_m0 * a_now ** (-3.0) / max(e_now**2, 1e-20)
        d2 = -(2.0 + dlnh) * dd_lna + 1.5 * omega_m_a * d
        return np.array([dd_lna, d2], dtype=float)

    lna_init = lna_sorted[0]
    lna_final = lna_sorted[-1]
    a_init = np.exp(lna_init)
    y0 = np.array([a_init, a_init], dtype=float)

    sol = solve_ivp(
        _rhs,
        (lna_init, lna_final),
        y0,
        t_eval=lna_sorted,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(f"Linear-growth ODE failed: {sol.message}")

    d_sorted = sol.y[0]
    dd_sorted = sol.y[1]
    d0 = float(np.interp(0.0, lna_sorted, d_sorted))

    z_for_interp = np.exp(-lna_sorted) - 1.0
    z_for_interp = z_for_interp[::-1]
    d_for_interp = d_sorted[::-1]
    dd_for_interp = dd_sorted[::-1]

    lna_eff = np.log(1.0 / (1.0 + z_eff_arr))
    d_eff = np.interp(z_eff_arr, z_for_interp, d_for_interp)
    dd_eff = np.interp(z_eff_arr, z_for_interp, dd_for_interp)
    f_eff = dd_eff / np.clip(d_eff, 1e-30, None)
    sigma8_eff = sigma8_0 * d_eff / max(d0, 1e-30)
    return f_eff * sigma8_eff


def predict_fsig8(
    z_eff: np.ndarray,
    z_grid: np.ndarray,
    h_grid: np.ndarray,
    omega_m0: float,
    sigma8_0: float = 0.811,
    mode: str = "exploratory_gamma",
    backend: str = "auto",
    gamma: float = 0.55,
) -> GrowthPrediction:
    """Growth backend adapter with production/exploratory modes.

    mode='production' computes fσ8 from perturbation-level linear-growth ODE.
    mode='exploratory_gamma' keeps the historical γ-approximation path.

    The backend selector accepts {'auto','class','camb'}.  CLASS/CAMB are
    attempted when requested; if unavailable, the adapter falls back to the
    internal perturbation-level ODE backend and records this in metadata.
    """
    mode_l = mode.lower()
    backend_l = backend.lower()

    if mode_l == "exploratory_gamma":
        vals = fsig8_pred(z_eff, z_grid, h_grid, omega_m0, sigma8_0=sigma8_0, gamma=gamma)
        return GrowthPrediction(
            values=np.asarray(vals, dtype=float),
            backend_used="gamma",
            exploratory=True,
            metadata={"mode": mode_l, "gamma": float(gamma)},
        )

    if mode_l != "production":
        raise ValueError("mode must be 'production' or 'exploratory_gamma'")

    status = backend_availability(backend=backend_l)
    attempted = status["backend_attempted"]
    class_available = bool(status["class_available"])
    camb_available = bool(status["camb_available"])

    h0 = float(np.asarray(h_grid, dtype=float)[0])

    if backend_l == "class" and not class_available:
        raise RuntimeError(
            "CLASS backend was explicitly requested but `classy` is unavailable in this runtime. "
            "Install CLASS/classy or choose --growth-backend auto."
        )
    if backend_l == "camb" and not camb_available:
        raise RuntimeError(
            "CAMB backend was explicitly requested but `camb` is unavailable in this runtime. "
            "Install CAMB or choose --growth-backend auto."
        )

    if backend_l in ("auto", "class") and class_available:
        try:
            vals = _predict_fsig8_class(z_eff, h0=h0, omega_m0=omega_m0, sigma8_0=sigma8_0)
            return GrowthPrediction(
                values=np.asarray(vals, dtype=float),
                backend_used="class",
                exploratory=False,
                metadata={
                    "mode": mode_l,
                    "backend_requested": backend_l,
                    "backend_attempted": attempted,
                    "class_available": class_available,
                    "camb_available": camb_available,
                },
            )
        except Exception as exc:
            if backend_l == "class":
                raise RuntimeError(f"CLASS backend failed during prediction: {exc}") from exc

    if backend_l in ("auto", "camb") and camb_available:
        try:
            vals = _predict_fsig8_camb(z_eff, h0=h0, omega_m0=omega_m0, sigma8_0=sigma8_0)
            return GrowthPrediction(
                values=np.asarray(vals, dtype=float),
                backend_used="camb",
                exploratory=False,
                metadata={
                    "mode": mode_l,
                    "backend_requested": backend_l,
                    "backend_attempted": attempted,
                    "class_available": class_available,
                    "camb_available": camb_available,
                },
            )
        except Exception as exc:
            if backend_l == "camb":
                raise RuntimeError(f"CAMB backend failed during prediction: {exc}") from exc

    vals = _linear_growth_ode_fsig8(z_eff, z_grid, h_grid, omega_m0, sigma8_0)
    return GrowthPrediction(
        values=np.asarray(vals, dtype=float),
        backend_used="internal_perturbation_ode",
        exploratory=False,
        metadata={
            "mode": mode_l,
            "backend_requested": backend_l,
            "backend_attempted": attempted,
            "class_available": class_available,
            "camb_available": camb_available,
            "fallback_reason": (
                "CLASS/CAMB unavailable in runtime environment (or auto-mode fallback); "
                "using internal perturbation-level linear-growth ODE backend"
            ),
        },
    )
