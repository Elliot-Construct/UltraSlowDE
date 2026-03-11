from __future__ import annotations

from typing import Any

import numpy as np

from .model_a import E_model_a, xde_model_a
from .model_b import ModelBParams, solve_model_b
from .params import CosmoParams, ModelAParams

Z_RECOMB = 1090.0


def _flatten_chains(chains: np.ndarray) -> np.ndarray:
    arr = np.asarray(chains, dtype=float)
    if arr.ndim != 3:
        raise ValueError("chains must have shape (n_chains, n_steps, n_params)")
    return arr.reshape(-1, arr.shape[-1])


def compute_fde_at_recombination(
    chains: np.ndarray,
    param_names: list[str],
    model: str,
    z_star: float = Z_RECOMB,
    max_samples: int = 2000,
    seed: int = 12345,
) -> dict[str, Any]:
    """Evaluate f_de(z*) over posterior samples and report consistency summary."""
    flat = _flatten_chains(np.asarray(chains, dtype=float))
    n_total = flat.shape[0]

    if n_total == 0:
        raise ValueError("No samples supplied")

    idx = np.arange(n_total)
    if n_total > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=max_samples, replace=False)
    sel = flat[idx]

    pidx = {p: i for i, p in enumerate(param_names)}
    model_l = model.lower()
    fde_vals: list[float] = []

    if model_l == "a":
        for th in sel:
            cosmo = CosmoParams(
                h0=float(th[pidx["h0"]]),
                omega_m=float(th[pidx["omega_m"]]),
            )
            ma = ModelAParams(
                w0=float(th[pidx["w0"]]),
                B=float(th[pidx["B"]]),
                C=float(th[pidx["C"]]),
                omega=float(th[pidx["omega"]]),
            )
            a_star = 1.0 / (1.0 + z_star)
            xde = float(np.asarray(xde_model_a(np.array([a_star]), ma))[0])
            e2 = float(np.asarray(E_model_a(np.array([z_star]), cosmo, ma))[0] ** 2)
            omega_de0 = cosmo.resolved_omega_de()
            fde_vals.append(float(omega_de0 * xde / max(e2, 1e-30)))
    elif model_l == "b":
        for th in sel:
            omega_m = float(th[pidx["omega_m"]])
            mb = ModelBParams(mu=float(th[pidx["mu"]]))
            sol = solve_model_b(
                np.array([z_star], dtype=float),
                omega_m=omega_m,
                omega_r=0.0,
                model=mb,
                z_init=2000.0,
            )
            fde_vals.append(float(np.asarray(sol["omega_phi"], dtype=float)[0]))
    else:
        raise ValueError("model must be 'a' or 'b'")

    fde = np.asarray(fde_vals, dtype=float)
    p95 = float(np.quantile(fde, 0.95))
    threshold = 1e-3
    return {
        "model": model_l,
        "z_star": float(z_star),
        "n_samples_total": int(n_total),
        "n_samples_used": int(fde.size),
        "fde_p95": p95,
        "fde_max": float(np.max(fde)),
        "fde_mean": float(np.mean(fde)),
        "threshold": threshold,
        "passes_threshold": bool(p95 < threshold),
    }
