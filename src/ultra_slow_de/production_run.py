"""Production MCMC driver for the ultra-slow DE paper.

Runs converged multi-chain random-walk MH sampling for:
  - ΛCDM (model='lcdm')
  - Model A (model='a')
  - Model B (model='b')

Outputs:
  - JSON summary: output/production_results.json
  - Convergence table: output/tables/convergence_diagnostics_production.md
  - Parameter constraints table: output/tables/parameter_constraints.md
  - Publication figures: output/figures/*
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    def _tqdm(it, **kwargs):  # type: ignore[misc]
        return it

from .baseline_lcdm import H_lcdm
from .builtin_data import load_all_bao
from .inference import predict_observable
from .ingest import load_pantheon_plus
from .model_a import H_model_a, w_model_a
from .model_b import H_model_b, ModelBParams, solve_model_b
from .observables import deceleration_parameter, luminosity_distance_flat
from .params import CosmoParams, ModelAParams
from .sampler import run_mcmc_multichain

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # pragma: no cover - optional runtime dependency
    HAS_MPL = False

try:
    import corner  # type: ignore

    HAS_CORNER = True
except Exception:  # pragma: no cover - optional runtime dependency
    HAS_CORNER = False

# ---------------------------------------------------------------------------
# GPU availability detection (CuPy / NVIDIA CUDA)
# ---------------------------------------------------------------------------
def _ensure_cuda_home() -> None:
    """Ensure CUDA_HOME points to an existing toolkit directory with cuBLAS.

    On Windows the installer sometimes sets CUDA_HOME to a stale version while
    CUDA_PATH points to the actual installation.  Fix this transparently so
    CuPy's DLL pathfinder succeeds.
    """
    import os
    import glob as _glob

    cuda_home = os.environ.get("CUDA_HOME", "")
    if cuda_home and os.path.isdir(cuda_home):
        # Already valid — check that cublasLt lives there
        if _glob.glob(os.path.join(cuda_home, "bin", "cublasLt*.dll")):
            return
    # Search known candidates: CUDA_PATH, then enumerate the toolkit root
    candidates: list[str] = []
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        candidates.append(cuda_path)
    toolkit_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(toolkit_root):
        candidates += sorted(
            (str(p) for p in __import__("pathlib").Path(toolkit_root).iterdir() if p.is_dir()),
            reverse=True,  # highest version first
        )
    for candidate in candidates:
        if _glob.glob(os.path.join(candidate, "bin", "cublasLt*.dll")):
            os.environ["CUDA_HOME"] = candidate
            # Prepend the bin dir so Windows DLL loader finds it too
            bin_dir = os.path.join(candidate, "bin")
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            return


_ensure_cuda_home()

try:
    import cupy as _cp  # type: ignore[import]  # noqa: F401
    _cp.zeros(1)  # Forces CUDA context init; fails fast if no GPU
    # Set the CuPy memory pool to preallocate a larger chunk up front.
    # This prevents pool-resize stalls during the MCMC hot loop which can
    # look like the GPU "stopping" mid-run.
    _pool = _cp.get_default_memory_pool()
    _pool.set_limit(fraction=0.80)   # allow CuPy up to 80% of VRAM
    # Warm up the allocator so the first real allocation is instant.
    _warmup_blk = _cp.empty(1645 * 1645, dtype=_cp.float64)
    del _warmup_blk
    _pool.free_all_blocks()
    USE_GPU = True
    del _cp, _pool
except Exception:
    USE_GPU = False


def _configure_windows_gpu_scheduler() -> None:
    """Increase the Windows TDR delay so long GPU MCMC kernels don't get
    killed by the timeout-detection-and-recovery watchdog.

    Writes HKLM\\System\\CurrentControlSet\\Control\\GraphicsDrivers\\TdrDelay
    to 60 seconds (default is 2 s).  Requires admin rights; silently skips if
    the process is unprivileged.
    """
    import sys
    if sys.platform != "win32":
        return
    try:
        import winreg  # type: ignore[import]
        key_path = r"System\CurrentControlSet\Control\GraphicsDrivers"
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            key_path,
            0,
            winreg.KEY_SET_VALUE | winreg.KEY_WOW64_64KEY,
        ) as k:
            winreg.SetValueEx(k, "TdrDelay", 0, winreg.REG_DWORD, 60)
    except (PermissionError, OSError):
        pass  # Not admin — TDR stays at default.  GPU may still drop out but
              # we now recover gracefully via the try/except in loglike().


if USE_GPU:
    _configure_windows_gpu_scheduler()


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "output"
FIG_ROOT = OUTPUT_ROOT / "figures"
TABLE_ROOT = OUTPUT_ROOT / "tables"

N_DATA = 1645
K_BY_MODEL = {"lcdm": 2, "a": 6, "b": 3}
MODEL_LABEL = {"lcdm": "LambdaCDM", "a": "ModelA", "b": "ModelB"}


@dataclass
class SamplerConfig:
    n_chains: int = 4
    n_steps_primary: int = 80_000   # ≥50k ensures adequate ESS with GPU acceleration
    n_steps_fallback: int = 20_000
    burn_frac: float = 0.25          # 25% burn-in gives more conservative discard
    adapt_steps: int = 20_000        # ~25% of primary; 800 adaptation rounds at interval=25
    adapt_interval: int = 25
    seed: int = 12345
    runtime_limit_min: float = 180.0  # allow up to 3h per model
    use_nuts: bool = False            # use BlackJAX NUTS for Model A (better omega mixing)
    enable_gpu: bool = True           # CuPy acceleration for MH path (ignored by NUTS path)
    nuts_n_warmup: int = 800         # NUTS warm-up steps per chain
    nuts_n_samples: int = 4000       # NUTS post-warm-up samples per chain


SENSITIVITY_CFG = SamplerConfig(
    n_steps_primary=10_000,
    n_steps_fallback=5_000,
    burn_frac=0.2,
    adapt_steps=2_000,
    adapt_interval=25,
    seed=12345,
    runtime_limit_min=60.0,
)


def _sampler_setup(model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    if model == "lcdm":
        return (
            np.array([67.4, 0.315]),
            np.array([[60.0, 80.0], [0.1, 0.5]]),
            np.array([0.5, 0.01]),
            ["h0", "omega_m"],
            ["U(60,80)", "U(0.1,0.5)"],
        )
    if model == "a":
        return (
            np.array([67.4, 0.315, -1.0, 0.0, 0.0, 2.0]),
            np.array(
                [
                    [60.0, 80.0],
                    [0.1, 0.5],
                    [-1.5, -0.5],
                    [-0.3, 0.3],
                    [-0.3, 0.3],
                    [0.1, 5.0],
                ]
            ),
            np.array([0.6, 0.012, 0.03, 0.01, 0.01, 0.08]),
            ["h0", "omega_m", "w0", "B", "C", "omega"],
            ["U(60,80)", "U(0.1,0.5)", "U(-1.5,-0.5)", "U(-0.3,0.3)", "U(-0.3,0.3)", "U(0.1,5)"],
        )
    if model == "b":
        return (
            np.array([67.4, 0.315, 0.3]),
            np.array([[60.0, 80.0], [0.1, 0.5], [0.01, 5.0]]),
            np.array([0.6, 0.012, 0.06]),
            ["h0", "omega_m", "mu"],
            ["U(60,80)", "U(0.1,0.5)", "U(0.01,5)"],
        )
    raise ValueError(f"Unsupported model: {model}")


def _target_accept_for_dim(n_dim: int) -> float:
    return 0.35 if n_dim <= 2 else 0.234


def _exclude_bgs_point(datasets: list[tuple[Any, str]]) -> list[tuple[Any, str]]:
    """Return datasets with the DESI BGS point removed (z=0.295 in DESI DV set)."""
    trimmed: list[tuple[Any, str]] = []
    for ds, obs in datasets:
        if getattr(ds, "name", "") in ("desi_dr1_dv", "desi_dr2_dv") and len(ds.z) >= 1:
            # Drop the first isotropic DESI point (BGS-like low-z anchor).
            keep = np.arange(len(ds.z)) != 0
            cov = ds.cov[np.ix_(keep, keep)]
            new_ds = type(ds)(
                name=f"{ds.name}_no_bgs",
                kind=ds.kind,
                z=ds.z[keep],
                y_obs=ds.y_obs[keep],
                cov=cov,
                source=ds.source,
            )
            trimmed.append((new_ds, obs))
        else:
            trimmed.append((ds, obs))
    return trimmed


def _burn_index(n_steps: int, burn_frac: float) -> int:
    return max(1, int(round(n_steps * burn_frac)))


def _credible_interval_68(x: np.ndarray) -> tuple[float, float, float]:
    q16, q50, q84 = np.quantile(x, [0.16, 0.50, 0.84])
    return float(q50), float(q16), float(q84)


def _info_criteria(logl_max: float, k: int, n_data: int = N_DATA) -> dict[str, float]:
    bic = k * np.log(n_data) - 2.0 * logl_max
    aic = 2.0 * k - 2.0 * logl_max
    denom = (n_data - k - 1)
    aicc = aic + (2.0 * k * (k + 1) / denom) if denom > 0 else float("inf")
    return {"bic": float(bic), "aic": float(aic), "aicc": float(aicc)}


def _compute_per_point_loglike(
    theta_flat: np.ndarray,
    datasets: list,
    model: str,
    rd: float,
) -> np.ndarray:
    """Return array of shape (n_samples, n_data_points) with per-point log-likelihoods.

    For each posterior sample theta_s, each dataset is Cholesky-whitened:
      epsilon_i = (L^{-1} r)_i,  where r = y - y_model
      ell_i = -0.5*epsilon_i^2 - log(L_ii) - 0.5*log(2*pi)
    so that sum_i ell_i = dataset log-likelihood.

    For SN Ia (observable='mb'): uses best-fit M_B at each sample via GLS formula,
    then whitens the centred residuals.
    """
    from scipy.linalg import cho_factor, cho_solve, solve_triangular

    # Pre-factorise each dataset's Cholesky (done once, not per sample)
    factorisations = []
    for ds, obs in datasets:
        cho = cho_factor(ds.cov, lower=True, check_finite=False)
        # Extract lower triangular L so we can whiten via L^{-1} r
        L_lower = np.tril(cho[0])
        log_L_diag = np.log(np.diag(L_lower))  # diag(L) > 0 by construction
        c_inv_ones = cho_solve(cho, np.ones(len(ds.y_obs)), check_finite=False)
        e_sn = float(np.ones(len(ds.y_obs)) @ c_inv_ones)  # 1^T C^{-1} 1
        factorisations.append((cho, L_lower, log_L_diag, c_inv_ones, e_sn))

    n_samples = theta_flat.shape[0]
    n_points = sum(len(ds.y_obs) for ds, _ in datasets)
    ll_matrix = np.empty((n_samples, n_points), dtype=float)

    for s, theta in enumerate(theta_flat):
        cosmo, ma, mb = _model_params_from_theta(model, theta)
        col = 0
        for (ds, obs), (cho, L_lower, log_L_diag, c_inv_ones, e_sn) in zip(datasets, factorisations):
            n = len(ds.y_obs)
            if obs.lower() == "mb":
                mu_model = predict_observable(ds.z, "mu", cosmo, ma, mb, rd)
                raw_r = ds.y_obs - mu_model
                c_inv_r = cho_solve(cho, raw_r, check_finite=False)
                mb_hat = float(np.ones(n) @ c_inv_r) / e_sn
                r = raw_r - mb_hat
            else:
                y_model = predict_observable(ds.z, obs, cosmo, ma, mb, rd)
                r = ds.y_obs - y_model

            # Whitened residuals: eps = L^{-1} r  (not C^{-1} r)
            # This ensures sum_i ell_i = log p(y|theta) exactly.
            eps = solve_triangular(L_lower, r, lower=True, check_finite=False)
            # per-point: -0.5*eps_i^2 - log(L_ii) - 0.5*log(2pi)
            ll_matrix[s, col:col + n] = -0.5 * eps**2 - log_L_diag - 0.5 * np.log(2.0 * np.pi)
            col += n

    return ll_matrix


def compute_waic(per_point_ll: np.ndarray) -> dict:
    """Compute WAIC from per-point log-likelihood matrix (n_samples x n_points).

    Returns dict with keys: 'waic', 'lppd', 'p_waic'.
    Uses log-sum-exp for numerical stability.
    """
    # lppd = sum_i log(mean_s exp(ell_{s,i}))
    # Use logsumexp: log(mean exp(x)) = logsumexp(x) - log(n)
    S = per_point_ll.shape[0]
    # per-point log(mean_s p_i): shape (n_points,)
    lse = per_point_ll.max(axis=0)  # shape (n_points,) — stabiliser
    log_mean = lse + np.log(np.mean(np.exp(per_point_ll - lse[np.newaxis, :]), axis=0))
    lppd = float(np.sum(log_mean))

    # p_WAIC = sum_i Var_s[ell_{s,i}]
    p_waic = float(np.sum(np.var(per_point_ll, axis=0, ddof=1)))

    waic = -2.0 * lppd + 2.0 * p_waic
    return {"waic": waic, "lppd": lppd, "p_waic": p_waic}


def _flatten_posteriors(chains: np.ndarray, burn: int) -> np.ndarray:
    post = chains[:, burn:, :]
    return post.reshape(-1, post.shape[-1])


def _model_params_from_theta(model: str, theta: np.ndarray) -> tuple[CosmoParams, ModelAParams | None, ModelBParams | None]:
    if model == "lcdm":
        cosmo = CosmoParams(h0=float(theta[0]), omega_m=float(theta[1]))
        return cosmo, None, None
    if model == "a":
        cosmo = CosmoParams(h0=float(theta[0]), omega_m=float(theta[1]))
        ma = ModelAParams(w0=float(theta[2]), B=float(theta[3]), C=float(theta[4]), omega=float(theta[5]))
        return cosmo, ma, None
    if model == "b":
        cosmo = CosmoParams(h0=float(theta[0]), omega_m=float(theta[1]))
        mb = ModelBParams(mu=float(theta[2]))
        return cosmo, None, mb
    raise ValueError(model)


def run_single_model(
    model: str,
    datasets: list[tuple[Any, str]],
    cfg: SamplerConfig,
    rd: float,
    run_tag: str,
) -> dict[str, Any]:
    theta0, bounds, proposal, param_names, priors = _sampler_setup(model)
    target_accept = _target_accept_for_dim(theta0.size)

    # ── NUTS path for LCDM, Model A, and Model B ──────────────────────────
    if model in ("lcdm", "a", "b") and cfg.use_nuts:
        from .sampler_nuts import run_nuts_model_a, run_nuts_model_b, run_nuts_lcdm
        if model == "a":
            _nuts_fn = run_nuts_model_a
        elif model == "b":
            _nuts_fn = run_nuts_model_b
        else:
            _nuts_fn = run_nuts_lcdm
        print(f"  [NUTS] Using BlackJAX NUTS for {MODEL_LABEL[model]}.")
        t0 = time.perf_counter()
        result = _nuts_fn(
            datasets=datasets,
            theta0=theta0,
            rd=rd,
            include_planck=True,
            n_chains=cfg.n_chains,
            n_warmup=cfg.nuts_n_warmup,
            n_samples=cfg.nuts_n_samples,
            seed=cfg.seed,
        )
        elapsed = time.perf_counter() - t0
        probe_elapsed = elapsed / max(1, cfg.n_steps_primary) * 50
        est_primary = elapsed
        used_fallback_steps = False
        n_steps_used = result.chains.shape[1]
        burn = 0  # NUTS warmup is separate; all returned samples are post-warmup
        flat = _flatten_posteriors(result.chains, burn)
        lls_all = result.loglike.reshape(-1)
        idx_max = int(np.argmax(lls_all))
        max_ll = float(lls_all[idx_max])
        best_theta = result.chains.reshape(-1, result.chains.shape[-1])[idx_max]
        chain_best = [float(np.max(result.loglike[i])) for i in range(result.loglike.shape[0])]
        posterior: dict[str, Any] = {}
        for j, p in enumerate(result.param_names):
            q50, q16, q84 = _credible_interval_68(flat[:, j])
            posterior[p] = {
                "mean": float(np.mean(flat[:, j])),
                "median": q50,
                "ci68_low": q16,
                "ci68_high": q84,
                "std": float(np.std(flat[:, j])),
                "best_fit": float(best_theta[j]),
                "prior": priors[j],
            }
        rhat_map = {}
        ess_map = {}
        if result.rhat_per_param is not None:
            rhat_map = {p: float(v) for p, v in zip(result.param_names, result.rhat_per_param)}
        if result.ess_per_param is not None:
            ess_map = {p: float(v) for p, v in zip(result.param_names, result.ess_per_param)}
        info = _info_criteria(max_ll, k=K_BY_MODEL[model], n_data=N_DATA)
        _n_waic_draws = 500
        _waic_idx = _posterior_draw_indices(flat.shape[0], _n_waic_draws, seed=cfg.seed + 99)
        _waic_theta = flat[_waic_idx]
        _per_point_ll = _compute_per_point_loglike(_waic_theta, datasets, model, rd)
        waic_result = compute_waic(_per_point_ll)
        return {
            "run_id": f"{run_tag}_{MODEL_LABEL[model]}",
            "model": model,
            "model_label": MODEL_LABEL[model],
            "rd_mpc": rd,
            "steps_requested": cfg.n_steps_primary,
            "steps_used": int(n_steps_used),
            "used_fallback_steps": used_fallback_steps,
            "runtime_seconds": float(elapsed),
            "runtime_probe_seconds": float(probe_elapsed),
            "runtime_estimate_primary_seconds": float(est_primary),
            "burn_in_steps": burn,
            "n_chains": int(result.chains.shape[0]),
            "accept_rates": [float(x) for x in result.accept_rates],
            "accept_rate_mean": float(np.mean(result.accept_rates)),
            "target_accept": 0.8,  # NUTS target
            "sampler_backend": result.backend,
            "param_names": list(result.param_names),
            "posterior": posterior,
            "split_rhat": rhat_map,
            "ess_aggregate": ess_map,
            "max_split_rhat": float(np.nanmax(result.rhat_per_param)) if result.rhat_per_param is not None else float("nan"),
            "min_ess": float(np.min(result.ess_per_param)) if result.ess_per_param is not None else float("nan"),
            "chain_best_loglike": chain_best,
            "best_loglike_found": max_ll,
            "max_loglike_all_samples": max_ll,
            "information_criteria": info,
            "waic": waic_result,
            "prior_labels": priors,
            "samples_post_burn": int(flat.shape[0]),
            "best_fit_theta": [float(x) for x in best_theta],
            "chains": result.chains,
            "loglike": result.loglike,
        }
    # ── End NUTS path (A and B) ──────────────────────────────────────────

    def _run_with_steps(n_steps: int, seed_offset: int = 0, n_chains_override: int | None = None) -> tuple[Any, float]:
        t0 = time.perf_counter()
        # Wider initial scatter for high-dim models to span the posterior better
        _init_scatter = 1.5 if theta0.size >= 5 else 0.8
        res = run_mcmc_multichain(
            datasets=datasets,
            model=model,
            theta0=theta0,
            bounds=bounds,
            proposal_sigma=proposal,
            n_steps=n_steps,
            n_chains=n_chains_override or cfg.n_chains,
            seed=cfg.seed + seed_offset,
            rd=rd,
            include_planck=True,
            adapt_proposal=True,
            adapt_steps=min(cfg.adapt_steps, n_steps),
            adapt_interval=cfg.adapt_interval,
            target_accept=target_accept,
            compute_ess=True,
            use_gpu=USE_GPU and cfg.enable_gpu,
            init_scatter=_init_scatter,
        )
        elapsed = time.perf_counter() - t0
        return res, elapsed

    # Preflight runtime estimate: one short single-chain run to project cost.
    probe_steps = min(120, max(50, cfg.adapt_interval * 2))
    _, probe_elapsed = _run_with_steps(probe_steps, seed_offset=500_000, n_chains_override=1)
    est_primary = probe_elapsed * (cfg.n_steps_primary * cfg.n_chains) / max(1, probe_steps)

    planned_steps = cfg.n_steps_primary
    if est_primary > cfg.runtime_limit_min * 60.0:
        planned_steps = cfg.n_steps_fallback

    result, elapsed = _run_with_steps(planned_steps)
    used_fallback_steps = False
    if (
        planned_steps == cfg.n_steps_primary
        and elapsed > cfg.runtime_limit_min * 60.0
        and cfg.n_steps_fallback < cfg.n_steps_primary
    ):
        result, elapsed = _run_with_steps(cfg.n_steps_fallback, seed_offset=10_000)
        used_fallback_steps = True
    elif planned_steps == cfg.n_steps_fallback:
        used_fallback_steps = True

    n_steps_used = result.chains.shape[1]
    burn = _burn_index(n_steps_used, cfg.burn_frac)
    flat = _flatten_posteriors(result.chains, burn)

    lls_all = result.loglike.reshape(-1)
    idx_max = int(np.argmax(lls_all))
    max_ll = float(lls_all[idx_max])
    best_theta = result.chains.reshape(-1, result.chains.shape[-1])[idx_max]
    chain_best = [float(np.max(result.loglike[i])) for i in range(result.loglike.shape[0])]

    posterior: dict[str, Any] = {}
    for j, p in enumerate(result.param_names):
        q50, q16, q84 = _credible_interval_68(flat[:, j])
        posterior[p] = {
            "mean": float(np.mean(flat[:, j])),
            "median": q50,
            "ci68_low": q16,
            "ci68_high": q84,
            "std": float(np.std(flat[:, j])),
            "best_fit": float(best_theta[j]),
            "prior": priors[j],
        }

    rhat_map = {}
    ess_map = {}
    if result.rhat_per_param is not None:
        rhat_map = {p: float(v) for p, v in zip(result.param_names, result.rhat_per_param)}
    if result.ess_per_param is not None:
        ess_map = {p: float(v) for p, v in zip(result.param_names, result.ess_per_param)}

    info = _info_criteria(max_ll, k=K_BY_MODEL[model], n_data=N_DATA)

    # WAIC from thinned posterior draws (≥500 samples to keep memory bounded)
    _n_waic_draws = 500
    _waic_idx = _posterior_draw_indices(flat.shape[0], _n_waic_draws, seed=cfg.seed + 99)
    _waic_theta = flat[_waic_idx]
    _waic_datasets = [(ds, obs) for ds, obs in datasets]
    _per_point_ll = _compute_per_point_loglike(_waic_theta, _waic_datasets, model, rd)
    waic_result = compute_waic(_per_point_ll)

    return {
        "run_id": f"{run_tag}_{MODEL_LABEL[model]}",
        "model": model,
        "model_label": MODEL_LABEL[model],
        "rd_mpc": rd,
        "steps_requested": cfg.n_steps_primary,
        "steps_used": int(n_steps_used),
        "used_fallback_steps": used_fallback_steps,
        "runtime_seconds": float(elapsed),
        "runtime_probe_seconds": float(probe_elapsed),
        "runtime_estimate_primary_seconds": float(est_primary),
        "burn_in_steps": burn,
        "n_chains": int(result.chains.shape[0]),
        "accept_rates": [float(x) for x in result.accept_rates],
        "accept_rate_mean": float(np.mean(result.accept_rates)),
        "target_accept": float(target_accept),
        "param_names": list(result.param_names),
        "posterior": posterior,
        "split_rhat": rhat_map,
        "ess_aggregate": ess_map,
        "max_split_rhat": float(np.nanmax(result.rhat_per_param)) if result.rhat_per_param is not None else float("nan"),
        "min_ess": float(np.min(result.ess_per_param)) if result.ess_per_param is not None else float("nan"),
        "chain_best_loglike": chain_best,
        "best_loglike_found": max_ll,
        "max_loglike_all_samples": max_ll,
        "best_fit_theta": [float(x) for x in best_theta],
        "information_criteria": info,
        "waic": waic_result,
        "samples_post_burn": int(flat.shape[0]),
        "chains": result.chains,
        "loglike": result.loglike,
    }


def _run_single_model_worker(
    model: str,
    datasets: list[tuple[Any, str]],
    cfg: SamplerConfig,
    rd: float,
    run_tag: str,
) -> tuple[str, dict[str, Any]]:
    """Top-level worker entrypoint for process-based parallel execution."""
    return model, run_single_model(
        model=model,
        datasets=datasets,
        cfg=cfg,
        rd=rd,
        run_tag=run_tag,
    )


def _run_model_batch(
    *,
    models: tuple[str, ...],
    datasets: list[tuple[Any, str]],
    cfg: SamplerConfig,
    rd: float,
    run_tag: str,
    stage_label: str,
    parallel_models: bool,
    parallel_workers: int,
) -> dict[str, dict[str, Any]]:
    """Run one model batch either sequentially or in parallel workers."""
    if not parallel_models or len(models) <= 1:
        out: dict[str, dict[str, Any]] = {}
        for model in _tqdm(models, desc=f"{stage_label} runs", unit="model"):
            print(f"\n[{stage_label}] Running {MODEL_LABEL[model]}...")
            out[model] = run_single_model(
                model=model,
                datasets=datasets,
                cfg=cfg,
                rd=rd,
                run_tag=run_tag,
            )
        return out

    worker_count = max(1, min(int(parallel_workers), len(models)))
    print(f"[{stage_label}] Parallel model execution enabled ({worker_count} workers).")

    import multiprocessing as _mp

    out_parallel: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=_mp.get_context("spawn")) as ex:
        futures = {
            ex.submit(_run_single_model_worker, model, datasets, cfg, rd, run_tag): model
            for model in models
        }
        for fut in _tqdm(as_completed(futures), total=len(futures), desc=f"{stage_label} runs", unit="model"):
            model = futures[fut]
            try:
                model_key, model_result = fut.result()
            except Exception as exc:
                for f in futures:
                    f.cancel()
                raise RuntimeError(f"[{stage_label}] {MODEL_LABEL[model]} worker failed") from exc

            out_parallel[model_key] = model_result
            print(f"[{stage_label}] Completed {MODEL_LABEL[model_key]}.")

    # Preserve canonical ordering for downstream tables/figures.
    return {m: out_parallel[m] for m in models}


def _compute_delta_metrics(results_by_model: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        m: results_by_model[m]["information_criteria"] for m in ("lcdm", "a", "b")
    }
    waic_metrics = {
        m: results_by_model[m].get("waic", {}).get("waic", float("nan"))
        for m in ("lcdm", "a", "b")
    }
    keys = ("bic", "aic", "aicc")
    mins = {k: min(metrics[m][k] for m in metrics) for k in keys}
    lcdm_ref = {k: metrics["lcdm"][k] for k in keys}
    waic_ref = waic_metrics["lcdm"]
    out: dict[str, Any] = {"raw": metrics, "waic": waic_metrics, "delta_vs_best": {}, "delta_vs_lcdm": {}}
    for m in metrics:
        out["delta_vs_best"][m] = {k: float(metrics[m][k] - mins[k]) for k in keys}
        out["delta_vs_lcdm"][m] = {k: float(metrics[m][k] - lcdm_ref[k]) for k in keys}
        out["delta_vs_lcdm"][m]["waic"] = float(waic_metrics[m] - waic_ref) if (
            np.isfinite(waic_metrics[m]) and np.isfinite(waic_ref)
        ) else float("nan")
    return out


def _format_float(x: float, ndp: int = 4) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{ndp}f}"


def write_convergence_table(path: Path, production_results: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Convergence Diagnostics (Production)",
        "",
        "| Run ID | Model | Purpose | Chains×Steps | Acceptance (mean) | Max split R-hat | Min ESS |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    details: list[str] = ["", "## Per-parameter diagnostics", ""]
    for model in ("lcdm", "a", "b"):
        r = production_results[model]
        lines.append(
            f"| {r['run_id']} | {r['model_label']} | production | {r['n_chains']}x{r['steps_used']} | "
            f"{_format_float(r['accept_rate_mean'], 3)} | {_format_float(r['max_split_rhat'], 4)} | {_format_float(r['min_ess'], 1)} |"
        )
        parts = []
        for p in r["param_names"]:
            rv = r["split_rhat"].get(p, float("nan"))
            ev = r["ess_aggregate"].get(p, float("nan"))
            parts.append(f"{p}: R-hat={_format_float(rv, 4)}, ESS={_format_float(ev, 1)}")
        details.append(f"- **{r['model_label']}** — " + ", ".join(parts))

    path.write_text("\n".join(lines + details) + "\n", encoding="utf-8")


def write_parameter_constraints_table(path: Path, production_results: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Parameter Constraints (Production)", ""]
    for model in ("lcdm", "a", "b"):
        r = production_results[model]
        lines.extend(
            [
                f"## {r['model_label']}",
                "",
                "| Parameter | Prior | Posterior Mean ± 68% CI | Best-fit |",
                "|---|---|---:|---:|",
            ]
        )
        for p in r["param_names"]:
            pr = r["posterior"][p]
            mean = pr["mean"]
            lo = pr["ci68_low"]
            hi = pr["ci68_high"]
            lines.append(
                f"| {p} | {pr['prior']} | {mean:.6g} [{lo:.6g}, {hi:.6g}] | {pr['best_fit']:.6g} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _posterior_draw_indices(n: int, n_draws: int, seed: int = 12345) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n <= n_draws:
        return np.arange(n)
    return rng.choice(n, size=n_draws, replace=False)


def _model_h_q_w_dl(model: str, theta: np.ndarray, z: np.ndarray) -> dict[str, np.ndarray]:
    cosmo, ma, mb = _model_params_from_theta(model, theta)
    if model == "lcdm":
        h = H_lcdm(z, cosmo)
        w = np.full_like(z, -1.0)
    elif model == "a":
        h = H_model_a(z, cosmo, ma or ModelAParams())
        a = 1.0 / (1.0 + z)
        w = w_model_a(a, ma or ModelAParams())
    else:
        h = H_model_b(z, cosmo.h0, cosmo.omega_m, cosmo.omega_r, mb or ModelBParams())
        w = solve_model_b(z, omega_m=cosmo.omega_m, omega_r=cosmo.omega_r, model=mb or ModelBParams())["w_phi"]
    dl = luminosity_distance_flat(z, h)
    q = deceleration_parameter(z, h)
    return {"h": h, "dl": dl, "q": q, "w": w}


def _weighted_mb_best_fit(sn_dataset: Any, mu_model: np.ndarray) -> float:
    """Best-fit SN nuisance offset M_B from generalized least squares."""
    from scipy.linalg import cho_factor, cho_solve

    y = np.asarray(sn_dataset.y_obs)
    delta = y - mu_model
    ones = np.ones_like(delta)
    cho = cho_factor(sn_dataset.cov, lower=True, check_finite=False)
    c_inv_delta = cho_solve(cho, delta, check_finite=False)
    c_inv_ones = cho_solve(cho, ones, check_finite=False)
    b = float(ones @ c_inv_delta)
    e = float(ones @ c_inv_ones)
    return b / e


def generate_publication_figures(
    production_results: dict[str, dict[str, Any]],
    sn_dataset: Any,
    out_dir: Path,
    z_max: float = 2.5,
    n_z: int = 350,
    n_draws: int = 250,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not HAS_MPL:
        return []

    z = np.linspace(1e-4, z_max, n_z)
    model_order = ("lcdm", "a", "b")
    colors = {"lcdm": "k", "a": "C0", "b": "C1"}
    labels = {"lcdm": r"$\Lambda$CDM", "a": "Model A", "b": "Model B"}

    center: dict[str, dict[str, np.ndarray]] = {}
    bands: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    for i, m in enumerate(model_order):
        r = production_results[m]
        mean_theta = np.array([r["posterior"][p]["mean"] for p in r["param_names"]], dtype=float)
        center[m] = _model_h_q_w_dl(m, mean_theta, z)

        chains = r["chains"]
        burn = int(r["burn_in_steps"])
        flat = chains[:, burn:, :].reshape(-1, chains.shape[-1])
        draw_idx = _posterior_draw_indices(flat.shape[0], n_draws, seed=12345 + i)
        draws = flat[draw_idx]

        h_draws = []
        dl_draws = []
        q_draws = []
        w_draws = []
        for th in draws:
            obs = _model_h_q_w_dl(m, th, z)
            h_draws.append(obs["h"])
            dl_draws.append(obs["dl"])
            q_draws.append(obs["q"])
            w_draws.append(obs["w"])

        def _band(arrs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            a = np.asarray(arrs)
            lo, hi = np.quantile(a, [0.16, 0.84], axis=0)
            return lo, hi

        bands[m] = {
            "h": _band(h_draws),
            "dl": _band(dl_draws),
            "q": _band(q_draws),
            "w": _band(w_draws),
        }

    saved: list[str] = []

    # 1) H(z) comparison + bands
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for m in model_order:
        ax.plot(z, center[m]["h"], color=colors[m], lw=1.8, label=labels[m])
        lo, hi = bands[m]["h"]
        ax.fill_between(z, lo, hi, color=colors[m], alpha=0.15)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_title("H(z) posteriors")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = out_dir / "hz_comparison_posterior.pdf"
    fig.savefig(p)
    plt.close(fig)
    saved.append(str(p))

    # 2) ΔH(z) residuals vs LCDM + bands
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0.0, color="0.3", lw=0.8)
    h0 = center["lcdm"]["h"]
    _h0_safe = np.where(h0 > 0, h0, np.nan)
    for m in ("a", "b"):
        d = np.where(h0 > 0, (center[m]["h"] - h0) / _h0_safe, 0.0)
        lo, hi = bands[m]["h"]
        d_lo = np.where(h0 > 0, (lo - h0) / _h0_safe, 0.0)
        d_hi = np.where(h0 > 0, (hi - h0) / _h0_safe, 0.0)
        ax.plot(z, d, color=colors[m], lw=1.8, label=labels[m])
        ax.fill_between(z, d_lo, d_hi, color=colors[m], alpha=0.18)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$\Delta_H(z)$")
    ax.set_title(r"Fractional Hubble residuals vs $\Lambda$CDM")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = out_dir / "delta_hz_posterior.pdf"
    fig.savefig(p)
    plt.close(fig)
    saved.append(str(p))

    # 3) ΔD_L(z) residuals + bands
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0.0, color="0.3", lw=0.8)
    dl0 = center["lcdm"]["dl"]
    # dl0[0] = 0 by cumulative_trapezoid initialisation; guard against divide-by-zero
    _dl0_safe = np.where(dl0 > 0, dl0, np.nan)
    for m in ("a", "b"):
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(dl0 > 0, (center[m]["dl"] - dl0) / _dl0_safe, 0.0)
            lo, hi = bands[m]["dl"]
            d_lo = np.where(dl0 > 0, (lo - dl0) / _dl0_safe, 0.0)
            d_hi = np.where(dl0 > 0, (hi - dl0) / _dl0_safe, 0.0)
        ax.plot(z, d, color=colors[m], lw=1.8, label=labels[m])
        ax.fill_between(z, d_lo, d_hi, color=colors[m], alpha=0.18)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$\Delta_{D_L}(z)$")
    ax.set_title(r"Luminosity-distance residuals vs $\Lambda$CDM")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = out_dir / "delta_dl_posterior.pdf"
    fig.savefig(p)
    plt.close(fig)
    saved.append(str(p))

    # 4) q(z)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
    for m in model_order:
        ax.plot(z, center[m]["q"], color=colors[m], lw=1.8, label=labels[m])
        lo, hi = bands[m]["q"]
        ax.fill_between(z, lo, hi, color=colors[m], alpha=0.15)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$q(z)$")
    ax.set_title("Deceleration parameter posterior bands")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = out_dir / "qz_posterior.pdf"
    fig.savefig(p)
    plt.close(fig)
    saved.append(str(p))

    # 5) w(z) for A/B
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(-1.0, color="0.4", lw=0.8, ls=":")
    for m in ("a", "b"):
        ax.plot(z, center[m]["w"], color=colors[m], lw=1.8, label=labels[m])
        lo, hi = bands[m]["w"]
        ax.fill_between(z, lo, hi, color=colors[m], alpha=0.18)
    ax.set_xlabel("z")
    ax.set_ylabel(r"$w(z)$")
    ax.set_title("Dark-energy equation-of-state posterior bands")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = out_dir / "wz_posterior.pdf"
    fig.savefig(p)
    plt.close(fig)
    saved.append(str(p))

    # 6) Corner plots (if available)
    _LATEX_LABELS = {
        "h0":     r"$H_0\ [\mathrm{km\,s^{-1}\,Mpc^{-1}}]$",
        "omega_m": r"$\Omega_m$",
        "w0":     r"$w_0$",
        "B":      r"$B$",
        "C":      r"$C$",
        "omega":  r"$\omega$",
        "mu":     r"$\mu$",
    }
    # Explicit axis ranges for parameters with non-obvious sign (forces axis ticks
    # into the correct domain so PDF text-extraction cannot drop the minus sign).
    _PARAM_RANGES: dict[str, tuple[float, float]] = {
        "w0": (-1.52, -0.48),   # prior U(-1.5,-0.5); explicit negative range
    }
    if HAS_CORNER:
        for m in model_order:
            r = production_results[m]
            chains = r["chains"]
            burn = int(r["burn_in_steps"])
            flat = chains[:, burn:, :].reshape(-1, chains.shape[-1])
            latex_labels = [_LATEX_LABELS.get(p, p) for p in r["param_names"]]
            corner_range = [
                _PARAM_RANGES[p] if p in _PARAM_RANGES else 0.999
                for p in r["param_names"]
            ]
            cfig = corner.corner(
                flat,
                labels=latex_labels,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_fmt=".4f",
                title_kwargs={"fontsize": 8},
                label_kwargs={"fontsize": 9},
                range=corner_range,
            )
            cp = out_dir / f"corner_{m}.pdf"
            cfig.savefig(cp, bbox_inches="tight")
            plt.close(cfig)
            saved.append(str(cp))

    # 7) SN Ia Hubble diagram (m_b_corr)
    z_sn = np.asarray(sn_dataset.z)
    y_sn = np.asarray(sn_dataset.y_obs)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.scatter(z_sn, y_sn, s=5, alpha=0.35, color="0.35", label="Pantheon+ data")

    z_line = np.linspace(max(1e-4, np.min(z_sn)), np.max(z_sn), 300)
    for m in model_order:
        r = production_results[m]
        mean_theta = np.array([r["posterior"][p]["mean"] for p in r["param_names"]], dtype=float)
        cosmo, ma, mb = _model_params_from_theta(m, mean_theta)
        mu_sn = predict_observable(z_sn, "mu", cosmo, ma, mb)
        mb_hat = _weighted_mb_best_fit(sn_dataset, mu_sn)
        mu_line = predict_observable(z_line, "mu", cosmo, ma, mb)
        m_model = mu_line + mb_hat
        ax.plot(z_line, m_model, color=colors[m], lw=1.6, label=f"{labels[m]} best-fit")

    ax.set_xlabel("z")
    ax.set_ylabel(r"$m_b^{\mathrm{corr}}$")
    ax.set_title("SN Ia Hubble diagram")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = out_dir / "sn_hubble_diagram.pdf"
    fig.savefig(p)
    plt.close(fig)
    saved.append(str(p))

    return saved


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def run_production(
    use_nuts: bool = False,
    parallel_models: bool = False,
    parallel_workers: int = 3,
    allow_gpu_parallel: bool = False,
) -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)

    cfg = SamplerConfig(use_nuts=use_nuts)
    sensitivity_cfg = SENSITIVITY_CFG

    gpu_label = "ON (NVIDIA CUDA via CuPy)" if USE_GPU else "OFF (CPU only)"
    print(f"[Production] GPU acceleration: {gpu_label}")
    if parallel_models:
        print(f"[Production] Parallel model mode: ON (workers={parallel_workers})")

    if parallel_models and USE_GPU and not allow_gpu_parallel:
        print(
            "[Production] Parallel workers requested with CuPy available. "
            "Disabling CuPy in MH batches to avoid GPU memory contention. "
            "Use --allow-gpu-parallel to override."
        )
        if not cfg.use_nuts:
            cfg = replace(cfg, enable_gpu=False)
        if not sensitivity_cfg.use_nuts:
            sensitivity_cfg = replace(sensitivity_cfg, enable_gpu=False)

    sn = load_pantheon_plus()
    bao = load_all_bao()
    datasets = [(sn, "mb")] + bao

    production = _run_model_batch(
        models=("lcdm", "a", "b"),
        datasets=datasets,
        cfg=cfg,
        rd=147.09,
        run_tag="production",
        stage_label="Production",
        parallel_models=parallel_models,
        parallel_workers=parallel_workers,
    )

    # Sensitivity A: r_d change
    sensitivity_rd = _run_model_batch(
        models=("lcdm", "a", "b"),
        datasets=datasets,
        cfg=sensitivity_cfg,
        rd=147.21,
        run_tag="sensitivity_rd14721",
        stage_label="Sensitivity r_d",
        parallel_models=parallel_models,
        parallel_workers=parallel_workers,
    )

    # Sensitivity B: remove DESI BGS-like point
    datasets_no_bgs = _exclude_bgs_point(datasets)
    sensitivity_no_bgs = _run_model_batch(
        models=("lcdm", "a", "b"),
        datasets=datasets_no_bgs,
        cfg=sensitivity_cfg,
        rd=147.09,
        run_tag="sensitivity_no_bgs",
        stage_label="Sensitivity no-BGS",
        parallel_models=parallel_models,
        parallel_workers=parallel_workers,
    )

    model_comp = _compute_delta_metrics(production)

    # Tables
    conv_path = TABLE_ROOT / "convergence_diagnostics_production.md"
    write_convergence_table(conv_path, production)

    param_path = TABLE_ROOT / "parameter_constraints.md"
    write_parameter_constraints_table(param_path, production)

    # Write JSON *before* figures so a matplotlib crash cannot prevent data persistence.
    out = {
        "config": asdict(cfg),
        "constants": {"N_data": N_DATA, "k_by_model": K_BY_MODEL},
        "production": production,
        "sensitivity": {
            "rd_14721": sensitivity_rd,
            "exclude_desi_bgs": sensitivity_no_bgs,
        },
        "model_comparison": model_comp,
        "artifacts": {
            "convergence_table": str(conv_path),
            "parameter_table": str(param_path),
            "figures": [],  # updated after figure generation below
            "has_matplotlib": HAS_MPL,
            "has_corner": HAS_CORNER,
        },
    }

    json_path = OUTPUT_ROOT / "production_results.json"
    json_path.write_text(json.dumps(_to_jsonable(out), indent=2), encoding="utf-8")
    print(f"[Production] Wrote intermediate JSON: {json_path}")

    # Figures (wrapped so a crash here does not lose the JSON above)
    try:
        figure_files = generate_publication_figures(production, sn, FIG_ROOT)
        # Patch the figures list into the already-written JSON
        out["artifacts"]["figures"] = figure_files
        json_path.write_text(json.dumps(_to_jsonable(out), indent=2), encoding="utf-8")
    except Exception as _fig_err:  # pragma: no cover
        figure_files = []
        print(f"[Production] WARNING: figure generation failed: {_fig_err}")


    print("=" * 88)
    print("Production MCMC summary (SN + BAO + Planck compressed priors)")
    print("=" * 88)
    print("| Model | Chains×Steps | <acc> | max R-hat | min ESS | max lnL | BIC | AIC | AICc | WAIC |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in ("lcdm", "a", "b"):
        r = production[model]
        ic = r["information_criteria"]
        waic_val = r.get("waic", {}).get("waic", float("nan"))
        waic_str = f"{waic_val:.3f}" if np.isfinite(waic_val) else "nan"
        print(
            f"| {r['model_label']} | {r['n_chains']}x{r['steps_used']} | {r['accept_rate_mean']:.3f} | "
            f"{r['max_split_rhat']:.4f} | {r['min_ess']:.1f} | {r['max_loglike_all_samples']:.3f} | "
            f"{ic['bic']:.3f} | {ic['aic']:.3f} | {ic['aicc']:.3f} | {waic_str} |"
        )

    print("\nInformation-criterion deltas (vs lcdm):")
    d_lcdm = model_comp["delta_vs_lcdm"]
    for model in ("lcdm", "a", "b"):
        dwaic = d_lcdm[model].get("waic", float("nan"))
        dwaic_str = f"{dwaic:+.3f}" if np.isfinite(dwaic) else "nan"
        print(
            f"  {MODEL_LABEL[model]:<9s}  dBIC={d_lcdm[model]['bic']:+.3f}, "
            f"dAIC={d_lcdm[model]['aic']:+.2f}, dAICc={d_lcdm[model]['aicc']:+.2f}, "
            f"dWAIC={dwaic_str}"
        )

    print(f"\nWrote JSON: {json_path}")
    print(f"Wrote table: {conv_path}")
    print(f"Wrote table: {param_path}")
    if figure_files:
        print("Wrote figures:")
        for fp in figure_files:
            print(f"  - {fp}")
    else:
        print("No figures written (matplotlib unavailable).")

    return out


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Ultra-slow DE production MCMC run.")
    _parser.add_argument(
        "--nuts",
        action="store_true",
        default=False,
        help="Use BlackJAX NUTS for all models (ΛCDM, Model A, Model B; requires JAX + BlackJAX).",
    )
    _parser.add_argument(
        "--parallel-models",
        action="store_true",
        default=False,
        help="Run model batches in parallel worker processes (lcdm/a/b concurrently).",
    )
    _parser.add_argument(
        "--parallel-workers",
        type=int,
        default=3,
        help="Number of worker processes for --parallel-models (default: 3).",
    )
    _parser.add_argument(
        "--allow-gpu-parallel",
        action="store_true",
        default=False,
        help="Allow GPU-backed MH workers in parallel mode (may cause GPU memory contention).",
    )
    _args = _parser.parse_args()
    run_production(
        use_nuts=_args.nuts,
        parallel_models=_args.parallel_models,
        parallel_workers=_args.parallel_workers,
        allow_gpu_parallel=_args.allow_gpu_parallel,
    )