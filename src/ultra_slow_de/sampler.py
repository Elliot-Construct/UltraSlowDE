"""Lightweight MCMC sampler for pilot parameter sweeps.

Uses a Metropolis–Hastings random-walk kernel.  This is intentionally
minimal — production runs should use emcee or cobaya.  The pilot sampler
validates the likelihood pipeline and parameter space before investing
compute.
"""

from dataclasses import dataclass

import numpy as np

try:
    from tqdm import tqdm as _tqdm_cls
    def _tqdm(it, **kwargs):  # type: ignore[misc]
        return _tqdm_cls(it, **kwargs)
except ImportError:  # pragma: no cover
    def _tqdm(it, **kwargs):  # type: ignore[misc]
        return it

from .datasets import GaussianDataset
from .inference import joint_logposterior
from .likelihood import SNLikelihoodCached
from .model_a import ModelAParams
from .model_b import ModelBParams
from .params import CosmoParams


@dataclass
class MCMCResult:
    chain: np.ndarray        # (n_steps, n_params)
    loglike: np.ndarray      # (n_steps,)
    accept_rate: float
    param_names: list[str]
    proposal_sigma_final: np.ndarray | None = None
    ess_per_param: np.ndarray | None = None
    backend: str = "numpy"
    device: str = "cpu"
    used_fallback: bool = False


@dataclass
class MultiChainMCMCResult:
    chains: np.ndarray           # (n_chains, n_steps, n_params)
    loglike: np.ndarray          # (n_chains, n_steps)
    accept_rates: np.ndarray     # (n_chains,)
    param_names: list[str]
    rhat_per_param: np.ndarray | None = None
    ess_per_param: np.ndarray | None = None
    proposal_sigma_final: np.ndarray | None = None
    backend: str = "numpy"
    device: str = "cpu"
    used_fallback: bool = False


def _pack_params_model_a(cosmo: CosmoParams, model: ModelAParams) -> np.ndarray:
    return np.array([cosmo.h0, cosmo.omega_m, model.w0, model.B, model.C, model.omega])


def _unpack_params_model_a(theta: np.ndarray) -> tuple[CosmoParams, ModelAParams]:
    h0, omega_m, w0, B, C, omega = theta
    return (
        CosmoParams(h0=h0, omega_m=omega_m),
        ModelAParams(w0=w0, B=B, C=C, omega=omega),
    )

MODEL_A_NAMES = ["h0", "omega_m", "w0", "B", "C", "omega"]


def _pack_params_model_b(cosmo: CosmoParams, model: ModelBParams) -> np.ndarray:
    return np.array([cosmo.h0, cosmo.omega_m, model.mu])


def _unpack_params_model_b(theta: np.ndarray) -> tuple[CosmoParams, ModelBParams]:
    h0, omega_m, mu = theta
    return (
        CosmoParams(h0=h0, omega_m=omega_m),
        ModelBParams(mu=mu),
    )

MODEL_B_NAMES = ["h0", "omega_m", "mu"]


def _flat_prior(theta: np.ndarray, bounds: np.ndarray) -> float:
    if np.all((theta >= bounds[:, 0]) & (theta <= bounds[:, 1])):
        return 0.0
    return -np.inf


def _reflect_into_bounds(theta: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Reflect a parameter vector into [lo, hi] bounds per dimension.

    Reflection preserves proposal symmetry for a Gaussian random walk with
    hard box constraints and avoids wasted out-of-bounds proposals.
    """
    out = theta.copy()
    for j in range(theta.size):
        lo, hi = bounds[j]
        width = hi - lo
        if not np.isfinite(width) or width <= 0:
            continue
        y = (out[j] - lo) % (2.0 * width)
        if y > width:
            y = 2.0 * width - y
        out[j] = lo + y
    return out


def _estimate_ess_1d(x: np.ndarray) -> float:
    """Estimate effective sample size using initial positive autocorrelation sum."""
    n = x.size
    if n < 4:
        return float(n)
    xc = x - np.mean(x)
    var = float(np.dot(xc, xc) / n)
    if var <= 0.0:
        return float(n)

    # O(n^2) is fine for pilot-chain sizes used here.
    max_lag = min(n - 1, 500)
    rho_sum = 0.0
    for lag in range(1, max_lag + 1):
        c = float(np.dot(xc[:-lag], xc[lag:]) / (n - lag))
        rho = c / var
        if rho <= 0.0:
            break
        rho_sum += rho
    tau = 1.0 + 2.0 * rho_sum
    return float(max(1.0, n / tau))


def _estimate_ess(chain: np.ndarray) -> np.ndarray:
    """Per-parameter ESS estimate for a single chain."""
    return np.array([_estimate_ess_1d(chain[:, j]) for j in range(chain.shape[1])], dtype=float)


def _rhat_split(chains: np.ndarray) -> np.ndarray:
    """Compute split-Rhat for array of shape (m, n, d)."""
    m, n, d = chains.shape
    if m < 2 or n < 4:
        return np.full(d, np.nan, dtype=float)

    # Split each chain in half to stabilise non-stationarity diagnostics.
    n2 = n // 2
    if n2 < 2:
        return np.full(d, np.nan, dtype=float)
    split = np.concatenate([chains[:, :n2, :], chains[:, n2: 2 * n2, :]], axis=0)
    m_s, n_s, _ = split.shape

    chain_means = np.mean(split, axis=1)
    chain_vars = np.var(split, axis=1, ddof=1)

    w = np.mean(chain_vars, axis=0)
    b = n_s * np.var(chain_means, axis=0, ddof=1)
    var_hat = ((n_s - 1.0) / n_s) * w + (1.0 / n_s) * b
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_hat / w)
    rhat[~np.isfinite(rhat)] = np.nan
    return rhat


def _aggregate_ess(chains: np.ndarray) -> np.ndarray:
    """Conservative aggregate ESS: sum per-chain ESS after burn-in."""
    m, _, d = chains.shape
    out = np.zeros(d, dtype=float)
    for i in range(m):
        out += _estimate_ess(chains[i])
    return out


def _default_target_accept(n_params: int) -> float:
    """Heuristic target acceptance for random-walk MH."""
    if n_params <= 2:
        return 0.35
    return 0.25


def run_mcmc(
    datasets: list[tuple[GaussianDataset, str]],
    model: str,
    theta0: np.ndarray,
    bounds: np.ndarray,
    proposal_sigma: np.ndarray,
    n_steps: int = 500,
    seed: int = 42,
    rd: float = 147.09,
    include_planck: bool = False,
    adapt_proposal: bool = True,
    adapt_steps: int | None = None,
    adapt_interval: int = 25,
    target_accept: float | None = None,
    reflect_bounds: bool = True,
    proposal_scale_min: float = 0.05,
    proposal_scale_max: float = 20.0,
    compute_ess: bool = True,
    use_gpu: bool = False,
    growth_likelihood_mode: str = "exploratory_gamma",
    growth_backend: str = "auto",
) -> MCMCResult:
    """Metropolis–Hastings pilot sampler.

    Parameters
    ----------
    datasets : list of (GaussianDataset, observable_name)
    model : "a" | "b" | "lcdm"
    theta0 : initial parameter vector
    bounds : (n_params, 2) array of [lo, hi] for flat prior
    proposal_sigma : std-dev for Gaussian proposal per parameter
    n_steps : number of MH steps
    seed : RNG seed for reproducibility
    rd : sound horizon at drag epoch in Mpc
    include_planck : include Planck compressed distance priors
    adapt_proposal : adapt proposal scale during warm-up
    adapt_steps : number of warm-up steps for adaptation (default n_steps//5)
    adapt_interval : adaptation update cadence in steps
    target_accept : target acceptance during adaptation; if None, uses a
        dimension-aware default (0.35 for 1--2D, else 0.25)
    reflect_bounds : reflect proposals into parameter bounds
    proposal_scale_min/proposal_scale_max : clamp adaptation scale relative to
        initial proposal_sigma
    compute_ess : compute per-parameter ESS estimate from post-burn chain
    """
    rng = np.random.default_rng(seed)
    n_params = len(theta0)
    chain = np.empty((n_steps, n_params))
    ll = np.empty(n_steps)
    accepted = 0
    accepted_window = 0

    if adapt_steps is None:
        adapt_steps = max(1, n_steps // 5)
    if target_accept is None:
        target_accept = _default_target_accept(n_params)

    base_sigma = np.asarray(proposal_sigma, dtype=float)
    current_sigma = base_sigma.copy()
    log_scale = 0.0

    # Pre-factorise the SN covariance once (O(N^3)) to make each step O(N^2).
    # With use_gpu=True the factorisation and each solve run on the GPU.
    sn_cache = None
    for ds, obs in datasets:
        if obs.lower() == "mb":
            sn_cache = SNLikelihoodCached(ds, use_gpu=use_gpu)
            break

    def logpost(theta: np.ndarray) -> float:
        lp = _flat_prior(theta, bounds)
        if not np.isfinite(lp):
            return -np.inf
        try:
            if model == "a":
                cosmo, ma = _unpack_params_model_a(theta)
                return lp + joint_logposterior(
                    datasets, cosmo, model_a=ma, rd=rd,
                    include_planck=include_planck,
                    sn_cache=sn_cache,
                    growth_likelihood_mode=growth_likelihood_mode,
                    growth_backend=growth_backend,
                )
            elif model == "b":
                cosmo, mb = _unpack_params_model_b(theta)
                return lp + joint_logposterior(
                    datasets, cosmo, model_b=mb, rd=rd,
                    include_planck=include_planck,
                    sn_cache=sn_cache,
                    growth_likelihood_mode=growth_likelihood_mode,
                    growth_backend=growth_backend,
                )
            else:  # lcdm
                cosmo = CosmoParams(h0=theta[0], omega_m=theta[1])
                return lp + joint_logposterior(
                    datasets, cosmo, rd=rd,
                    include_planck=include_planck,
                    sn_cache=sn_cache,
                    growth_likelihood_mode=growth_likelihood_mode,
                    growth_backend=growth_backend,
                )
        except Exception:
            return -np.inf

    current = theta0.copy()
    current_ll = logpost(current)

    step_iter = _tqdm(range(n_steps), desc=f"MCMC [{model}]", unit="step", leave=False, ascii=True)

    for i in step_iter:
        proposal = current + rng.normal(0.0, current_sigma)
        if reflect_bounds:
            proposal = _reflect_into_bounds(proposal, bounds)
        prop_ll = logpost(proposal)
        log_alpha = prop_ll - current_ll
        if np.log(rng.uniform()) < log_alpha:
            current = proposal
            current_ll = prop_ll
            accepted += 1
            accepted_window += 1
        chain[i] = current
        ll[i] = current_ll

        if (
            adapt_proposal
            and (i + 1) <= adapt_steps
            and adapt_interval > 0
            and (i + 1) % adapt_interval == 0
        ):
            acc = accepted_window / adapt_interval
            t = (i + 1) / adapt_interval
            eta = 1.0 / np.sqrt(t + 1.0)
            log_scale += eta * (acc - target_accept)
            scale = float(np.exp(log_scale))
            current_sigma = np.clip(
                base_sigma * scale,
                base_sigma * proposal_scale_min,
                base_sigma * proposal_scale_max,
            )
            accepted_window = 0

    if model == "a":
        names = MODEL_A_NAMES
    elif model == "b":
        names = MODEL_B_NAMES
    else:
        names = ["h0", "omega_m"]

    ess = None
    if compute_ess and n_steps >= 10:
        burn = max(1, n_steps // 5)
        ess = _estimate_ess(chain[burn:])

    return MCMCResult(
        chain=chain,
        loglike=ll,
        accept_rate=accepted / n_steps,
        param_names=names,
        proposal_sigma_final=current_sigma,
        ess_per_param=ess,
        backend="numpy",
        device="cpu",
        used_fallback=False,
    )


def run_mcmc_multichain(
    datasets: list[tuple[GaussianDataset, str]],
    model: str,
    theta0: np.ndarray,
    bounds: np.ndarray,
    proposal_sigma: np.ndarray,
    n_steps: int = 500,
    n_chains: int = 4,
    seed: int = 42,
    init_scatter: float = 0.5,
    rd: float = 147.09,
    include_planck: bool = False,
    adapt_proposal: bool = True,
    adapt_steps: int | None = None,
    adapt_interval: int = 25,
    target_accept: float | None = None,
    reflect_bounds: bool = True,
    proposal_scale_min: float = 0.05,
    proposal_scale_max: float = 20.0,
    compute_ess: bool = True,
    use_gpu: bool = False,
    growth_likelihood_mode: str = "exploratory_gamma",
    growth_backend: str = "auto",
) -> MultiChainMCMCResult:
    """Run independent pilot chains and compute convergence diagnostics.

    Returns split-Rhat and aggregate ESS on post-burn segments when feasible.
    """
    if n_chains < 1:
        raise ValueError("n_chains must be >= 1")

    rng = np.random.default_rng(seed)
    starts = np.empty((n_chains, theta0.size), dtype=float)
    for i in range(n_chains):
        jitter = rng.normal(0.0, np.asarray(proposal_sigma, dtype=float) * init_scatter)
        starts[i] = np.asarray(theta0, dtype=float) + jitter
        starts[i] = _reflect_into_bounds(starts[i], bounds)

    chain_list: list[np.ndarray] = []
    ll_list: list[np.ndarray] = []
    acc_list: list[float] = []
    ess_list: list[np.ndarray | None] = []
    names: list[str] | None = None
    sigma_final_list: list[np.ndarray] = []

    chain_iter = _tqdm(range(n_chains), desc=f"Chains [{model}]", unit="chain", ascii=True)

    for i in chain_iter:
        res = run_mcmc(
            datasets=datasets,
            model=model,
            theta0=starts[i],
            bounds=bounds,
            proposal_sigma=proposal_sigma,
            n_steps=n_steps,
            seed=seed + i + 1,
            rd=rd,
            include_planck=include_planck,
            adapt_proposal=adapt_proposal,
            adapt_steps=adapt_steps,
            adapt_interval=adapt_interval,
            target_accept=target_accept,
            reflect_bounds=reflect_bounds,
            proposal_scale_min=proposal_scale_min,
            proposal_scale_max=proposal_scale_max,
            compute_ess=compute_ess,
            use_gpu=use_gpu,
            growth_likelihood_mode=growth_likelihood_mode,
            growth_backend=growth_backend,
        )
        chain_list.append(res.chain)
        ll_list.append(res.loglike)
        acc_list.append(res.accept_rate)
        ess_list.append(res.ess_per_param)
        sigma_final_list.append(
            res.proposal_sigma_final if res.proposal_sigma_final is not None else np.asarray(proposal_sigma, dtype=float)
        )
        names = res.param_names

    chains = np.stack(chain_list, axis=0)
    lls = np.stack(ll_list, axis=0)
    accept_rates = np.asarray(acc_list, dtype=float)

    burn = max(1, n_steps // 5)
    post = chains[:, burn:, :]

    rhat = _rhat_split(post) if (n_chains >= 2 and post.shape[1] >= 4) else None
    ess = _aggregate_ess(post) if (compute_ess and post.shape[1] >= 4) else None
    sigma_final = np.mean(np.stack(sigma_final_list, axis=0), axis=0)

    return MultiChainMCMCResult(
        chains=chains,
        loglike=lls,
        accept_rates=accept_rates,
        param_names=names or [],
        rhat_per_param=rhat,
        ess_per_param=ess,
        proposal_sigma_final=sigma_final,
        backend="numpy",
        device="cpu",
        used_fallback=False,
    )


def run_mcmc_backend(
    datasets: list[tuple[GaussianDataset, str]],
    model: str,
    theta0: np.ndarray,
    bounds: np.ndarray,
    proposal_sigma: np.ndarray,
    n_steps: int = 500,
    seed: int = 42,
    rd: float = 147.09,
    include_planck: bool = False,
    adapt_proposal: bool = True,
    adapt_steps: int | None = None,
    adapt_interval: int = 25,
    target_accept: float | None = None,
    reflect_bounds: bool = True,
    proposal_scale_min: float = 0.05,
    proposal_scale_max: float = 20.0,
    compute_ess: bool = True,
    backend: str = "numpy",
    use_gpu: bool = False,
    growth_likelihood_mode: str = "exploratory_gamma",
    growth_backend: str = "auto",
) -> MCMCResult:
    """Run pilot MCMC with backend selection.

    Parameters are identical to :func:`run_mcmc` with an additional
    ``backend`` selector (``"numpy"`` or ``"jax"``).
    """
    selected = backend.lower()
    if selected == "numpy":
        return run_mcmc(
            datasets=datasets,
            model=model,
            theta0=theta0,
            bounds=bounds,
            proposal_sigma=proposal_sigma,
            n_steps=n_steps,
            seed=seed,
            rd=rd,
            include_planck=include_planck,
            adapt_proposal=adapt_proposal,
            adapt_steps=adapt_steps,
            adapt_interval=adapt_interval,
            target_accept=target_accept,
            reflect_bounds=reflect_bounds,
            proposal_scale_min=proposal_scale_min,
            proposal_scale_max=proposal_scale_max,
            compute_ess=compute_ess,
            use_gpu=use_gpu,
            growth_likelihood_mode=growth_likelihood_mode,
            growth_backend=growth_backend,
        )
    if selected == "jax":
        from .sampler_jax import run_mcmc_jax

        return run_mcmc_jax(
            datasets=datasets,
            model=model,
            theta0=theta0,
            bounds=bounds,
            proposal_sigma=proposal_sigma,
            n_steps=n_steps,
            seed=seed,
            rd=rd,
            include_planck=include_planck,
            adapt_proposal=adapt_proposal,
            adapt_steps=adapt_steps,
            adapt_interval=adapt_interval,
            target_accept=target_accept,
            reflect_bounds=reflect_bounds,
            proposal_scale_min=proposal_scale_min,
            proposal_scale_max=proposal_scale_max,
            compute_ess=compute_ess,
        )
    raise ValueError(f"Unsupported backend '{backend}'. Use 'numpy' or 'jax'.")


def run_mcmc_multichain_backend(
    datasets: list[tuple[GaussianDataset, str]],
    model: str,
    theta0: np.ndarray,
    bounds: np.ndarray,
    proposal_sigma: np.ndarray,
    n_steps: int = 500,
    n_chains: int = 4,
    seed: int = 42,
    init_scatter: float = 0.5,
    rd: float = 147.09,
    include_planck: bool = False,
    adapt_proposal: bool = True,
    adapt_steps: int | None = None,
    adapt_interval: int = 25,
    target_accept: float | None = None,
    reflect_bounds: bool = True,
    proposal_scale_min: float = 0.05,
    proposal_scale_max: float = 20.0,
    compute_ess: bool = True,
    backend: str = "numpy",
    use_gpu: bool = False,
    growth_likelihood_mode: str = "exploratory_gamma",
    growth_backend: str = "auto",
) -> MultiChainMCMCResult:
    """Run multi-chain pilot MCMC with backend selection.

    Parameters are identical to :func:`run_mcmc_multichain` with an additional
    ``backend`` selector (``"numpy"`` or ``"jax"``).
    """
    selected = backend.lower()
    if selected == "numpy":
        return run_mcmc_multichain(
            datasets=datasets,
            model=model,
            theta0=theta0,
            bounds=bounds,
            proposal_sigma=proposal_sigma,
            n_steps=n_steps,
            n_chains=n_chains,
            seed=seed,
            init_scatter=init_scatter,
            rd=rd,
            include_planck=include_planck,
            adapt_proposal=adapt_proposal,
            adapt_steps=adapt_steps,
            adapt_interval=adapt_interval,
            target_accept=target_accept,
            reflect_bounds=reflect_bounds,
            proposal_scale_min=proposal_scale_min,
            proposal_scale_max=proposal_scale_max,
            compute_ess=compute_ess,
            growth_likelihood_mode=growth_likelihood_mode,
            growth_backend=growth_backend,
        )
    if selected == "jax":
        from .sampler_jax import run_mcmc_multichain_jax

        return run_mcmc_multichain_jax(
            datasets=datasets,
            model=model,
            theta0=theta0,
            bounds=bounds,
            proposal_sigma=proposal_sigma,
            n_steps=n_steps,
            n_chains=n_chains,
            seed=seed,
            init_scatter=init_scatter,
            rd=rd,
            include_planck=include_planck,
            adapt_proposal=adapt_proposal,
            adapt_steps=adapt_steps,
            adapt_interval=adapt_interval,
            target_accept=target_accept,
            reflect_bounds=reflect_bounds,
            proposal_scale_min=proposal_scale_min,
            proposal_scale_max=proposal_scale_max,
            compute_ess=compute_ess,
        )
    raise ValueError(f"Unsupported backend '{backend}'. Use 'numpy' or 'jax'.")
