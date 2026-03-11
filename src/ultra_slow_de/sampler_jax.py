"""Optional JAX-backed pilot samplers.

This module provides a pragmatic, vectorised multi-chain Metropolis backend
that can use JAX random generation and array ops. It keeps compatibility with
existing sampler result schemas and falls back to the NumPy backend when JAX is
not installed.

Extension hooks for NumPyro/BlackJAX are intentionally exposed through the
module-level API and metadata fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .datasets import GaussianDataset
from .inference import joint_logposterior
from .likelihood import SNLikelihoodCached
from .params import CosmoParams, ModelAParams
from .sampler import (
    MCMCResult,
    MultiChainMCMCResult,
    MODEL_A_NAMES,
    MODEL_B_NAMES,
    _aggregate_ess,
    _default_target_accept,
    _rhat_split,
    _reflect_into_bounds,
    _unpack_params_model_a,
    _unpack_params_model_b,
    run_mcmc_multichain,
)

try:
    import jax  # type: ignore[import-not-found]
    import jax.numpy as jnp  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - exercised in fallback tests
    jax = None
    jnp = None


@dataclass(frozen=True)
class JAXBackendInfo:
    available: bool
    device: str


def jax_backend_info() -> JAXBackendInfo:
    """Return JAX backend availability and default device."""
    if jax is None:
        return JAXBackendInfo(available=False, device="cpu")
    try:
        return JAXBackendInfo(available=True, device=str(jax.default_backend()))
    except Exception:
        return JAXBackendInfo(available=True, device="cpu")


def jax_backend_available() -> bool:
    return jax_backend_info().available


def _reflect_into_bounds_jax(theta: Any, bounds: Any) -> Any:
    if jnp is None:
        raise RuntimeError("JAX backend requested but jax.numpy is unavailable")
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    width = hi - lo
    safe_width = jnp.where((jnp.isfinite(width)) & (width > 0), width, 1.0)
    y = jnp.mod(theta - lo, 2.0 * safe_width)
    y = jnp.where(y > safe_width, 2.0 * safe_width - y, y)
    reflected = lo + y
    valid = (jnp.isfinite(width)) & (width > 0)
    return jnp.where(valid, reflected, theta)


def _logpost_factory(
    datasets: list[tuple[GaussianDataset, str]],
    model: str,
    bounds: np.ndarray,
    rd: float,
    include_planck: bool,
):
    sn_cache = None
    for ds, obs in datasets:
        if obs.lower() == "mb":
            sn_cache = SNLikelihoodCached(ds)
            break

    def logpost(theta: np.ndarray) -> float:
        if not np.all((theta >= bounds[:, 0]) & (theta <= bounds[:, 1])):
            return -np.inf
        try:
            if model == "a":
                cosmo, ma = _unpack_params_model_a(theta)
                return float(
                    joint_logposterior(
                        datasets,
                        cosmo,
                        model_a=ma,
                        rd=rd,
                        include_planck=include_planck,
                        sn_cache=sn_cache,
                    )
                )
            if model == "b":
                cosmo, mb = _unpack_params_model_b(theta)
                return float(
                    joint_logposterior(
                        datasets,
                        cosmo,
                        model_b=mb,
                        rd=rd,
                        include_planck=include_planck,
                        sn_cache=sn_cache,
                    )
                )
            cosmo = CosmoParams(h0=float(theta[0]), omega_m=float(theta[1]))
            return float(
                joint_logposterior(
                    datasets,
                    cosmo,
                    rd=rd,
                    include_planck=include_planck,
                    sn_cache=sn_cache,
                )
            )
        except Exception:
            return -np.inf

    return logpost


def _param_names(model: str) -> list[str]:
    if model == "a":
        return MODEL_A_NAMES
    if model == "b":
        return MODEL_B_NAMES
    return ["h0", "omega_m"]


def run_mcmc_jax(
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
) -> MCMCResult:
    """Single-chain wrapper for JAX backend with deterministic fallback."""
    res = run_mcmc_multichain_jax(
        datasets=datasets,
        model=model,
        theta0=theta0,
        bounds=bounds,
        proposal_sigma=proposal_sigma,
        n_steps=n_steps,
        n_chains=1,
        seed=seed,
        init_scatter=0.0,
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

    chain = res.chains[0]
    ll = res.loglike[0]
    burn = max(1, n_steps // 5)
    ess = None
    if compute_ess and n_steps >= 10:
        from .sampler import _estimate_ess

        ess = _estimate_ess(chain[burn:])

    return MCMCResult(
        chain=chain,
        loglike=ll,
        accept_rate=float(res.accept_rates[0]),
        param_names=res.param_names,
        proposal_sigma_final=res.proposal_sigma_final,
        ess_per_param=ess,
        backend=res.backend,
        device=res.device,
        used_fallback=res.used_fallback,
    )


def run_mcmc_multichain_jax(
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
) -> MultiChainMCMCResult:
    """Vectorised multi-chain random-walk Metropolis backend using JAX.

    Falls back to the NumPy backend when JAX is unavailable.
    """
    if n_chains < 1:
        raise ValueError("n_chains must be >= 1")

    info = jax_backend_info()
    if not info.available or jax is None or jnp is None:
        fallback = run_mcmc_multichain(
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
        fallback.used_fallback = True
        return fallback

    n_params = int(np.asarray(theta0).size)
    base_sigma_np = np.asarray(proposal_sigma, dtype=float)
    bounds_np = np.asarray(bounds, dtype=float)

    if adapt_steps is None:
        adapt_steps = max(1, n_steps // 5)
    if target_accept is None:
        target_accept = _default_target_accept(n_params)

    # Deterministic chain starts generated on host, then moved to JAX device.
    rng = np.random.default_rng(seed)
    starts = np.empty((n_chains, n_params), dtype=float)
    for i in range(n_chains):
        jitter = rng.normal(0.0, base_sigma_np * init_scatter)
        starts[i] = np.asarray(theta0, dtype=float) + jitter
        starts[i] = _reflect_into_bounds(starts[i], bounds_np)

    key = jax.random.PRNGKey(seed)
    current = jnp.asarray(starts)
    bounds_j = jnp.asarray(bounds_np)

    logpost = _logpost_factory(datasets, model, bounds_np, rd, include_planck)
    current_ll = np.array([logpost(starts[i]) for i in range(n_chains)], dtype=float)

    chains = np.empty((n_chains, n_steps, n_params), dtype=float)
    lls = np.empty((n_chains, n_steps), dtype=float)
    accepted = np.zeros(n_chains, dtype=int)
    accepted_window = np.zeros(n_chains, dtype=int)

    base_sigma_j = jnp.asarray(base_sigma_np)
    current_sigma = jnp.broadcast_to(base_sigma_j, (n_chains, n_params))
    log_scale = np.zeros(n_chains, dtype=float)

    for i in range(n_steps):
        key, k_prop, k_u = jax.random.split(key, 3)
        eps = jax.random.normal(k_prop, shape=(n_chains, n_params))
        proposal = current + current_sigma * eps
        if reflect_bounds:
            proposal = _reflect_into_bounds_jax(proposal, bounds_j)

        proposal_np = np.asarray(proposal)
        prop_ll = np.array([logpost(proposal_np[c]) for c in range(n_chains)], dtype=float)
        log_alpha = prop_ll - current_ll

        u = np.asarray(jax.random.uniform(k_u, shape=(n_chains,)))
        accept = np.log(u) < log_alpha

        current = jnp.where(jnp.asarray(accept)[:, None], proposal, current)
        current_ll = np.where(accept, prop_ll, current_ll)
        accepted += accept.astype(int)
        accepted_window += accept.astype(int)

        chains[:, i, :] = np.asarray(current)
        lls[:, i] = current_ll

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
            scale = np.exp(log_scale)[:, None]
            sigma_np = np.clip(
                base_sigma_np[None, :] * scale,
                base_sigma_np[None, :] * proposal_scale_min,
                base_sigma_np[None, :] * proposal_scale_max,
            )
            current_sigma = jnp.asarray(sigma_np)
            accepted_window[:] = 0

    burn = max(1, n_steps // 5)
    post = chains[:, burn:, :]

    rhat = _rhat_split(post) if (n_chains >= 2 and post.shape[1] >= 4) else None
    ess = _aggregate_ess(post) if (compute_ess and post.shape[1] >= 4) else None
    sigma_final = np.mean(np.asarray(current_sigma), axis=0)

    return MultiChainMCMCResult(
        chains=chains,
        loglike=lls,
        accept_rates=accepted / max(1, n_steps),
        param_names=_param_names(model),
        rhat_per_param=rhat,
        ess_per_param=ess,
        proposal_sigma_final=sigma_final,
        backend="jax",
        device=info.device,
        used_fallback=False,
    )
