"""Multi-probe analysis demo with real datasets.

Loads Pantheon+ SN Ia, DESI/eBOSS BAO, and Planck compressed priors,
evaluates log-likelihoods for ΛCDM and Model A at fiducial parameters,
then optionally runs a short pilot MCMC.

Usage:
    python -m ultra_slow_de.demo_real_data [--mcmc] [--n-steps 200] [--backend numpy|jax]
"""

import argparse
import numpy as np

from .builtin_data import (
    load_all_bao,
    load_desi_dh,
    load_desi_dm,
    load_desi_dv,
    load_planck_compressed,
)
from .inference import joint_logposterior, loglike_for_dataset, predict_observable
from .ingest import load_pantheon_plus
from .model_b import ModelBParams
from .params import CosmoParams, ModelAParams
from .sampler import run_mcmc_backend, run_mcmc_multichain_backend


def _fiducial_cosmo() -> CosmoParams:
    h = 67.4 / 100.0
    omega_r = 4.176e-5 / h**2  # photons + 3.046 massless neutrinos
    return CosmoParams(h0=67.4, omega_m=0.315, omega_r=omega_r)


def _fiducial_model_a() -> ModelAParams:
    return ModelAParams(w0=-1.0, B=0.02, C=0.0, omega=2.0)


def evaluate_likelihoods(verbose: bool = True) -> dict[str, float]:
    """Evaluate ΛCDM and Model-A log-likelihoods on all real datasets."""
    cosmo = _fiducial_cosmo()
    ma = _fiducial_model_a()
    rd = 147.09  # Mpc, Planck 2018

    # --- Pantheon+ ---
    sn = load_pantheon_plus()

    # --- BAO ---
    bao_datasets = load_all_bao()

    # --- Assemble dataset lists ---
    all_datasets: list[tuple] = [(sn, "mb")] + bao_datasets

    ll_lcdm = joint_logposterior(
        all_datasets, cosmo, rd=rd, include_planck=True
    )
    ll_ma = joint_logposterior(
        all_datasets, cosmo, model_a=ma, rd=rd, include_planck=True
    )

    results = {
        "logL_lcdm": ll_lcdm,
        "logL_model_a": ll_ma,
        "delta_logL": ll_ma - ll_lcdm,
        "n_sn": len(sn.z),
        "n_bao_datasets": len(bao_datasets),
    }

    if verbose:
        print("=" * 60)
        print("Multi-probe likelihood evaluation (SN + BAO + Planck)")
        print("=" * 60)
        print(f"Pantheon+ SNe         : {results['n_sn']}")
        print(f"BAO datasets          : {results['n_bao_datasets']}")
        print(f"ln L (ΛCDM)           : {results['logL_lcdm']:.2f}")
        print(f"ln L (Model A)        : {results['logL_model_a']:.2f}")
        print(f"Δ ln L (A − ΛCDM)     : {results['delta_logL']:.2f}")
        print()

    return results


def run_pilot_mcmc(n_steps: int = 200, verbose: bool = True, backend: str = "numpy") -> dict:
    """Run a short ΛCDM pilot MCMC on SN + BAO + Planck."""
    sn = load_pantheon_plus()
    bao_datasets = load_all_bao()
    all_datasets = [(sn, "mb")] + bao_datasets

    theta0 = np.array([67.4, 0.315])  # h0, omega_m
    bounds = np.array([[60.0, 80.0], [0.1, 0.5]])
    proposal = np.array([0.5, 0.01])

    if verbose:
        print(f"Running ΛCDM pilot MCMC ({n_steps} steps) ...")

    result = run_mcmc_backend(
        datasets=all_datasets,
        model="lcdm",
        theta0=theta0,
        bounds=bounds,
        proposal_sigma=proposal,
        n_steps=n_steps,
        include_planck=True,
        backend=backend,
    )

    if verbose:
        burn = n_steps // 5
        chain_post = result.chain[burn:]
        print(f"Accept rate       : {result.accept_rate:.3f}")
        print(f"h0 (posterior)    : {chain_post[:, 0].mean():.2f} ± {chain_post[:, 0].std():.2f}")
        print(f"Ω_m (posterior)   : {chain_post[:, 1].mean():.4f} ± {chain_post[:, 1].std():.4f}")
        if result.ess_per_param is not None:
            ess_fmt = ", ".join(
                f"{name}={ess:.1f}" for name, ess in zip(result.param_names, result.ess_per_param)
            )
            print(f"ESS (post-burn)   : {ess_fmt}")
        if result.proposal_sigma_final is not None:
            sig_fmt = ", ".join(
                f"{name}={sig:.4g}" for name, sig in zip(result.param_names, result.proposal_sigma_final)
            )
            print(f"Final proposal σ  : {sig_fmt}")
        print(f"Backend/device    : {result.backend}/{result.device} (fallback={result.used_fallback})")

    return {"result": result}


def _sampler_setup(model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = model.lower()
    if m == "lcdm":
        theta0 = np.array([67.4, 0.315])
        bounds = np.array([[60.0, 80.0], [0.1, 0.5]])
        proposal = np.array([0.5, 0.01])
    elif m == "a":
        theta0 = np.array([67.4, 0.315, -1.0, 0.0, 0.0, 2.0])
        bounds = np.array([
            [60.0, 80.0],
            [0.1, 0.5],
            [-1.5, -0.5],
            [-0.3, 0.3],
            [-0.3, 0.3],
            [0.1, 5.0],
        ])
        proposal = np.array([0.6, 0.012, 0.03, 0.01, 0.01, 0.08])
    elif m == "b":
        theta0 = np.array([67.4, 0.315, 0.3])
        bounds = np.array([[60.0, 80.0], [0.1, 0.5], [0.01, 5.0]])
        proposal = np.array([0.6, 0.012, 0.06])
    else:
        raise ValueError(f"Unsupported model: {model}")
    return theta0, bounds, proposal


def run_pilot_mcmc_model(
    model: str = "lcdm",
    n_steps: int = 200,
    verbose: bool = True,
    backend: str = "numpy",
) -> dict:
    """Run a short pilot MCMC for a selected model on SN + BAO + Planck."""
    sn = load_pantheon_plus()
    bao_datasets = load_all_bao()
    all_datasets = [(sn, "mb")] + bao_datasets
    theta0, bounds, proposal = _sampler_setup(model)

    if verbose:
        print(f"Running {model.upper()} pilot MCMC ({n_steps} steps) ...")

    result = run_mcmc_backend(
        datasets=all_datasets,
        model=model.lower(),
        theta0=theta0,
        bounds=bounds,
        proposal_sigma=proposal,
        n_steps=n_steps,
        include_planck=True,
        backend=backend,
    )

    if verbose:
        burn = n_steps // 5
        chain_post = result.chain[burn:]
        print(f"Accept rate       : {result.accept_rate:.3f}")
        means = np.mean(chain_post, axis=0)
        stds = np.std(chain_post, axis=0)
        for name, mu, sig in zip(result.param_names, means, stds):
            print(f"{name:>16s} : {mu:.5g} ± {sig:.3g}")
        if result.ess_per_param is not None:
            ess_fmt = ", ".join(
                f"{name}={ess:.1f}" for name, ess in zip(result.param_names, result.ess_per_param)
            )
            print(f"ESS (post-burn)   : {ess_fmt}")
        if result.proposal_sigma_final is not None:
            sig_fmt = ", ".join(
                f"{name}={sig:.4g}" for name, sig in zip(result.param_names, result.proposal_sigma_final)
            )
            print(f"Final proposal σ  : {sig_fmt}")
        print(f"Backend/device    : {result.backend}/{result.device} (fallback={result.used_fallback})")

    return {"result": result}


def run_pilot_mcmc_multichain(
    model: str = "lcdm",
    n_steps: int = 200,
    n_chains: int = 4,
    verbose: bool = True,
    backend: str = "numpy",
) -> dict:
    """Run a short multi-chain pilot MCMC with convergence diagnostics."""
    sn = load_pantheon_plus()
    bao_datasets = load_all_bao()
    all_datasets = [(sn, "mb")] + bao_datasets
    theta0, bounds, proposal = _sampler_setup(model)

    if verbose:
        print(f"Running {model.upper()} multi-chain pilot MCMC ({n_chains} chains × {n_steps} steps) ...")

    result = run_mcmc_multichain_backend(
        datasets=all_datasets,
        model=model.lower(),
        theta0=theta0,
        bounds=bounds,
        proposal_sigma=proposal,
        n_steps=n_steps,
        n_chains=n_chains,
        include_planck=True,
        backend=backend,
    )

    if verbose:
        burn = n_steps // 5
        post = result.chains[:, burn:, :]
        print(f"Accept rates      : {', '.join(f'{x:.3f}' for x in result.accept_rates)}")
        flat = post.reshape(-1, post.shape[-1])
        means = np.mean(flat, axis=0)
        stds = np.std(flat, axis=0)
        for name, mu, sig in zip(result.param_names, means, stds):
            print(f"{name:>16s} : {mu:.5g} ± {sig:.3g}")
        if result.rhat_per_param is not None:
            rhat_fmt = ", ".join(
                f"{name}={val:.4f}" for name, val in zip(result.param_names, result.rhat_per_param)
            )
            print(f"Split R-hat       : {rhat_fmt}")
        if result.ess_per_param is not None:
            ess_fmt = ", ".join(
                f"{name}={ess:.1f}" for name, ess in zip(result.param_names, result.ess_per_param)
            )
            print(f"ESS (aggregate)   : {ess_fmt}")
        if result.proposal_sigma_final is not None:
            sig_fmt = ", ".join(
                f"{name}={sig:.4g}" for name, sig in zip(result.param_names, result.proposal_sigma_final)
            )
            print(f"Final proposal σ  : {sig_fmt}")
        print(f"Backend/device    : {result.backend}/{result.device} (fallback={result.used_fallback})")

    return {"result": result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-data multi-probe demo")
    parser.add_argument("--mcmc", action="store_true", help="run pilot MCMC")
    parser.add_argument("--multi-chain", action="store_true", help="run multi-chain pilot MCMC")
    parser.add_argument("--model", choices=["lcdm", "a", "b"], default="lcdm")
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--backend", choices=["numpy", "jax"], default="numpy")
    args = parser.parse_args()

    evaluate_likelihoods()

    if args.mcmc:
        run_pilot_mcmc_model(model=args.model, n_steps=args.n_steps, backend=args.backend)
    if args.multi_chain:
        run_pilot_mcmc_multichain(
            model=args.model,
            n_steps=args.n_steps,
            n_chains=args.n_chains,
            backend=args.backend,
        )


if __name__ == "__main__":
    main()
