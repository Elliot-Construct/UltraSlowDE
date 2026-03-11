"""fσ₈ growth-rate sanity check.

Adds the eBOSS DR16 fσ₈(z) dataset to the standard SN+BAO+Planck likelihood
and reruns a quick MCMC sensitivity analysis for all three models.  Compares
posterior parameter constraints against the production (distance-only) run to
assess how much growth data shifts cosmological inferences.

Observable prediction uses the γ-parameterization:
    f(z) = [Ω_m(z)]^0.55   (Linder & Cahn 2007)
    σ₈(z) = σ₈,0 × D(z)/D(0)   (Heath-Carroll growth integral)
with σ₈,0 = 0.811 fixed (Planck 2018 TT,TE,EE+lowE).

This is a background-level check: dark-energy perturbation clustering and
sound-speed effects are not modelled.

Usage:
    python -m src.ultra_slow_de.fsig8_check
    python -m src.ultra_slow_de.fsig8_check --steps 20000
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .builtin_data import load_all_bao, load_eboss_fsig8
from .compressed_prior_check import compute_fde_at_recombination as _compute_fde_at_recombination
from .growth_backend import backend_availability
from .ingest import load_pantheon_plus
from .production_run import (
    SENSITIVITY_CFG,
    MODEL_LABEL,
    _credible_interval_68,
    _flatten_posteriors,
    _info_criteria,
    _posterior_draw_indices,
    _compute_per_point_loglike,
    compute_waic,
    run_single_model,
    N_DATA,
    K_BY_MODEL,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "output"


def compute_fde_at_recombination(
    chains: np.ndarray,
    param_names: list[str],
    model: str,
) -> dict[str, Any]:
    """Public wrapper for compressed-prior consistency checks."""
    return _compute_fde_at_recombination(chains=chains, param_names=param_names, model=model)


def run_fsig8_check(
    n_steps: int = 10_000,
    seed: int = 99,
    use_nuts: bool = True,
    float_sigma8: bool = True,
    growth_likelihood_mode: str = "production",
    growth_backend: str = "auto",
) -> dict[str, Any]:
    """Run SN+BAO+Planck+fσ₈ sensitivity MCMC for all three models.

    Parameters
    ----------
    float_sigma8 : bool
        If True (default), σ8,0 is treated as a free nuisance parameter with
        a Planck 2018 Gaussian prior N(0.811, 0.006²).  This neutralises the
        'fixed σ8 artefact' objection and turns the run into a credible
        robustness check.  Requires ``use_nuts=True``.
    """
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    sn = load_pantheon_plus()
    bao = load_all_bao()
    fsig8_ds = load_eboss_fsig8()

    # Base datasets: same as production
    datasets_base = [(sn, "mb")] + bao
    # Extended: add eBOSS fσ₈
    datasets_fsig8 = datasets_base + [(fsig8_ds, "fsig8")]

    # N_data for extended set (adds 2 fσ₈ points)
    n_data_fsig8 = N_DATA + len(fsig8_ds.y_obs)

    from .production_run import SamplerConfig
    mode_l = growth_likelihood_mode.lower()
    if mode_l == "production" and use_nuts:
        print("[fsig8 check] production growth mode requested; switching from NUTS to MH path for perturbation backend consistency.")
        use_nuts = False

    cfg = SamplerConfig(
        n_steps_primary=n_steps,
        n_steps_fallback=max(n_steps // 2, 5_000),
        burn_frac=0.25,
        adapt_steps=min(n_steps // 4, 5_000),
        adapt_interval=25,
        seed=seed,
        runtime_limit_min=120.0,
        use_nuts=use_nuts,
        float_sigma8=(float_sigma8 and use_nuts),
        growth_likelihood_mode=mode_l,
        growth_backend=growth_backend,
    )

    backend_status = backend_availability(cfg.growth_backend)

    sigma8_mode = "floated (Planck prior N(0.811, 0.006^2))" if (float_sigma8 and use_nuts) else "0.811 (fixed)"
    print("[fsig8 check] Running SN+BAO+Planck+fsig8 sensitivity MCMC ...")
    print(
        "  growth_likelihood_mode = "
        f"{cfg.growth_likelihood_mode} | backend_requested = {cfg.growth_backend} | "
        f"backend_resolved = {backend_status['backend_resolved']}"
    )
    print(f"  sigma8_0 = {sigma8_mode}  |  datasets: {n_data_fsig8} points")
    nuts_label = " NUTS all models" if use_nuts else ""
    print(f"  MCMC: {cfg.n_steps_primary} steps x 4 chains (burn 25%){nuts_label}")

    results_fsig8: dict[str, dict[str, Any]] = {}
    for model in ("lcdm", "a", "b"):
        print("[fsig8] Running " + MODEL_LABEL[model] + "...")
        results_fsig8[model] = run_single_model(
            model=model,
            datasets=datasets_fsig8,
            cfg=cfg,
            rd=147.09,
            run_tag="sensitivity_fsig8",
        )

    # ---------------------------------------------------------------------------
    # Compare with production (distance-only) posteriors
    # ---------------------------------------------------------------------------
    prod_path = OUTPUT_ROOT / "production_results.json"
    prod_results: dict[str, Any] | None = None
    if prod_path.exists():
        with open(prod_path, encoding="utf-8") as f:
            prod_raw = json.load(f)
        prod_results = prod_raw.get("production", None)

    print("\n" + "=" * 80)
    print("fsig8 sensitivity check -- parameter shifts (fsig8 run vs production distance-only)")
    print("=" * 80)
    headers = ["Model", "Param", "Dist-only (mean)", "+fsig8 (mean)", "Shift (sigma)"]
    print(f"  {'Model':<8}  {'Param':<10}  {'Dist-only':>14}  {'+fsig8 run':>14}  {'Shift/sigma':>12}")
    print("  " + "-" * 64)

    comparison: dict[str, Any] = {}
    for model in ("lcdm", "a", "b"):
        r_new = results_fsig8[model]
        comparison[model] = {}
        for p in r_new["param_names"]:
            stats_new = r_new["posterior"][p]
            row: dict[str, Any] = {"fsig8_mean": stats_new["mean"], "fsig8_std": stats_new["std"]}
            shift_str = "—"
            dist_str = "—"
            if prod_results and p in prod_results.get(model, {}).get("posterior", {}):
                stats_old = prod_results[model]["posterior"][p]
                dist_str = f"{stats_old['mean']:.4g}"
                combined_std = max(stats_old["std"], stats_new["std"])
                if combined_std > 0:
                    shift = (stats_new["mean"] - stats_old["mean"]) / combined_std
                    shift_str = f"{shift:+.2f}"
                row["dist_mean"] = stats_old["mean"]
                row["dist_std"] = stats_old["std"]
                row["shift_sigma"] = float(shift_str) if shift_str != "—" else None
            comparison[model][p] = row
            print(f"  {MODEL_LABEL[model]:<8}  {p:<10}  {dist_str:>14}  {stats_new['mean']:>14.4g}  {shift_str:>8}")
    print("=" * 80)

    # fσ8 predictions at posterior mean for each model
    print("\nfsig8 predictions at posterior mean (sigma8_0=0.811 fixed):")
    print(f"  {'z_eff':>8}  {'observed':>10}  {'LCDM':>10}  {'Model A':>10}  {'Model B':>10}")
    from .inference import predict_observable
    from .params import CosmoParams
    from .model_a import H_model_a
    from .model_b import ModelBParams

    z_fsig8 = fsig8_ds.z
    y_fsig8 = fsig8_ds.y_obs

    # Build model mean thetas from fsig8 run
    def _pred_fsig8(model: str, r: dict) -> np.ndarray:
        from .params import CosmoParams, ModelAParams
        from .model_b import ModelBParams
        p = r["param_names"]
        post = r["posterior"]
        theta = np.array([post[x]["mean"] for x in p])
        if model == "lcdm":
            cosmo = CosmoParams(h0=theta[0], omega_m=theta[1])
            return predict_observable(z_fsig8, "fsig8", cosmo)
        if model == "a":
            from .params import ModelAParams
            cosmo = CosmoParams(h0=theta[0], omega_m=theta[1])
            ma = ModelAParams(w0=theta[2], B=theta[3], C=theta[4], omega=theta[5])
            return predict_observable(z_fsig8, "fsig8", cosmo, model_a=ma)
        cosmo = CosmoParams(h0=theta[0], omega_m=theta[1])
        mb = ModelBParams(mu=theta[2])
        return predict_observable(z_fsig8, "fsig8", cosmo, model_b=mb)

    preds = {m: _pred_fsig8(m, results_fsig8[m]) for m in ("lcdm", "a", "b")}
    for i, zi in enumerate(z_fsig8):
        print(
            f"  {zi:>8.3f}  {y_fsig8[i]:>10.4f}  "
            f"{preds['lcdm'][i]:>10.4f}  {preds['a'][i]:>10.4f}  {preds['b'][i]:>10.4f}"
        )

    # Save JSON
    sigma8_mode_str = "floated_planck_prior" if (float_sigma8 and use_nuts) else "fixed_0.811"
    out = {
        "description": "SN+BAO+Planck+fsig8 sensitivity run",
        "growth_likelihood_mode": cfg.growth_likelihood_mode,
        "growth_backend_requested": cfg.growth_backend,
        "growth_backend_status": backend_status,
        "sigma8_0_treatment": sigma8_mode_str,
        "sigma8_0_prior": "N(0.811, 0.006^2)" if (float_sigma8 and use_nuts) else "fixed at 0.811",
        "fsig8_data": {"z": z_fsig8.tolist(), "y_obs": y_fsig8.tolist(),
                       "sigma": np.sqrt(np.diag(fsig8_ds.cov)).tolist()},
        "results": _strip_for_json(results_fsig8),
        "comparison": comparison,
        "fsig8_predictions_at_posterior_mean": {m: preds[m].tolist() for m in preds},
    }
    out_path = OUTPUT_ROOT / "fsig8_check_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_path}")
    return out


def _strip_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_for_json(v) for k, v in obj.items() if k not in ("chains", "loglike")}
    if isinstance(obj, (list, tuple)):
        return [_strip_for_json(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="fsig8 growth-rate sanity check MCMC.")
    p.add_argument("--steps", type=int, default=10_000,
                   help="MCMC steps per chain (default: 10000)")
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--nuts", action="store_true", default=True,
                   help="Use NUTS for all models (default: True)")
    p.add_argument("--no-nuts", dest="nuts", action="store_false",
                   help="Disable NUTS, use random-walk MH instead")
    p.add_argument("--no-float-sigma8", dest="float_sigma8", action="store_false",
                   default=True,
                   help="Fix sigma8_0=0.811 instead of floating it as a nuisance parameter")
    p.add_argument(
        "--growth-mode",
        type=str,
        default="production",
        choices=["production", "exploratory_gamma"],
        help="Growth-likelihood mode: production (perturbation backend) or exploratory_gamma",
    )
    p.add_argument(
        "--growth-backend",
        type=str,
        default="auto",
        choices=["auto", "class", "camb"],
        help="Requested perturbation backend adapter for production growth mode",
    )
    args = p.parse_args()
    run_fsig8_check(n_steps=args.steps, seed=args.seed, use_nuts=args.nuts,
                    float_sigma8=args.float_sigma8,
                    growth_likelihood_mode=args.growth_mode,
                    growth_backend=args.growth_backend)
